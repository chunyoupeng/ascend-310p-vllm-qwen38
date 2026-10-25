#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sched.h>
#include <omp.h>
#if defined(__aarch64__)
#include <arm_neon.h>
#endif

#define MAX_RANKS 8
#define SLOT_SIZE (32UL * 1024UL * 1024UL)
// The all-rank wait_arrive barrier already orders step N against step N-1, so
// two ring slots suffice for correctness; 4 keeps margin while leaving the whole
// ring L3-resident (Kunpeng-920: 32 MiB L3 per socket, 128 B lines).
#define RING_SLOTS 4
// Must cover the largest tensor the decode graph reduces, or the capture
// callback silently degrades to the 3-barrier in_buf path. hidden=5120 fp16 at
// cudagraph_capture_sizes=32 (max_num_seqs 8 x MTP 1+3) is 320 KiB.
#define RING_SLOT_BYTES (512UL * 1024UL)

typedef struct __attribute__((aligned(128))) {
    volatile uint64_t init_seq;
    volatile uint64_t arrive_seq;
    volatile uint64_t mid_seq;
    volatile uint64_t done_seq;
    uint8_t pad[128 - 32];
} RankSync;

typedef struct {
    RankSync rank_sync[MAX_RANKS];
    uint8_t pad[4096 - sizeof(RankSync) * MAX_RANKS];
    uint8_t __attribute__((aligned(128))) ring_buf[RING_SLOTS][MAX_RANKS][RING_SLOT_BYTES];
    uint8_t __attribute__((aligned(128))) in_buf[MAX_RANKS][SLOT_SIZE];
    uint8_t __attribute__((aligned(128))) out_buf[SLOT_SIZE * MAX_RANKS];
} ShmRegion;

static_assert(sizeof(RankSync) == 128, "RankSync must be exactly one 128 B L3 line");
static_assert(sizeof(RankSync) * MAX_RANKS < 4096, "rank_sync[] must leave room for pad[]");
static_assert((RING_SLOTS & (RING_SLOTS - 1)) == 0, "RING_SLOTS must be a power of two");

typedef void (*aclrtHostFunc)(void *args);

// Internal torch_npu symbol to drain its C++ async task queue
namespace c10_npu {
struct NPUStream {
    uint64_t data[2];
    ~NPUStream() {}
    void* stream(bool need_empty) const;
};
NPUStream getCurrentNPUStream(signed char device_index);
}

static inline void* flush_npu_stream() {
    c10_npu::NPUStream s = c10_npu::getCurrentNPUStream(-1);
    return s.stream(true);
}

extern "C" {

extern int aclrtSynchronizeStream(void *stream);
extern int aclrtMemcpy(void *dst, size_t destMax, const void *src, size_t count, int kind);
extern int aclrtMemcpyAsync(void *dst, size_t destMax, const void *src, size_t count, int kind, void *stream);
extern int aclrtMallocHost(void **hostPtr, size_t size);
extern int aclrtFreeHost(void *hostPtr);
extern int aclrtLaunchHostFunc(void *stream, aclrtHostFunc fn, void *args);
extern int aclrtGetCurrentContext(void **context);
extern int aclrtSetCurrentContext(void *context);

#define ACL_MEMCPY_HOST_TO_HOST 0
#define ACL_MEMCPY_HOST_TO_DEVICE 1
#define ACL_MEMCPY_DEVICE_TO_HOST 2
#define ACL_MEMCPY_DEVICE_TO_DEVICE 3

static inline void wait_init(RankSync *rs, int world_size, uint64_t step) {
    for (int i = 0; i < world_size; i++) {
        uint64_t spins = 0;
        while (__atomic_load_n(&rs[i].init_seq, __ATOMIC_ACQUIRE) < step) {
            if (++spins > 200000) {
                sched_yield();
                spins = 0;
            } else {
                __asm__ volatile("yield" ::: "memory");
            }
        }
    }
}

static inline void wait_arrive(RankSync *rs, int world_size, uint64_t step) {
    for (int i = 0; i < world_size; i++) {
        uint64_t spins = 0;
        while (__atomic_load_n(&rs[i].arrive_seq, __ATOMIC_ACQUIRE) < step) {
            if (++spins > 200000) {
                sched_yield();
                spins = 0;
            } else {
                __asm__ volatile("yield" ::: "memory");
            }
        }
    }
}

static inline void wait_mid(RankSync *rs, int world_size, uint64_t step) {
    for (int i = 0; i < world_size; i++) {
        uint64_t spins = 0;
        while (__atomic_load_n(&rs[i].mid_seq, __ATOMIC_ACQUIRE) < step) {
            if (++spins > 200000) {
                sched_yield();
                spins = 0;
            } else {
                __asm__ volatile("yield" ::: "memory");
            }
        }
    }
}

static inline void wait_done(RankSync *rs, int world_size, uint64_t step) {
    for (int i = 0; i < world_size; i++) {
        uint64_t spins = 0;
        while (__atomic_load_n(&rs[i].done_seq, __ATOMIC_ACQUIRE) < step) {
            if (++spins > 200000) {
                sched_yield();
                spins = 0;
            } else {
                __asm__ volatile("yield" ::: "memory");
            }
        }
    }
}

// NOTE: ShmGraphTask allocations below are deliberately never freed. The
// recorded NPUGraph node holds the pointer and every graph.replay() dereferences
// it, so freeing it in the callback would hand the second replay a dangling
// pointer. Bounded by capture_sizes x collectives per process (tens of KiB).
typedef struct {
    ShmRegion *region;
    void *host_bounce;
    void *acl_ctx;
    size_t bounce_size;
    uint64_t step;
    int world_size;
    int rank;
} ShmComm;

static __thread void *tls_acl_ctx = NULL;
// aclrtLaunchHostFunc callbacks run on a CANN runtime thread, not the vLLM
// worker thread, so they do not inherit its ACL context.
// Measured and rejected: pinning this thread to the NPU-local socket via
// sched_setaffinity was slower at every size but 320 KiB.
static inline void ensure_acl_ctx(ShmComm *comm) {
    if (comm->acl_ctx != NULL && tls_acl_ctx != comm->acl_ctx) {
        aclrtSetCurrentContext(comm->acl_ctx);
        tls_acl_ctx = comm->acl_ctx;
    }
}

// A collective too large for the capture path falls through to the synchronous
// branch, which calls aclrtSynchronizeStream on a capturing stream and silently
// corrupts the graph. Make that loud rather than mysterious.
static void warn_capture_unsupported(const char *fn, size_t bytes) {
    static int warned = 0;
    if (__atomic_exchange_n(&warned, 1, __ATOMIC_SEQ_CST)) return;
    fprintf(stderr,
            "[shm_hccl] %s: %zu bytes while the stream is capturing, but the "
            "graph path accepts at most %zu and broadcast/gather have no graph "
            "path at all. The synchronous fallback will corrupt the captured "
            "graph.\n",
            fn, bytes, (size_t)SLOT_SIZE);
    fflush(stderr);
}

ShmComm* shm_comm_init(const char *name, int world_size, int rank, int create_fresh) {
    char path[256];
    snprintf(path, sizeof(path), "/dev/shm/vllm_shm_%s", name);
    int fd = open(path, O_RDWR | O_CREAT, 0666);
    if (fd < 0) return NULL;
    size_t total_size = sizeof(ShmRegion);
    if (ftruncate(fd, total_size) != 0) {
        close(fd);
        return NULL;
    }
    void *ptr = mmap(NULL, total_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (ptr == MAP_FAILED) return NULL;

    ShmComm *comm = (ShmComm*)malloc(sizeof(ShmComm));
    comm->region = (ShmRegion*)ptr;
    comm->step = 0;
    comm->world_size = world_size;
    comm->rank = rank;
    comm->bounce_size = SLOT_SIZE * (size_t)world_size;
    comm->host_bounce = NULL;
    comm->acl_ctx = NULL;
    aclrtGetCurrentContext(&comm->acl_ctx);
    if (posix_memalign(&comm->host_bounce, 128, comm->bounce_size) != 0) comm->host_bounce = NULL;


    comm->region->rank_sync[rank].arrive_seq = 0;
    comm->region->rank_sync[rank].mid_seq = 0;
    comm->region->rank_sync[rank].done_seq = 0;
    uint64_t next_init = __atomic_load_n(&comm->region->rank_sync[rank].init_seq, __ATOMIC_SEQ_CST) + 1;
    __atomic_store_n(&comm->region->rank_sync[rank].init_seq, next_init, __ATOMIC_SEQ_CST);
    wait_init(comm->region->rank_sync, world_size, next_init);

    if (rank == 0 && create_fresh) {
        unlink(path);
    }
    return comm;
}

void shm_comm_destroy(ShmComm *comm) {
    if (!comm) return;
    if (comm->host_bounce) {
        free(comm->host_bounce);
        comm->host_bounce = NULL;
    }
    if (comm->region) {
        munmap(comm->region, sizeof(ShmRegion));
        comm->region = NULL;
    }
    free(comm);
}

int shm_barrier(ShmComm *comm) {
    if (!comm) return 0;
    ShmRegion *reg = comm->region;
    int rank = comm->rank;
    int ws = comm->world_size;
    uint64_t step = __atomic_add_fetch(&comm->step, 1, __ATOMIC_SEQ_CST);
    __atomic_store_n(&reg->rank_sync[rank].arrive_seq, step, __ATOMIC_RELEASE);
    wait_arrive(reg->rank_sync, ws, step);
    return 0;
}

static size_t dtype_size(int dtype) {
    switch (dtype) {
        case 0: return 2; // float16
        case 1: return 4; // float32
        case 2: return 8; // int64
        case 3: return 4; // int32
        case 4: return 1; // int8
        case 5: return 1; // uint8 / bool
        case 6: return 2; // bfloat16
        default: return 1;
    }
}

static inline float bf16_to_fp32(uint16_t u) {
    uint32_t bits = ((uint32_t)u) << 16;
    float f;
    memcpy(&f, &bits, sizeof(float));
    return f;
}

static inline uint16_t fp32_to_bf16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(float));
    uint32_t rounding_bias = ((bits >> 16) & 1) + 0x7FFF;
    return (uint16_t)((bits + rounding_bias) >> 16);
}

// Fast ring reduction directly into dst_ptr[0 .. (end_elem - start_elem)]
static inline void reduce_ring_to_dst(
    uint8_t *dst_ptr,
    uint8_t slot_buf[MAX_RANKS][RING_SLOT_BYTES],
    int world_size,
    size_t start_elem,
    size_t end_elem,
    int dtype,
    int op
) {
    if (start_elem >= end_elem) return;
    size_t count = end_elem - start_elem;

#if defined(__aarch64__)
    if (dtype == 0 && op == 0) { // float16 SUM
        float16_t *dst = (float16_t*)dst_ptr;
        if (world_size == 4) {
            const float16_t *s0 = (const float16_t*)slot_buf[0] + start_elem;
            const float16_t *s1 = (const float16_t*)slot_buf[1] + start_elem;
            const float16_t *s2 = (const float16_t*)slot_buf[2] + start_elem;
            const float16_t *s3 = (const float16_t*)slot_buf[3] + start_elem;
            size_t i = 0;
            for (; i + 8 <= count; i += 8) {
                float16x8_t v0 = vld1q_f16(s0 + i);
                float16x8_t v1 = vld1q_f16(s1 + i);
                float16x8_t v2 = vld1q_f16(s2 + i);
                float16x8_t v3 = vld1q_f16(s3 + i);
                float32x4_t lo = vaddq_f32(
                    vaddq_f32(vcvt_f32_f16(vget_low_f16(v0)), vcvt_f32_f16(vget_low_f16(v1))),
                    vaddq_f32(vcvt_f32_f16(vget_low_f16(v2)), vcvt_f32_f16(vget_low_f16(v3)))
                );
                float32x4_t hi = vaddq_f32(
                    vaddq_f32(vcvt_f32_f16(vget_high_f16(v0)), vcvt_f32_f16(vget_high_f16(v1))),
                    vaddq_f32(vcvt_f32_f16(vget_high_f16(v2)), vcvt_f32_f16(vget_high_f16(v3)))
                );
                vst1q_f16(dst + i, vcombine_f16(vcvt_f16_f32(lo), vcvt_f16_f32(hi)));
            }
            for (; i < count; i++) {
                float acc = (float)s0[i] + (float)s1[i] + (float)s2[i] + (float)s3[i];
                dst[i] = (float16_t)acc;
            }
            return;
        } else if (world_size == 2) {
            const float16_t *s0 = (const float16_t*)slot_buf[0] + start_elem;
            const float16_t *s1 = (const float16_t*)slot_buf[1] + start_elem;
            size_t i = 0;
            for (; i + 8 <= count; i += 8) {
                float16x8_t v0 = vld1q_f16(s0 + i);
                float16x8_t v1 = vld1q_f16(s1 + i);
                float32x4_t lo = vaddq_f32(vcvt_f32_f16(vget_low_f16(v0)), vcvt_f32_f16(vget_low_f16(v1)));
                float32x4_t hi = vaddq_f32(vcvt_f32_f16(vget_high_f16(v0)), vcvt_f32_f16(vget_high_f16(v1)));
                vst1q_f16(dst + i, vcombine_f16(vcvt_f16_f32(lo), vcvt_f16_f32(hi)));
            }
            for (; i < count; i++) {
                dst[i] = (float16_t)((float)s0[i] + (float)s1[i]);
            }
            return;
        }
    } else if (dtype == 1 && op == 0 && world_size == 4) { // float32 SUM
        float *dst = (float*)dst_ptr;
        const float *s0 = (const float*)slot_buf[0] + start_elem;
        const float *s1 = (const float*)slot_buf[1] + start_elem;
        const float *s2 = (const float*)slot_buf[2] + start_elem;
        const float *s3 = (const float*)slot_buf[3] + start_elem;
        size_t i = 0;
        for (; i + 4 <= count; i += 4) {
            float32x4_t v = vaddq_f32(
                vaddq_f32(vld1q_f32(s0 + i), vld1q_f32(s1 + i)),
                vaddq_f32(vld1q_f32(s2 + i), vld1q_f32(s3 + i))
            );
            vst1q_f32(dst + i, v);
        }
        for (; i < count; i++) {
            dst[i] = s0[i] + s1[i] + s2[i] + s3[i];
        }
        return;
    }
#endif

    if (dtype == 0) { // float16
        __fp16 *dst = (__fp16*)dst_ptr;
        for (size_t idx = 0; idx < count; idx++) {
            size_t i = start_elem + idx;
            float acc = (float)(((__fp16*)slot_buf[0])[i]);
            for (int r = 1; r < world_size; r++) {
                float v = (float)(((__fp16*)slot_buf[r])[i]);
                if (op == 0) acc += v;
                else if (op == 1) { if (v > acc) acc = v; }
                else if (op == 2) { if (v < acc) acc = v; }
                else if (op == 3) acc *= v;
            }
            dst[idx] = (__fp16)acc;
        }
    } else if (dtype == 1) { // float32
        float *dst = (float*)dst_ptr;
        for (size_t idx = 0; idx < count; idx++) {
            size_t i = start_elem + idx;
            float acc = ((float*)slot_buf[0])[i];
            for (int r = 1; r < world_size; r++) {
                float v = ((float*)slot_buf[r])[i];
                if (op == 0) acc += v;
                else if (op == 1) { if (v > acc) acc = v; }
                else if (op == 2) { if (v < acc) acc = v; }
                else if (op == 3) acc *= v;
            }
            dst[idx] = acc;
        }
    } else if (dtype == 2) { // int64
        int64_t *dst = (int64_t*)dst_ptr;
        for (size_t idx = 0; idx < count; idx++) {
            size_t i = start_elem + idx;
            int64_t acc = ((int64_t*)slot_buf[0])[i];
            for (int r = 1; r < world_size; r++) {
                int64_t v = ((int64_t*)slot_buf[r])[i];
                if (op == 0) acc += v;
                else if (op == 1) { if (v > acc) acc = v; }
                else if (op == 2) { if (v < acc) acc = v; }
                else if (op == 3) acc *= v;
            }
            dst[idx] = acc;
        }
    } else if (dtype == 3) { // int32
        int32_t *dst = (int32_t*)dst_ptr;
        for (size_t idx = 0; idx < count; idx++) {
            size_t i = start_elem + idx;
            int32_t acc = ((int32_t*)slot_buf[0])[i];
            for (int r = 1; r < world_size; r++) {
                int32_t v = ((int32_t*)slot_buf[r])[i];
                if (op == 0) acc += v;
                else if (op == 1) { if (v > acc) acc = v; }
                else if (op == 2) { if (v < acc) acc = v; }
                else if (op == 3) acc *= v;
            }
            dst[idx] = acc;
        }
    } else if (dtype == 4 || dtype == 5) { // int8 / uint8
        uint8_t *dst = (uint8_t*)dst_ptr;
        for (size_t idx = 0; idx < count; idx++) {
            size_t i = start_elem + idx;
            uint8_t acc = slot_buf[0][i];
            for (int r = 1; r < world_size; r++) {
                uint8_t v = slot_buf[r][i];
                if (op == 0) acc += v;
                else if (op == 1) { if (v > acc) acc = v; }
                else if (op == 2) { if (v < acc) acc = v; }
            }
            dst[idx] = acc;
        }
    } else if (dtype == 6) { // bfloat16
        uint16_t *dst = (uint16_t*)dst_ptr;
        for (size_t idx = 0; idx < count; idx++) {
            size_t i = start_elem + idx;
            float acc = bf16_to_fp32(((uint16_t*)slot_buf[0])[i]);
            for (int r = 1; r < world_size; r++) {
                float v = bf16_to_fp32(((uint16_t*)slot_buf[r])[i]);
                if (op == 0) acc += v;
                else if (op == 1) { if (v > acc) acc = v; }
                else if (op == 2) { if (v < acc) acc = v; }
                else if (op == 3) acc *= v;
            }
            dst[idx] = fp32_to_bf16(acc);
        }
    }
}

static void reduce_slice(
    uint8_t *out_ptr,
    uint8_t in_buf[MAX_RANKS][SLOT_SIZE],
    int world_size,
    size_t start_elem,
    size_t end_elem,
    int dtype,
    int op
) {
    if (start_elem >= end_elem) return;
    size_t count = end_elem - start_elem;
    // Prefill reduces ~8 M elements per slice; 4 threads leaves most of the
    // socket idle. Decode stays single-threaded so a 10 K tensor never pays for
    // a thread-pool wakeup.
    int nthreads = 1;
    if (count >= 4194304) nthreads = 16;
    else if (count >= 1048576) nthreads = 8;
    else if (count >= 262144) nthreads = 4;

#if defined(__aarch64__)
    // reduce_ring_to_dst has a NEON kernel but this function did not, so every
    // reduction above RING_SLOT_BYTES -- i.e. all of prefill -- ran scalar with
    // the op test in the innermost loop.
    if (op == 0 && dtype == 0) { // float16 SUM
        float16_t *dst = (float16_t*)out_ptr;
        #pragma omp parallel for schedule(static) num_threads(nthreads) if(nthreads > 1)
        for (size_t blk = 0; blk < count; blk += 8) {
            size_t i = start_elem + blk;
            size_t rem = count - blk;
            if (rem >= 8) {
                float32x4_t lo = vdupq_n_f32(0.0f);
                float32x4_t hi = vdupq_n_f32(0.0f);
                for (int r = 0; r < world_size; r++) {
                    float16x8_t v = vld1q_f16((const float16_t*)in_buf[r] + i);
                    lo = vaddq_f32(lo, vcvt_f32_f16(vget_low_f16(v)));
                    hi = vaddq_f32(hi, vcvt_f32_f16(vget_high_f16(v)));
                }
                vst1q_f16(dst + i, vcombine_f16(vcvt_f16_f32(lo), vcvt_f16_f32(hi)));
            } else {
                for (size_t k = 0; k < rem; k++) {
                    float acc = 0.0f;
                    for (int r = 0; r < world_size; r++) acc += (float)(((const float16_t*)in_buf[r])[i + k]);
                    dst[i + k] = (float16_t)acc;
                }
            }
        }
        return;
    }
    if (op == 0 && dtype == 1) { // float32 SUM
        float *dst = (float*)out_ptr;
        #pragma omp parallel for schedule(static) num_threads(nthreads) if(nthreads > 1)
        for (size_t blk = 0; blk < count; blk += 4) {
            size_t i = start_elem + blk;
            size_t rem = count - blk;
            if (rem >= 4) {
                float32x4_t acc = vdupq_n_f32(0.0f);
                for (int r = 0; r < world_size; r++) acc = vaddq_f32(acc, vld1q_f32((const float*)in_buf[r] + i));
                vst1q_f32(dst + i, acc);
            } else {
                for (size_t k = 0; k < rem; k++) {
                    float a = 0.0f;
                    for (int r = 0; r < world_size; r++) a += ((const float*)in_buf[r])[i + k];
                    dst[i + k] = a;
                }
            }
        }
        return;
    }
#endif

    if (dtype == 0) { // float16
        __fp16 *dst = (__fp16*)out_ptr;
        #pragma omp parallel for schedule(static) num_threads(nthreads) if(count >= 262144)
        for (size_t i = start_elem; i < end_elem; i++) {
            float acc = (float)(((__fp16*)in_buf[0])[i]);
            for (int r = 1; r < world_size; r++) {
                float v = (float)(((__fp16*)in_buf[r])[i]);
                if (op == 0) acc += v;
                else if (op == 1) { if (v > acc) acc = v; }
                else if (op == 2) { if (v < acc) acc = v; }
                else if (op == 3) acc *= v;
            }
            dst[i] = (__fp16)acc;
        }
    } else if (dtype == 1) { // float32
        float *dst = (float*)out_ptr;
        #pragma omp parallel for schedule(static) num_threads(nthreads) if(count >= 262144)
        for (size_t i = start_elem; i < end_elem; i++) {
            float acc = ((float*)in_buf[0])[i];
            for (int r = 1; r < world_size; r++) {
                float v = ((float*)in_buf[r])[i];
                if (op == 0) acc += v;
                else if (op == 1) { if (v > acc) acc = v; }
                else if (op == 2) { if (v < acc) acc = v; }
                else if (op == 3) acc *= v;
            }
            dst[i] = acc;
        }
    } else if (dtype == 2) { // int64
        int64_t *dst = (int64_t*)out_ptr;
        #pragma omp parallel for schedule(static) num_threads(nthreads) if(count >= 262144)
        for (size_t i = start_elem; i < end_elem; i++) {
            int64_t acc = ((int64_t*)in_buf[0])[i];
            for (int r = 1; r < world_size; r++) {
                int64_t v = ((int64_t*)in_buf[r])[i];
                if (op == 0) acc += v;
                else if (op == 1) { if (v > acc) acc = v; }
                else if (op == 2) { if (v < acc) acc = v; }
                else if (op == 3) acc *= v;
            }
            dst[i] = acc;
        }
    } else if (dtype == 3) { // int32
        int32_t *dst = (int32_t*)out_ptr;
        #pragma omp parallel for schedule(static) num_threads(nthreads) if(count >= 262144)
        for (size_t i = start_elem; i < end_elem; i++) {
            int32_t acc = ((int32_t*)in_buf[0])[i];
            for (int r = 1; r < world_size; r++) {
                int32_t v = ((int32_t*)in_buf[r])[i];
                if (op == 0) acc += v;
                else if (op == 1) { if (v > acc) acc = v; }
                else if (op == 2) { if (v < acc) acc = v; }
                else if (op == 3) acc *= v;
            }
            dst[i] = acc;
        }
    } else if (dtype == 4 || dtype == 5) { // int8 / uint8
        uint8_t *dst = (uint8_t*)out_ptr;
        for (size_t i = start_elem; i < end_elem; i++) {
            uint8_t acc = in_buf[0][i];
            for (int r = 1; r < world_size; r++) {
                uint8_t v = in_buf[r][i];
                if (op == 0) acc += v;
                else if (op == 1) { if (v > acc) acc = v; }
                else if (op == 2) { if (v < acc) acc = v; }
            }
            dst[i] = acc;
        }
    } else if (dtype == 6) { // bfloat16
        uint16_t *dst = (uint16_t*)out_ptr;
        #pragma omp parallel for schedule(static) num_threads(nthreads) if(count >= 262144)
        for (size_t i = start_elem; i < end_elem; i++) {
            float acc = bf16_to_fp32(((uint16_t*)in_buf[0])[i]);
            for (int r = 1; r < world_size; r++) {
                float v = bf16_to_fp32(((uint16_t*)in_buf[r])[i]);
                if (op == 0) acc += v;
                else if (op == 1) { if (v > acc) acc = v; }
                else if (op == 2) { if (v < acc) acc = v; }
                else if (op == 3) acc *= v;
            }
            dst[i] = fp32_to_bf16(acc);
        }
    }
}

typedef struct {
    ShmComm *comm;
    const void *in_dev_ptr;
    void *out_dev_ptr;
    size_t numel;
    size_t bytes;
    int dtype;
    int op;
} ShmGraphTask;

static void host_all_reduce_cb(void *arg) {
    ShmGraphTask *t = (ShmGraphTask*)arg;
    ShmComm *comm = t->comm;
    ensure_acl_ctx(comm);
    int rank = comm->rank;
    int ws = comm->world_size;
    ShmRegion *reg = comm->region;
    size_t chunk_elems = t->numel;
    size_t chunk_bytes = t->bytes;

    uint64_t step = __atomic_add_fetch(&comm->step, 1, __ATOMIC_SEQ_CST);
    if (chunk_bytes <= RING_SLOT_BYTES) {
        uint8_t (*slot_buf)[RING_SLOT_BYTES] = reg->ring_buf[step & (RING_SLOTS - 1)];
        aclrtMemcpy(slot_buf[rank], chunk_bytes, t->in_dev_ptr, chunk_bytes, ACL_MEMCPY_DEVICE_TO_HOST);
        __atomic_store_n(&reg->rank_sync[rank].arrive_seq, step, __ATOMIC_RELEASE);
        wait_arrive(reg->rank_sync, ws, step);
        reduce_ring_to_dst((uint8_t*)comm->host_bounce, slot_buf, ws, 0, chunk_elems, t->dtype, t->op);
        aclrtMemcpy(t->out_dev_ptr, chunk_bytes, comm->host_bounce, chunk_bytes, ACL_MEMCPY_HOST_TO_DEVICE);
        return;
    }

    aclrtMemcpy(reg->in_buf[rank], chunk_bytes, t->in_dev_ptr, chunk_bytes, ACL_MEMCPY_DEVICE_TO_HOST);

    __atomic_store_n(&reg->rank_sync[rank].arrive_seq, step, __ATOMIC_RELEASE);
    wait_arrive(reg->rank_sync, ws, step);

    size_t per_rank = (chunk_elems + ws - 1) / ws;
    size_t s = rank * per_rank;
    size_t e = s + per_rank;
    if (s > chunk_elems) s = chunk_elems;
    if (e > chunk_elems) e = chunk_elems;

    reduce_slice(reg->out_buf, reg->in_buf, ws, s, e, t->dtype, t->op);

    __atomic_store_n(&reg->rank_sync[rank].mid_seq, step, __ATOMIC_RELEASE);
    wait_mid(reg->rank_sync, ws, step);

    aclrtMemcpy(t->out_dev_ptr, chunk_bytes, reg->out_buf, chunk_bytes, ACL_MEMCPY_HOST_TO_DEVICE);

    __atomic_store_n(&reg->rank_sync[rank].done_seq, step, __ATOMIC_RELEASE);
    wait_done(reg->rank_sync, ws, step);
}

static void host_all_gather_cb(void *arg) {
    ShmGraphTask *t = (ShmGraphTask*)arg;
    ShmComm *comm = t->comm;
    ensure_acl_ctx(comm);
    int rank = comm->rank;
    int ws = comm->world_size;
    ShmRegion *reg = comm->region;
    size_t in_bytes = t->bytes;
    size_t out_bytes = in_bytes * (size_t)ws;

    uint64_t step = __atomic_add_fetch(&comm->step, 1, __ATOMIC_SEQ_CST);
    if (in_bytes <= RING_SLOT_BYTES) {
        uint8_t (*slot_buf)[RING_SLOT_BYTES] = reg->ring_buf[step & (RING_SLOTS - 1)];
        aclrtMemcpy(slot_buf[rank], in_bytes, t->in_dev_ptr, in_bytes, ACL_MEMCPY_DEVICE_TO_HOST);
        __atomic_store_n(&reg->rank_sync[rank].arrive_seq, step, __ATOMIC_RELEASE);
        wait_arrive(reg->rank_sync, ws, step);
        for (int r = 0; r < ws; r++) {
            aclrtMemcpy((uint8_t*)t->out_dev_ptr + (size_t)r * in_bytes, in_bytes, slot_buf[r], in_bytes, ACL_MEMCPY_HOST_TO_DEVICE);
        }
        return;
    }

    aclrtMemcpy(reg->in_buf[rank], in_bytes, t->in_dev_ptr, in_bytes, ACL_MEMCPY_DEVICE_TO_HOST);

    __atomic_store_n(&reg->rank_sync[rank].arrive_seq, step, __ATOMIC_RELEASE);
    wait_arrive(reg->rank_sync, ws, step);

    for (int r = 0; r < ws; r++) {
        memcpy((uint8_t*)comm->host_bounce + (size_t)r * in_bytes, reg->in_buf[r], in_bytes);
    }
    aclrtMemcpy(t->out_dev_ptr, out_bytes, comm->host_bounce, out_bytes, ACL_MEMCPY_HOST_TO_DEVICE);

    __atomic_store_n(&reg->rank_sync[rank].done_seq, step, __ATOMIC_RELEASE);
    wait_done(reg->rank_sync, ws, step);
}

static void host_reduce_scatter_cb(void *arg) {
    ShmGraphTask *t = (ShmGraphTask*)arg;
    ShmComm *comm = t->comm;
    ensure_acl_ctx(comm);
    int rank = comm->rank;
    int ws = comm->world_size;
    ShmRegion *reg = comm->region;
    size_t total_elems = t->numel;
    size_t total_bytes = t->bytes;
    size_t out_elems = total_elems / ws;
    size_t out_bytes = total_bytes / ws;

    uint64_t step = __atomic_add_fetch(&comm->step, 1, __ATOMIC_SEQ_CST);
    if (total_bytes <= RING_SLOT_BYTES) {
        uint8_t (*slot_buf)[RING_SLOT_BYTES] = reg->ring_buf[step & (RING_SLOTS - 1)];
        aclrtMemcpy(slot_buf[rank], total_bytes, t->in_dev_ptr, total_bytes, ACL_MEMCPY_DEVICE_TO_HOST);
        __atomic_store_n(&reg->rank_sync[rank].arrive_seq, step, __ATOMIC_RELEASE);
        wait_arrive(reg->rank_sync, ws, step);
        size_t s = rank * out_elems;
        size_t e = s + out_elems;
        reduce_ring_to_dst((uint8_t*)comm->host_bounce, slot_buf, ws, s, e, t->dtype, t->op);
        aclrtMemcpy(t->out_dev_ptr, out_bytes, comm->host_bounce, out_bytes, ACL_MEMCPY_HOST_TO_DEVICE);
        return;
    }

    aclrtMemcpy(reg->in_buf[rank], total_bytes, t->in_dev_ptr, total_bytes, ACL_MEMCPY_DEVICE_TO_HOST);

    __atomic_store_n(&reg->rank_sync[rank].arrive_seq, step, __ATOMIC_RELEASE);
    wait_arrive(reg->rank_sync, ws, step);

    size_t s = rank * out_elems;
    size_t e = s + out_elems;
    reduce_slice(reg->out_buf, reg->in_buf, ws, s, e, t->dtype, t->op);

    size_t el_sz = dtype_size(t->dtype);
    aclrtMemcpy(t->out_dev_ptr, out_bytes, reg->out_buf + s * el_sz, out_bytes, ACL_MEMCPY_HOST_TO_DEVICE);

    __atomic_store_n(&reg->rank_sync[rank].done_seq, step, __ATOMIC_RELEASE);
    wait_done(reg->rank_sync, ws, step);
}

static inline void ensure_bounce(ShmComm *comm) {
    if (comm->host_bounce == NULL) {
        if (posix_memalign(&comm->host_bounce, 128, comm->bounce_size) != 0) comm->host_bounce = NULL;
    }
}

int shm_all_reduce(ShmComm *comm, void *out_dev_ptr, const void *in_dev_ptr, size_t numel, int dtype, int op, void *stream, int is_capturing) {
    stream = flush_npu_stream();
    size_t el_sz = dtype_size(dtype);
    size_t total_bytes = numel * el_sz;
    if (is_capturing && total_bytes <= SLOT_SIZE) {
        ensure_bounce(comm);
        if (comm->host_bounce != NULL) {
             if (comm->acl_ctx == NULL) aclrtGetCurrentContext(&comm->acl_ctx);
            ShmGraphTask *t = (ShmGraphTask*)malloc(sizeof(ShmGraphTask));
            t->comm = comm;
            t->in_dev_ptr = in_dev_ptr;
            t->out_dev_ptr = out_dev_ptr;
            t->numel = numel;
            t->bytes = total_bytes;
            t->dtype = dtype;
            t->op = op;
            aclrtLaunchHostFunc(stream, host_all_reduce_cb, t);
            return 0;
        }
    }
    if (is_capturing) warn_capture_unsupported("shm_all_reduce", total_bytes);

    int rank = comm->rank;
    int ws = comm->world_size;
    ShmRegion *reg = comm->region;

    if (total_bytes <= RING_SLOT_BYTES) {
        ensure_bounce(comm);
        if (comm->host_bounce != NULL) {
            aclrtSynchronizeStream(stream);
            uint64_t step = __atomic_add_fetch(&comm->step, 1, __ATOMIC_SEQ_CST);
            uint8_t (*slot_buf)[RING_SLOT_BYTES] = reg->ring_buf[step & (RING_SLOTS - 1)];
            aclrtMemcpy(slot_buf[rank], total_bytes, in_dev_ptr, total_bytes, ACL_MEMCPY_DEVICE_TO_HOST);
            __atomic_store_n(&reg->rank_sync[rank].arrive_seq, step, __ATOMIC_RELEASE);
            wait_arrive(reg->rank_sync, ws, step);
            reduce_ring_to_dst((uint8_t*)comm->host_bounce, slot_buf, ws, 0, numel, dtype, op);
            aclrtMemcpy(out_dev_ptr, total_bytes, comm->host_bounce, total_bytes, ACL_MEMCPY_HOST_TO_DEVICE);
            return 0;
        }
    }

    size_t max_elems = SLOT_SIZE / el_sz;
    aclrtSynchronizeStream(stream);

    size_t offset_elems = 0;
    while (offset_elems < numel) {
        size_t chunk_elems = numel - offset_elems;
        if (chunk_elems > max_elems) chunk_elems = max_elems;
        size_t chunk_bytes = chunk_elems * el_sz;
        const uint8_t *cur_in = (const uint8_t*)in_dev_ptr + offset_elems * el_sz;
        uint8_t *cur_out = (uint8_t*)out_dev_ptr + offset_elems * el_sz;

        uint64_t step = __atomic_add_fetch(&comm->step, 1, __ATOMIC_SEQ_CST);
        aclrtMemcpy(reg->in_buf[rank], chunk_bytes, cur_in, chunk_bytes, ACL_MEMCPY_DEVICE_TO_HOST);

        __atomic_store_n(&reg->rank_sync[rank].arrive_seq, step, __ATOMIC_RELEASE);
        wait_arrive(reg->rank_sync, ws, step);

        size_t per_rank = (chunk_elems + ws - 1) / ws;
        size_t s = rank * per_rank;
        size_t e = s + per_rank;
        if (s > chunk_elems) s = chunk_elems;
        if (e > chunk_elems) e = chunk_elems;

        reduce_slice(reg->out_buf, reg->in_buf, ws, s, e, dtype, op);

        __atomic_store_n(&reg->rank_sync[rank].mid_seq, step, __ATOMIC_RELEASE);
        wait_mid(reg->rank_sync, ws, step);

        aclrtMemcpy(cur_out, chunk_bytes, reg->out_buf, chunk_bytes, ACL_MEMCPY_HOST_TO_DEVICE);

        __atomic_store_n(&reg->rank_sync[rank].done_seq, step, __ATOMIC_RELEASE);
        wait_done(reg->rank_sync, ws, step);

        offset_elems += chunk_elems;
    }
    return 0;
}

int shm_all_gather(ShmComm *comm, void *out_dev_ptr, const void *in_dev_ptr, size_t in_bytes, void *stream, int is_capturing) {
    stream = flush_npu_stream();
    if (is_capturing && in_bytes <= SLOT_SIZE) {
        ensure_bounce(comm);
        if (comm->host_bounce != NULL) {
            if (comm->acl_ctx == NULL) aclrtGetCurrentContext(&comm->acl_ctx);
            ShmGraphTask *t = (ShmGraphTask*)malloc(sizeof(ShmGraphTask));
            t->comm = comm;
            t->in_dev_ptr = in_dev_ptr;
            t->out_dev_ptr = out_dev_ptr;
            t->numel = 0;
            t->bytes = in_bytes;
            t->dtype = 4;
            t->op = 0;
            aclrtLaunchHostFunc(stream, host_all_gather_cb, t);
            return 0;
        }
    }
    if (is_capturing) warn_capture_unsupported("shm_all_gather", in_bytes);

    int rank = comm->rank;
    int ws = comm->world_size;
    ShmRegion *reg = comm->region;

    if (in_bytes <= RING_SLOT_BYTES) {
        ensure_bounce(comm);
        if (comm->host_bounce != NULL) {
            size_t out_bytes = in_bytes * (size_t)ws;
            aclrtSynchronizeStream(stream);
            uint64_t step = __atomic_add_fetch(&comm->step, 1, __ATOMIC_SEQ_CST);
            uint8_t (*slot_buf)[RING_SLOT_BYTES] = reg->ring_buf[step & (RING_SLOTS - 1)];
            aclrtMemcpy(slot_buf[rank], in_bytes, in_dev_ptr, in_bytes, ACL_MEMCPY_DEVICE_TO_HOST);
            __atomic_store_n(&reg->rank_sync[rank].arrive_seq, step, __ATOMIC_RELEASE);
            wait_arrive(reg->rank_sync, ws, step);
            for (int r = 0; r < ws; r++) {
                aclrtMemcpy((uint8_t*)out_dev_ptr + (size_t)r * in_bytes, in_bytes, slot_buf[r], in_bytes, ACL_MEMCPY_HOST_TO_DEVICE);
            }
            return 0;
        }
    }

    aclrtSynchronizeStream(stream);

    size_t offset = 0;
    while (offset < in_bytes) {
        size_t chunk = in_bytes - offset;
        if (chunk > SLOT_SIZE) chunk = SLOT_SIZE;

        uint64_t step = __atomic_add_fetch(&comm->step, 1, __ATOMIC_SEQ_CST);
        aclrtMemcpy(reg->in_buf[rank], chunk, (const uint8_t*)in_dev_ptr + offset, chunk, ACL_MEMCPY_DEVICE_TO_HOST);

        __atomic_store_n(&reg->rank_sync[rank].arrive_seq, step, __ATOMIC_RELEASE);
        wait_arrive(reg->rank_sync, ws, step);

        for (int r = 0; r < ws; r++) {
            uint8_t *dst = (uint8_t*)out_dev_ptr + (size_t)r * in_bytes + offset;
            aclrtMemcpy(dst, chunk, reg->in_buf[r], chunk, ACL_MEMCPY_HOST_TO_DEVICE);
        }

        __atomic_store_n(&reg->rank_sync[rank].done_seq, step, __ATOMIC_RELEASE);
        wait_done(reg->rank_sync, ws, step);

        offset += chunk;
    }
    return 0;
}

int shm_reduce_scatter(ShmComm *comm, void *out_dev_ptr, const void *in_dev_ptr, size_t total_numel, int dtype, int op, void *stream, int is_capturing) {
    stream = flush_npu_stream();
    size_t el_sz = dtype_size(dtype);
    size_t total_bytes = total_numel * el_sz;
    int ws = comm->world_size;
    int rank = comm->rank;
    size_t out_bytes = total_bytes / ws;
    size_t out_numel = total_numel / ws;

    if (is_capturing && total_bytes <= SLOT_SIZE) {
        ensure_bounce(comm);
        if (comm->host_bounce != NULL) {
            if (comm->acl_ctx == NULL) aclrtGetCurrentContext(&comm->acl_ctx);
            ShmGraphTask *t = (ShmGraphTask*)malloc(sizeof(ShmGraphTask));
            t->comm = comm;
            t->in_dev_ptr = in_dev_ptr;
            t->out_dev_ptr = out_dev_ptr;
            t->numel = total_numel;
            t->bytes = total_bytes;
            t->dtype = dtype;
            t->op = op;
            aclrtLaunchHostFunc(stream, host_reduce_scatter_cb, t);
            return 0;
        }
    }
    if (is_capturing) warn_capture_unsupported("shm_reduce_scatter", total_bytes);

    ShmRegion *reg = comm->region;

    if (total_bytes <= RING_SLOT_BYTES) {
        ensure_bounce(comm);
        if (comm->host_bounce != NULL) {
            aclrtSynchronizeStream(stream);
            uint64_t step = __atomic_add_fetch(&comm->step, 1, __ATOMIC_SEQ_CST);
            uint8_t (*slot_buf)[RING_SLOT_BYTES] = reg->ring_buf[step & (RING_SLOTS - 1)];
            aclrtMemcpy(slot_buf[rank], total_bytes, in_dev_ptr, total_bytes, ACL_MEMCPY_DEVICE_TO_HOST);
            __atomic_store_n(&reg->rank_sync[rank].arrive_seq, step, __ATOMIC_RELEASE);
            wait_arrive(reg->rank_sync, ws, step);
            size_t s = rank * out_numel;
            size_t e = s + out_numel;
            reduce_ring_to_dst((uint8_t*)comm->host_bounce, slot_buf, ws, s, e, dtype, op);
            aclrtMemcpy(out_dev_ptr, out_bytes, comm->host_bounce, out_bytes, ACL_MEMCPY_HOST_TO_DEVICE);
            return 0;
        }
    }

    aclrtSynchronizeStream(stream);

    if (total_bytes <= SLOT_SIZE) {
        uint64_t step = __atomic_add_fetch(&comm->step, 1, __ATOMIC_SEQ_CST);
        aclrtMemcpy(reg->in_buf[rank], total_bytes, in_dev_ptr, total_bytes, ACL_MEMCPY_DEVICE_TO_HOST);

        __atomic_store_n(&reg->rank_sync[rank].arrive_seq, step, __ATOMIC_RELEASE);
        wait_arrive(reg->rank_sync, ws, step);

        size_t s = rank * out_numel;
        size_t e = s + out_numel;
        reduce_slice(reg->out_buf, reg->in_buf, ws, s, e, dtype, op);

        aclrtMemcpy(out_dev_ptr, out_bytes, reg->out_buf + s * el_sz, out_bytes, ACL_MEMCPY_HOST_TO_DEVICE);

        __atomic_store_n(&reg->rank_sync[rank].done_seq, step, __ATOMIC_RELEASE);
        wait_done(reg->rank_sync, ws, step);
        return 0;
    }

    for (int target_r = 0; target_r < ws; target_r++) {
        size_t offset_elems = 0;
        size_t max_elems = SLOT_SIZE / el_sz;
        while (offset_elems < out_numel) {
            size_t chunk_elems = out_numel - offset_elems;
            if (chunk_elems > max_elems) chunk_elems = max_elems;
            size_t chunk_bytes = chunk_elems * el_sz;
            size_t global_elem_offset = (size_t)target_r * out_numel + offset_elems;

            uint64_t step = __atomic_add_fetch(&comm->step, 1, __ATOMIC_SEQ_CST);
            aclrtMemcpy(reg->in_buf[rank], chunk_bytes, (const uint8_t*)in_dev_ptr + global_elem_offset * el_sz, chunk_bytes, ACL_MEMCPY_DEVICE_TO_HOST);

            __atomic_store_n(&reg->rank_sync[rank].arrive_seq, step, __ATOMIC_RELEASE);
            wait_arrive(reg->rank_sync, ws, step);

            size_t per_rank = (chunk_elems + ws - 1) / ws;
            size_t s = rank * per_rank;
            size_t e = s + per_rank;
            if (s > chunk_elems) s = chunk_elems;
            if (e > chunk_elems) e = chunk_elems;
            reduce_slice(reg->out_buf, reg->in_buf, ws, s, e, dtype, op);

            __atomic_store_n(&reg->rank_sync[rank].mid_seq, step, __ATOMIC_RELEASE);
            wait_mid(reg->rank_sync, ws, step);

            if (rank == target_r) {
                aclrtMemcpy((uint8_t*)out_dev_ptr + offset_elems * el_sz, chunk_bytes, reg->out_buf, chunk_bytes, ACL_MEMCPY_HOST_TO_DEVICE);
            }

            __atomic_store_n(&reg->rank_sync[rank].done_seq, step, __ATOMIC_RELEASE);
            wait_done(reg->rank_sync, ws, step);

            offset_elems += chunk_elems;
        }
    }
    return 0;
}

int shm_broadcast(ShmComm *comm, void *dev_ptr, size_t total_bytes, int root, void *stream, int is_capturing) {
    stream = flush_npu_stream();
    if (is_capturing) warn_capture_unsupported("shm_broadcast", total_bytes);
    int rank = comm->rank;
    int ws = comm->world_size;
    ShmRegion *reg = comm->region;

    aclrtSynchronizeStream(stream);

    size_t offset = 0;
    while (offset < total_bytes) {
        size_t chunk = total_bytes - offset;
        if (chunk > SLOT_SIZE) chunk = SLOT_SIZE;

        uint64_t step = __atomic_add_fetch(&comm->step, 1, __ATOMIC_SEQ_CST);
        if (rank == root) {
            aclrtMemcpy(reg->out_buf, chunk, (const uint8_t*)dev_ptr + offset, chunk, ACL_MEMCPY_DEVICE_TO_HOST);
        }

        __atomic_store_n(&reg->rank_sync[rank].arrive_seq, step, __ATOMIC_RELEASE);
        wait_arrive(reg->rank_sync, ws, step);

        if (rank != root) {
            aclrtMemcpy((uint8_t*)dev_ptr + offset, chunk, reg->out_buf, chunk, ACL_MEMCPY_HOST_TO_DEVICE);
        }

        __atomic_store_n(&reg->rank_sync[rank].done_seq, step, __ATOMIC_RELEASE);
        wait_done(reg->rank_sync, ws, step);

        offset += chunk;
    }
    return 0;
}

int shm_gather(ShmComm *comm, void *out_dev_ptr, const void *in_dev_ptr, size_t in_bytes, int dst, void *stream, int is_capturing) {
    stream = flush_npu_stream();
    if (is_capturing) warn_capture_unsupported("shm_gather", in_bytes);
    int rank = comm->rank;
    int ws = comm->world_size;
    ShmRegion *reg = comm->region;

    aclrtSynchronizeStream(stream);

    size_t offset = 0;
    while (offset < in_bytes) {
        size_t chunk = in_bytes - offset;
        if (chunk > SLOT_SIZE) chunk = SLOT_SIZE;

        uint64_t step = __atomic_add_fetch(&comm->step, 1, __ATOMIC_SEQ_CST);
        aclrtMemcpy(reg->in_buf[rank], chunk, (const uint8_t*)in_dev_ptr + offset, chunk, ACL_MEMCPY_DEVICE_TO_HOST);

        __atomic_store_n(&reg->rank_sync[rank].arrive_seq, step, __ATOMIC_RELEASE);
        wait_arrive(reg->rank_sync, ws, step);

        if (rank == dst && out_dev_ptr != NULL) {
            for (int r = 0; r < ws; r++) {
                uint8_t *dst_ptr = (uint8_t*)out_dev_ptr + (size_t)r * in_bytes + offset;
                aclrtMemcpy(dst_ptr, chunk, reg->in_buf[r], chunk, ACL_MEMCPY_HOST_TO_DEVICE);
            }
        }

        __atomic_store_n(&reg->rank_sync[rank].done_seq, step, __ATOMIC_RELEASE);
        wait_done(reg->rank_sync, ws, step);

        offset += chunk;
    }
    return 0;
}

} // extern "C"
