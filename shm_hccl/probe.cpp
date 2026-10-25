// Isolates the fixed costs inside an NPUGraph: how much is the aclrtLaunchHostFunc
// dispatch itself, and how much is the synchronous D2H+H2D inside the callback.
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

typedef void (*aclrtHostFunc)(void *args);

namespace c10_npu {
struct NPUStream { uint64_t data[2]; ~NPUStream() {} void* stream(bool need_empty) const; };
NPUStream getCurrentNPUStream(signed char device_index);
}
static inline void* flush_npu_stream() {
    c10_npu::NPUStream s = c10_npu::getCurrentNPUStream(-1);
    return s.stream(true);
}

extern "C" {
extern int aclrtMemcpy(void *dst, size_t destMax, const void *src, size_t count, int kind);
extern int aclrtLaunchHostFunc(void *stream, aclrtHostFunc fn, void *args);
extern int aclrtGetCurrentContext(void **context);
extern int aclrtSetCurrentContext(void *context);
#define D2H 2
#define H2D 1

typedef struct { void *ctx; void *host; void *dev; size_t bytes; } Task;

static __thread void *tls_ctx = NULL;
static inline void bind(Task *t) {
    if (t->ctx && tls_ctx != t->ctx) { aclrtSetCurrentContext(t->ctx); tls_ctx = t->ctx; }
}
static void cb_empty(void *a) { (void)a; }
static void cb_d2h(void *a)  { Task *t = (Task*)a; bind(t); aclrtMemcpy(t->host, t->bytes, t->dev, t->bytes, D2H); }
static void cb_both(void *a) { Task *t = (Task*)a; bind(t); aclrtMemcpy(t->host, t->bytes, t->dev, t->bytes, D2H);
                                                           aclrtMemcpy(t->dev, t->bytes, t->host, t->bytes, H2D); }

// mode 0=empty 1=d2h 2=d2h+h2d ; records n nodes onto the capturing stream
int probe_record(int mode, int n, void *dev_ptr, size_t bytes, int pinned) {
    void *stream = flush_npu_stream();
    Task *t = (Task*)malloc(sizeof(Task));
    t->ctx = NULL; aclrtGetCurrentContext(&t->ctx);
    t->dev = dev_ptr; t->bytes = bytes;
    if (posix_memalign(&t->host, 128, bytes ? bytes : 64) != 0) return -1;
    memset(t->host, 0, bytes ? bytes : 64);
    (void)pinned;
    aclrtHostFunc fn = (mode == 0) ? cb_empty : (mode == 1 ? cb_d2h : cb_both);
    for (int i = 0; i < n; i++) aclrtLaunchHostFunc(stream, fn, t);
    return 0;
}
}
