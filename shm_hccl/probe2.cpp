// Is the D2H target's memory type what caps us at ~2.5 GiB/s, and can the
// /dev/shm mapping be registered with CANN to remove the staging copy?
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <time.h>

extern "C" {
extern int aclrtMemcpy(void *dst, size_t destMax, const void *src, size_t count, int kind);
extern int aclrtMallocHost(void **hostPtr, size_t size);
extern int aclrtHostRegister(void *ptr, uint64_t size, int type, void **devPtr);
extern int aclrtHostUnregister(void *ptr);
#define D2H 2

static double now_ms(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

// returns GiB/s
double probe_bw(void *host, void *dev, size_t bytes, int reps) {
    aclrtMemcpy(host, bytes, dev, bytes, D2H);
    double best = 1e18;
    for (int t = 0; t < 5; t++) {
        double t0 = now_ms();
        for (int i = 0; i < reps; i++) aclrtMemcpy(host, bytes, dev, bytes, D2H);
        double el = (now_ms() - t0) / reps;
        if (el < best) best = el;
    }
    return (double)bytes / (1024.0*1024.0*1024.0) / (best / 1000.0);
}

void probe_report(void *dev, size_t bytes) {
    void *a = NULL, *p = NULL;
    posix_memalign(&a, 128, bytes);
    memset(a, 0, bytes);
    printf("  posix_memalign (unpinned) : %6.2f GiB/s\n", probe_bw(a, dev, bytes, 8));

    int rc_reg = aclrtHostRegister(a, bytes, 0, &p);
    if (rc_reg == 0) {
        printf("  + aclrtHostRegister OK    : %6.2f GiB/s\n", probe_bw(a, dev, bytes, 8));
        aclrtHostUnregister(a);
    } else {
        printf("  + aclrtHostRegister       : FAILED rc=%d (0x%x)\n", rc_reg, rc_reg);
    }
    free(a);

    void *h = NULL;
    int rc = aclrtMallocHost(&h, bytes);
    if (rc == 0) {
        memset(h, 0, bytes);
        printf("  aclrtMallocHost (pinned)  : %6.2f GiB/s\n", probe_bw(h, dev, bytes, 8));
    } else {
        printf("  aclrtMallocHost           : FAILED rc=%d\n", rc);
    }

    // tmpfs mmap, exactly what ShmRegion uses
    int fd = open("/dev/shm/probe2_region", O_RDWR | O_CREAT, 0666);
    if (fd >= 0 && ftruncate(fd, bytes) == 0) {
        void *m = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        close(fd);
        if (m != MAP_FAILED) {
            memset(m, 0, bytes);
            printf("  tmpfs mmap (ShmRegion)    : %6.2f GiB/s\n", probe_bw(m, dev, bytes, 8));
            void *dp = NULL;
            int rc2 = aclrtHostRegister(m, bytes, 0, &dp);
            if (rc2 == 0) {
                printf("  + HostRegister on tmpfs OK: %6.2f GiB/s\n", probe_bw(m, dev, bytes, 8));
                aclrtHostUnregister(m);
            } else {
                printf("  + HostRegister on tmpfs   : FAILED rc=%d (0x%x)\n", rc2, rc2);
            }
            munmap(m, bytes);
        }
        unlink("/dev/shm/probe2_region");
    }
    if (h) { /* leak pinned buffer, process exits */ }
}
}
