"""Prefill-shaped all_reduce: [max_num_batched_tokens, 5120] fp16, eager.
These sizes exceed RING_SLOT_BYTES and SLOT_SIZE, so they take the chunked
in_buf path through reduce_slice -- the one with no NEON specialisation."""
import ctypes, os, time
import torch, torch_npu  # noqa: F401
rank = int(os.environ["LOCAL_RANK"]); ws = int(os.environ["WORLD_SIZE"])
torch.npu.set_device(rank); dev = f"npu:{rank}"
lib = ctypes.CDLL(os.environ.get("SHM_HCCL_LIB_PATH", "/home/models/shm_hccl/libshm_hccl.so"))
lib.shm_comm_init.argtypes=[ctypes.c_char_p,ctypes.c_int,ctypes.c_int,ctypes.c_int]
lib.shm_comm_init.restype=ctypes.c_void_p
lib.shm_all_reduce.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,
                             ctypes.c_size_t,ctypes.c_int,ctypes.c_int,ctypes.c_void_p,ctypes.c_int]
comm = lib.shm_comm_init(b"bench_prefill_v1", ws, rank, 1)
TRIALS, REPS = 5, 4
if rank == 0:
    print(f"{'tokens':>7} {'MiB':>7} {'ms/op(min)':>11} {'GiB/s':>7}  ok", flush=True)
for ntok in (512, 2048, 8192, 16384):
    x = torch.full((ntok, 5120), float(rank+1), dtype=torch.float16, device=dev)
    out = torch.zeros_like(x)
    mib = ntok*5120*2/1048576
    torch.npu.synchronize()
    for _ in range(2):
        lib.shm_all_reduce(comm, out.data_ptr(), x.data_ptr(), x.numel(), 0, 0, 0, 0)
    torch.npu.synchronize()
    best = 1e9
    for _ in range(TRIALS):
        t0 = time.perf_counter()
        for _ in range(REPS):
            lib.shm_all_reduce(comm, out.data_ptr(), x.data_ptr(), x.numel(), 0, 0, 0, 0)
        torch.npu.synchronize()
        best = min(best, (time.perf_counter()-t0)*1000/REPS)
    ok = abs(out[0,0].item()-10.0) < 1e-3
    if rank == 0:
        print(f"{ntok:>7} {mib:>7.1f} {best:>11.2f} {mib/1024/(best/1000):>7.2f}  {'y' if ok else 'FAIL'}", flush=True)
    del x, out
