"""130x all_reduce inside one NPUGraph, swept over the decode shapes vLLM
actually captures. [32, 5120] is the production shape: max_num_seqs 8 x MTP
(1+3) = 32 tokens, hidden 5120, fp16 -> 320 KiB, which straddles the old
RING_SLOT_BYTES of 256 KiB.

Run: torchrun --nproc_per_node=4 bench_shapes.py
Env: SHM_HCCL_LIB_PATH selects the .so.
"""
import ctypes
import os
import time

import torch
import torch_npu  # noqa: F401

rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])
torch.npu.set_device(rank)
dev = f"npu:{rank}"

lib_path = os.environ.get("SHM_HCCL_LIB_PATH", "/home/models/shm_hccl/libshm_hccl.so")
lib = ctypes.CDLL(lib_path)
lib.shm_comm_init.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
lib.shm_comm_init.restype = ctypes.c_void_p
lib.shm_all_reduce.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int,
]
lib.shm_all_gather.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_size_t, ctypes.c_void_p, ctypes.c_int,
]

comm = lib.shm_comm_init(b"bench_shapes_v1", world_size, rank, 1)
assert comm

NLAYER = 130
SHAPES = [(2, 5120), (8, 5120), (24, 5120), (32, 5120), (64, 5120)]
rows = []

for ntok, hidden in SHAPES:
    x = torch.full((ntok, hidden), float(rank + 1), dtype=torch.float16, device=dev)
    out = torch.zeros_like(x)
    torch.npu.synchronize()

    g = torch.npu.NPUGraph()
    with torch.npu.graph(g):
        st = torch.npu.current_stream().npu_stream
        cap = int(torch.npu.is_current_stream_capturing())
        for _ in range(NLAYER):
            lib.shm_all_reduce(comm, out.data_ptr(), x.data_ptr(), x.numel(), 0, 0, st, cap)
    for _ in range(5):
        g.replay()
    torch.npu.synchronize()

    t0 = time.perf_counter()
    for _ in range(20):
        g.replay()
    torch.npu.synchronize()
    t1 = time.perf_counter()

    ms_step = (t1 - t0) * 1000 / 20
    kib = ntok * hidden * 2 / 1024
    ok = abs(out[0, 0].item() - 10.0) < 1e-3
    rows.append((ntok, hidden, kib, ms_step, ms_step / NLAYER, ok))
    del g

# eager all_gather regression check (248 KiB per rank)
ag_in = torch.full((2, 62080), float(rank + 1), dtype=torch.float16, device=dev)
ag_out = torch.zeros((world_size, 2, 62080), dtype=torch.float16, device=dev)
s_ptr = torch.npu.current_stream().npu_stream
for _ in range(5):
    lib.shm_all_gather(comm, ag_out.data_ptr(), ag_in.data_ptr(), ag_in.numel() * 2, s_ptr, 0)
t2 = time.perf_counter()
for _ in range(20):
    lib.shm_all_gather(comm, ag_out.data_ptr(), ag_in.data_ptr(), ag_in.numel() * 2, s_ptr, 0)
t3 = time.perf_counter()
ag_ms = (t3 - t2) * 1000 / 20
ag_ok = ag_out[:, 0, 0].tolist() == [1.0, 2.0, 3.0, 4.0]

if rank == 0:
    print(f"=== {os.path.basename(lib_path)} ===", flush=True)
    print(f"{'tokens':>7} {'KiB':>8} {'ms/step':>9} {'ms/op':>8}  ok", flush=True)
    for ntok, hidden, kib, ms_step, ms_op, ok in rows:
        print(f"{ntok:>7} {kib:>8.0f} {ms_step:>9.2f} {ms_op:>8.3f}  {'y' if ok else 'FAIL'}", flush=True)
    print(f"eager all_gather 248KiB x4: {ag_ms:.3f} ms/op  {'y' if ag_ok else 'FAIL'}", flush=True)
