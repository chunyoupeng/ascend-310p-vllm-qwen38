"""Correctness matrix for libshm_hccl.so.

Exercises dtype x op x size x collective, in both eager and NPUGraph-capture
mode, against CPU-computed expectations.  Sizes are chosen to straddle the
RING_SLOT_BYTES and SLOT_SIZE thresholds so every internal path is hit.

Run: torchrun --nproc_per_node=4 test_shm_matrix.py
Env: SHM_HCCL_LIB_PATH selects which .so to validate.
"""
import ctypes
import os
import sys

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
lib.shm_reduce_scatter.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int,
]

comm = lib.shm_comm_init(b"matrix_test_v1", world_size, rank, 1)
assert comm, "shm_comm_init failed"

DT = {
    "fp16": (torch.float16, 0),
    "fp32": (torch.float32, 1),
    "int64": (torch.int64, 2),
    "int32": (torch.int32, 3),
}
OPS = {"sum": 0, "max": 1, "min": 2, "prod": 3}

# Sizes in elements, chosen to straddle RING_SLOT_BYTES (256K now, 512K after)
# and SLOT_SIZE (32 MiB).
SIZES = {
    "20KB": 10_240,          # ring on both old and new
    "320KB": 163_840,        # fp16: 320 KiB -> in_buf on old, ring on new
    "768KB": 393_216,        # above both ring sizes -> in_buf path
    "40MB": 20_971_520,      # fp16: 40 MiB -> above SLOT_SIZE, chunked loop
}

failures = []
checked = 0


def pattern(n, dtype, r):
    """Rank-dependent, index-dependent values; small enough to keep fp16 exact."""
    idx = torch.arange(n, dtype=torch.float32)
    vals = (r + 1) + (idx % 7)
    if dtype in (torch.int64, torch.int32):
        return vals.to(dtype)
    return vals.to(dtype)


def expect_reduce(n, dtype, op, lo=0, hi=None):
    hi = n if hi is None else hi
    stack = torch.stack([pattern(n, torch.float64, r) for r in range(world_size)])
    if op == "sum":
        out = stack.sum(0)
    elif op == "max":
        out = stack.max(0).values
    elif op == "min":
        out = stack.min(0).values
    else:
        out = stack.prod(0)
    return out[lo:hi].to(dtype)


def check(name, got, exp):
    global checked
    checked += 1
    got_c = got.cpu()
    exp_c = exp.cpu()
    if got_c.dtype in (torch.int64, torch.int32):
        ok = torch.equal(got_c, exp_c)
        err = 0 if ok else int((got_c - exp_c).abs().max())
    else:
        diff = (got_c.to(torch.float64) - exp_c.to(torch.float64)).abs()
        err = float(diff.max())
        ok = err <= 1e-3 * max(1.0, float(exp_c.to(torch.float64).abs().max()))
    if not ok:
        failures.append(f"{name}: max_err={err} got[:4]={got_c.flatten()[:4].tolist()} exp[:4]={exp_c.flatten()[:4].tolist()}")
    return ok


def run_all_reduce(size_name, dt_name, op_name, capturing=0):
    n = SIZES[size_name]
    dtype, dcode = DT[dt_name]
    if dt_name == "int64" and size_name == "40MB":
        return  # 160 MiB per rank, skip to keep runtime sane
    if op_name == "prod" and dt_name in ("int32", "int64") and size_name == "40MB":
        return
    x = pattern(n, dtype, rank).to(dev)
    out = torch.zeros_like(x)
    torch.npu.synchronize()
    lib.shm_all_reduce(comm, out.data_ptr(), x.data_ptr(), n, dcode, OPS[op_name], 0, capturing)
    torch.npu.synchronize()
    check(f"all_reduce/{size_name}/{dt_name}/{op_name}", out, expect_reduce(n, dtype, op_name).to(dev))


def run_all_gather(size_name, dt_name):
    n = SIZES[size_name]
    dtype, _ = DT[dt_name]
    if size_name == "40MB":
        return  # output would be 4x40 MiB on device
    x = pattern(n, dtype, rank).to(dev)
    out = torch.zeros((world_size, n), dtype=dtype, device=dev)
    nbytes = n * x.element_size()
    torch.npu.synchronize()
    lib.shm_all_gather(comm, out.data_ptr(), x.data_ptr(), nbytes, 0, 0)
    torch.npu.synchronize()
    exp = torch.stack([pattern(n, dtype, r) for r in range(world_size)]).to(dev)
    check(f"all_gather/{size_name}/{dt_name}", out, exp)


def run_reduce_scatter(size_name, dt_name, op_name):
    n = SIZES[size_name]
    dtype, dcode = DT[dt_name]
    if size_name == "40MB":
        return
    assert n % world_size == 0
    seg = n // world_size
    x = pattern(n, dtype, rank).to(dev)
    out = torch.zeros(seg, dtype=dtype, device=dev)
    torch.npu.synchronize()
    lib.shm_reduce_scatter(comm, out.data_ptr(), x.data_ptr(), n, dcode, OPS[op_name], 0, 0)
    torch.npu.synchronize()
    exp = expect_reduce(n, dtype, op_name, rank * seg, (rank + 1) * seg).to(dev)
    check(f"reduce_scatter/{size_name}/{dt_name}/{op_name}", out, exp)


# ---- eager matrix -------------------------------------------------------
for size_name in SIZES:
    for dt_name in DT:
        for op_name in OPS:
            run_all_reduce(size_name, dt_name, op_name)
        run_all_gather(size_name, dt_name)
        run_reduce_scatter(size_name, dt_name, "sum")

# ---- NPUGraph capture/replay -------------------------------------------
# Only sizes that the capture path accepts (<= SLOT_SIZE) and SUM/fp16+fp32,
# which is what vLLM's decode graph actually issues.
for size_name in ("20KB", "320KB", "768KB"):
    for dt_name in ("fp16", "fp32"):
        n = SIZES[size_name]
        dtype, dcode = DT[dt_name]
        x = pattern(n, dtype, rank).to(dev)
        out = torch.zeros_like(x)
        torch.npu.synchronize()
        g = torch.npu.NPUGraph()
        with torch.npu.graph(g):
            st = torch.npu.current_stream().npu_stream
            cap = int(torch.npu.is_current_stream_capturing())
            lib.shm_all_reduce(comm, out.data_ptr(), x.data_ptr(), n, dcode, 0, st, cap)
        torch.npu.synchronize()
        out.zero_()
        torch.npu.synchronize()
        g.replay()
        torch.npu.synchronize()
        check(f"graph_all_reduce/{size_name}/{dt_name}/sum", out, expect_reduce(n, dtype, "sum").to(dev))
        del g

torch.npu.synchronize()
if rank == 0:
    tag = os.path.basename(lib_path)
    if failures:
        print(f"[{tag}] FAIL {len(failures)}/{checked}", flush=True)
        for f in failures:
            print(f"   {f}", flush=True)
        sys.exit(1)
    print(f"[{tag}] PASS all {checked} checks", flush=True)
