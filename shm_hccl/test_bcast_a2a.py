"""Covers the paths the matrix test does not: shm_broadcast / shm_gather with
their real (6/7) arg counts, and the all_to_all loop that used to deadlock
whenever src_r == dst_r."""
import ctypes, os, sys
import torch, torch_npu  # noqa: F401
rank = int(os.environ["LOCAL_RANK"]); ws = int(os.environ["WORLD_SIZE"])
torch.npu.set_device(rank); dev = f"npu:{rank}"
lib = ctypes.CDLL(os.environ["SHM_HCCL_LIB_PATH"])
lib.shm_comm_init.argtypes=[ctypes.c_char_p,ctypes.c_int,ctypes.c_int,ctypes.c_int]
lib.shm_comm_init.restype=ctypes.c_void_p
# Declared with the arg counts the C functions actually have.
lib.shm_broadcast.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t,
                            ctypes.c_int,ctypes.c_void_p,ctypes.c_int]
lib.shm_gather.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,
                         ctypes.c_size_t,ctypes.c_int,ctypes.c_void_p,ctypes.c_int]
comm = lib.shm_comm_init(b"bcast_a2a_v1", ws, rank, 1)
fails = []

# --- broadcast from every root ---
for root in range(ws):
    t = torch.full((4, 512), float(rank + 1), dtype=torch.float16, device=dev)
    nb = t.numel() * t.element_size()
    lib.shm_broadcast(comm, t.data_ptr(), nb, root, 0, 0)
    torch.npu.synchronize()
    if abs(t[0, 0].item() - (root + 1)) > 1e-3:
        fails.append(f"broadcast root={root}: got {t[0,0].item()} want {root+1}")

# --- gather to every dst ---
for dst in range(ws):
    x = torch.full((4, 512), float(rank + 1), dtype=torch.float16, device=dev)
    nb = x.numel() * x.element_size()
    if rank == dst:
        buf = torch.zeros((ws, 4, 512), dtype=torch.float16, device=dev)
        lib.shm_gather(comm, buf.data_ptr(), x.data_ptr(), nb, dst, 0, 0)
    else:
        lib.shm_gather(comm, 0, x.data_ptr(), nb, dst, 0, 0)
    torch.npu.synchronize()
    if rank == dst:
        got = buf[:, 0, 0].tolist()
        if got != [float(r + 1) for r in range(ws)]:
            fails.append(f"gather dst={dst}: got {got}")

# --- all_to_all, the fixed loop (every rank enters every broadcast) ---
# input_list[i] on rank r is tagged (r+1)*100 + i so a misrouted chunk shows up.
in_list = [torch.full((2, 256), float((rank + 1) * 100 + i), dtype=torch.float16, device=dev)
           for i in range(ws)]
out_list = [torch.zeros((2, 256), dtype=torch.float16, device=dev) for _ in range(ws)]
for src_r in range(ws):
    for dst_r in range(ws):
        if rank == src_r:
            buf = in_list[dst_r].contiguous()
        else:
            buf = torch.empty_like(out_list[src_r])
        nb = buf.numel() * buf.element_size()
        lib.shm_broadcast(comm, buf.data_ptr(), nb, src_r, 0, 0)
        if rank == dst_r:
            out_list[src_r].copy_(buf)
torch.npu.synchronize()
# out_list[i] must hold rank i's chunk destined for us: (i+1)*100 + rank
for i in range(ws):
    want = float((i + 1) * 100 + rank)
    got = out_list[i][0, 0].item()
    if abs(got - want) > 1e-3:
        fails.append(f"all_to_all out[{i}]: got {got} want {want}")

torch.npu.synchronize()
if rank == 0:
    tag = os.path.basename(os.environ["SHM_HCCL_LIB_PATH"])
    if fails:
        print(f"[{tag}] FAIL", flush=True)
        for f in fails:
            print("  " + f, flush=True)
        sys.exit(1)
    print(f"[{tag}] PASS broadcast x{ws} roots, gather x{ws} dsts, all_to_all {ws}x{ws}", flush=True)
