import os, time, ctypes
import torch, torch_npu

rank = int(os.environ['LOCAL_RANK'])
world_size = int(os.environ['WORLD_SIZE'])
torch.npu.set_device(rank)

lib_path = os.environ.get('SHM_HCCL_LIB_PATH', '/home/models/shm_hccl/libshm_hccl.so')
lib = ctypes.CDLL(lib_path)
lib.shm_comm_init.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
lib.shm_comm_init.restype = ctypes.c_void_p
lib.shm_all_reduce.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int
]
lib.shm_all_gather.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_size_t, ctypes.c_void_p, ctypes.c_int
]

comm = lib.shm_comm_init(b'bench_group_v2', world_size, rank, 1)

# 1. Decode all_reduce size: [2, 5120] fp16 = 10240 elems (20 KB)
x = torch.full((2, 5120), float(rank + 1), dtype=torch.float16, device=f'npu:{rank}')
out = torch.zeros_like(x)
torch.npu.synchronize()

g = torch.npu.NPUGraph()
with torch.npu.graph(g):
    cap_stream = torch.npu.current_stream().npu_stream
    is_cap = int(torch.npu.is_current_stream_capturing())
    for _ in range(130):
        lib.shm_all_reduce(comm, out.data_ptr(), x.data_ptr(), x.numel(), 0, 0, cap_stream, is_cap)

for _ in range(5):
    g.replay()
torch.npu.synchronize()

t0 = time.perf_counter()
for _ in range(20):
    g.replay()
torch.npu.synchronize()
t1 = time.perf_counter()

# 2. Eager all_gather size: [2, 62080] fp16 = 248320 bytes/rank
ag_in = torch.full((2, 62080), float(rank + 1), dtype=torch.float16, device=f'npu:{rank}')
ag_out = torch.zeros((world_size, 2, 62080), dtype=torch.float16, device=f'npu:{rank}')
s_ptr = torch.npu.current_stream().npu_stream
for _ in range(5):
    lib.shm_all_gather(comm, ag_out.data_ptr(), ag_in.data_ptr(), ag_in.numel() * 2, s_ptr, 0)

t2 = time.perf_counter()
for _ in range(20):
    lib.shm_all_gather(comm, ag_out.data_ptr(), ag_in.data_ptr(), ag_in.numel() * 2, s_ptr, 0)
t3 = time.perf_counter()

if rank == 0:
    step_ar_ms = (t1 - t0) * 1000 / 20
    ag_ms = (t3 - t2) * 1000 / 20
    print(f'[{os.path.basename(lib_path)}] 130x NPUGraph all_reduce (20KB): {step_ar_ms:.2f} ms/step ({step_ar_ms/130:.3f} ms/op) | Eager all_gather (248KBx4): {ag_ms:.2f} ms/op | out={out[0,0].item()} ag_out={ag_out[:,0,0].tolist()}', flush=True)
