import ctypes, os
import torch, torch_npu  # noqa: F401
torch.npu.set_device(0)
lib = ctypes.CDLL("/home/models/shm_hccl/libprobe2.so")
lib.probe_report.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
for mib in (32, 256):
    n = mib*1024*1024//2
    x = torch.zeros(n, dtype=torch.float16, device="npu:0")
    torch.npu.synchronize()
    print(f"=== D2H {mib} MiB ===", flush=True)
    lib.probe_report(ctypes.c_void_p(x.data_ptr()), mib*1024*1024)
    del x
    torch.npu.empty_cache()
