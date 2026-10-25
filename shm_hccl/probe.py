import ctypes, os, time
import torch, torch_npu  # noqa: F401
rank = int(os.environ.get("LOCAL_RANK", "0"))
torch.npu.set_device(rank)
lib = ctypes.CDLL("/home/models/shm_hccl/libprobe.so")
lib.probe_record.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
N = 130
print(f"{'mode':>10} {'KiB':>7} {'ms/step':>9} {'us/op':>8}")
for mode, label in ((0, "empty"), (1, "d2h"), (2, "d2h+h2d")):
    for kib in (20, 320):
        n_el = kib * 1024 // 2
        x = torch.zeros(n_el, dtype=torch.float16, device=f"npu:{rank}")
        torch.npu.synchronize()
        g = torch.npu.NPUGraph()
        with torch.npu.graph(g):
            lib.probe_record(mode, N, ctypes.c_void_p(x.data_ptr()), kib * 1024, 0)
        for _ in range(5):
            g.replay()
        torch.npu.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            g.replay()
        torch.npu.synchronize()
        t1 = time.perf_counter()
        ms = (t1 - t0) * 1000 / 20
        print(f"{label:>10} {kib:>7} {ms:>9.2f} {ms/N*1000:>8.1f}", flush=True)
        del g
        if mode == 0:
            break  # size-independent
