#!/usr/bin/env python3
"""
Parallel 9-worker offline W8A8_DYNAMIC quantizer for Qwen3.8-27B on Ascend 310P.
Quantizes the 48 GDN linear_attn layers (in_proj_qkvz, out_proj), lm_head, and mtp.fc
from FP16 to symmetric per-channel INT8 (weight + weight_scale + weight_offset),
reducing model disk footprint from 36 GB to 29 GB and per-card NPU weight memory to 6.73 GiB.
"""
import glob
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC = "/home/models/Qwen3.8-27B-w8a8-310p"
DST = "/home/models/Qwen3.8-27B-w8a8-gdn-310p"


def should_quantize(k: str) -> bool:
    if not k.endswith(".weight"):
        return False
    p = k[:-7]
    return (
        p.endswith("linear_attn.in_proj_qkvz")
        or p.endswith("linear_attn.out_proj")
        or p in ("model.language_model.lm_head", "lm_head", "mtp.fc")
    )


def process_shard(sf_path: str) -> str:
    bn = os.path.basename(sf_path)
    dst_path = os.path.join(DST, bn)
    tensors = {}
    with safe_open(sf_path, framework="pt", device="cpu") as f:
        metadata = f.metadata()
        for k in f.keys():
            w = f.get_tensor(k)
            if should_quantize(k) and w.dtype in (torch.float16, torch.bfloat16) and w.ndim == 2:
                p = k[:-7]
                wf = w.float()
                max_abs = wf.abs().amax(dim=1, keepdim=True).clamp(min=1e-5)
                scale = max_abs / 127.0
                wq = torch.clamp(torch.round(wf / scale), -128, 127).to(torch.int8)
                tensors[k] = wq
                tensors[p + ".weight_scale"] = scale.to(torch.float32)
                tensors[p + ".weight_offset"] = torch.zeros_like(scale, dtype=torch.float32)
            else:
                tensors[k] = w
    save_file(tensors, dst_path, metadata=metadata)
    return bn


def main():
    os.makedirs(DST, exist_ok=True)
    t0 = time.time()
    shards = sorted(glob.glob(os.path.join(SRC, "*.safetensors")))
    with ProcessPoolExecutor(max_workers=9) as ex:
        for bn in ex.map(process_shard, shards):
            print(f"Done {bn} ({time.time() - t0:.1f}s)", flush=True)
    for item in os.listdir(SRC):
        if not item.endswith(".safetensors") and not os.path.exists(os.path.join(DST, item)):
            s = os.path.join(SRC, item)
            if os.path.isfile(s):
                shutil.copy2(s, os.path.join(DST, item))
    print(f"ALL_DONE in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
