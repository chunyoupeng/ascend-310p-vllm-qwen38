---
library_name: transformers
license: mit
pipeline_tag: image-text-to-text
tasks:
  - image-text-to-text
frameworks: PyTorch
base_model:
  - Qwen/Qwen3.8-27B
base_model_relation: quantized
---

# Qwen3.8-27B-w8a8-gdn-310p

Full **W8A8 / W8A8_DYNAMIC** quantized checkpoint of **Qwen3.8-27B** optimized for **Huawei Ascend 310P (4x Atlas 300I Pro, 21.5 GB/card)** with `vllm-ascend`.

## What Was Modified from `Qwen3.8-27B-w8a8-310p`
1. **48-Layer GDN (`linear_attn.in_proj_qkvz` & `linear_attn.out_proj`) + `lm_head` + `mtp.fc` Quantized to `W8A8_DYNAMIC`**:
   - Converts all 98 remaining large FP16 linear projection matrices into symmetric per-channel INT8 (`weight` + `weight_scale` + `weight_offset`), reducing disk footprint from `36 GB` to **`29 GB`**.
   - Combined with on-the-fly INT8 `embed_tokens` and MTP `lm_head` sharing ([chunyoupeng/ascend-310p-vllm-qwen38](https://github.com/chunyoupeng/ascend-310p-vllm-qwen38)), per-card NPU weight memory drops from `10.99 GiB` to **`6.73 GiB`**, freeing **`265,045` tokens** of KV Cache capacity (`128K` context + multi-request concurrency).
2. **Updated `quant_model_description.json` & `generation_config.json`**:
   - Registers all 98 `W8A8_DYNAMIC` layers for `vllm-ascend` ModelSlim loader and configures greedy MTP speculative decoding (`num_speculative_tokens: 3`, **`24.78 tok/s`** at 4K context, **`19.18 tok/s`** at 100K context).

## Deployment Suite & Patches
See **[chunyoupeng/ascend-310p-vllm-qwen38](https://github.com/chunyoupeng/ascend-310p-vllm-qwen38)** for the Host-SHM lock-free collective communication library (`libshm_hccl.so`), 310P operator patches, and container launch script.
