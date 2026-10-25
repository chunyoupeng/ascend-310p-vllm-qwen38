# Ascend 310P Multi-Card vLLM & Qwen3.8-27B Deployment Suite (`ascend-310p-vllm-qwen38`)

High-performance multi-card Tensor Parallel (`TP=4`) inference suite for **Qwen3.8-27B** on **Kunpeng 920 + 4× Atlas 300I Pro (Ascend 310P3, 21.5 GB/card)** with `vllm-ascend:v0.23.0.post1-310p`.

## Key Features & Optimizations

1. **Host-Shared-Memory Lock-Free Collective Communication (`shm_hccl/shm_hccl_opt.cpp` & `libshm_hccl.so`)**
   - Bypasses Ascend 310P PCIe P2P BAR limitations (`drv_devmm_host.ko ret=-14` / `207001`) via POSIX `/dev/shm` 64-slot lock-free ring buffers, 128-byte cache-line alignment, single-node `aclrtLaunchHostFunc` ACL graph capture, and ARMv8.2-A FP16 NEON vectorized reduction.
   - Reduces 130× `all_reduce` latency per decode step from `71.47 ms` (`0.550 ms/op`) to **`11.42 ms` (`0.088 ms/op`, 6.26× speedup)**.

2. **Full `W8A8_DYNAMIC` GDN + `lm_head` + `mtp.fc` Quantization & On-the-Fly INT8 `embed_tokens`**
   - [`scripts/quantize_gdn_w8a8.py`](scripts/quantize_gdn_w8a8.py): Parallel 9-worker offline converter quantizing all 48 `linear_attn` (GDN `in_proj_qkvz` + `out_proj`) layers, `lm_head`, and `mtp.fc` from FP16 to per-channel symmetric INT8 (`W8A8_DYNAMIC`).
   - [`shm_hccl/vocab_parallel_embedding_310p.py`](shm_hccl/vocab_parallel_embedding_310p.py): Quantizes `embed_tokens` on-the-fly into INT8 packed inside `float16` views compatible with `aclnnEmbedding` and eliminates redundant NZ weight copies.
   - [`shm_hccl/llm_base_proposer.py`](shm_hccl/llm_base_proposer.py): Shares `lm_head` between the main model and the MTP Drafter.
   - Reduces per-card NPU weight memory from `10.99 GiB` to **`6.73 GiB`** (saving **`4.26 GiB/card`**, **`17.04 GiB` total** across 4 cards) and expands KV Cache capacity to **`265,045` tokens** (`128K` context + concurrency).

3. **5-Step Divide-and-Conquer Lower-Triangular Solver for Prefill (`shm_hccl/chunk_gated_delta_rule_310p.py`)**
   - Replaces the 63-step serial Python loop in GDN WY representation with a numerically bounded 5-step recursive block solver ($T_{21} = T_{22} A_{21} T_{11}$), avoiding FP16 Cube overflow on slow-decay heads while accelerating long-context prefill.

4. **310P Native NPU Sampler, Sharded Penalties & Zero-Overhead Anti-Loop Detector (`shm_hccl/sampler_310p.py`)**
   - Executes random sampling natively on NPU with synchronized TP generators and `enable_reduce_sample: true` (local `62,080` vocab shards).
   - Fixes TP-sharded `apply_penalties` (`repetition_penalty`, `presence_penalty`, `frequency_penalty`) without out-of-bounds AICPU crashes.
   - Includes a zero-overhead tail periodic loop detector (`_detect_tail_loop_penalties_310p`) that prevents `<think>` greedy repetition loops while preserving 100% deterministic Top-1 greedy quality and full `MTP=3` speculative decoding speed.

## Single-Stream Performance (`TP=4`, `MTP=3`)

| Context Length | Prompt Tokens | TTFT (Cold) | Decode TPS (`MTP=3`) | Mean Acceptance (`tok/chunk`) |
| :--- | :---: | :---: | :---: | :---: |
| **4K** | 4,117 | `6.24 s` | **`24.78 tok/s`** | `2.78` |
| **32K** | 32,789 | `50.81 s` | **`21.22 tok/s`** | `2.63` |
| **64K** | 65,557 | `114.23 s` | **`19.57 tok/s`** | `2.70` |
| **100K** | 102,421 | `201.01 s` | **`19.18 tok/s`** | `2.94` |

## Directory Structure

- `shm_hccl/`: C++ shared-memory HCCL library (`shm_hccl_opt.cpp`, `libshm_hccl.so`), vLLM communicator patches, 310P GDN/Embedding/W8A8 operators, and `sampler_310p.py`.
- `scripts/s1_gdn_w8a8.sh`: Production container launch script for `qwen38-27b-async`.
- `scripts/quantize_gdn_w8a8.py`: Parallel offline safetensors quantizer (`Qwen3.8-27B-w8a8-310p` -> `Qwen3.8-27B-w8a8-gdn-310p`).
- `model_config/`: Updated `quant_model_description.json`, `generation_config.json`, and `config.json`.
- `昇腾310P多卡vLLM部署与HCCL绕过实战指南.md`: Full technical guide and root-cause analysis.
