# Ascend 310P Multi-Card vLLM & Qwen3.8-27B Deployment Suite (`ascend-310p-vllm-qwen38`)

High-performance multi-card Tensor Parallel (`TP=4`) inference suite for **Qwen3.8-27B** on **Kunpeng 920 + 4× Atlas 300I Pro (Ascend 310P3, 21.5 GB/card)** with `vllm-ascend:v0.23.0.post1-310p`.

## Key Features & Optimizations

1. **Host-Shared-Memory Lock-Free Collective Communication (`shm_hccl/shm_hccl_v2.cpp` & `libshm_hccl_v2.so`)**
   - Bypasses Ascend 310P PCIe P2P BAR limitations (`drv_devmm_host.ko ret=-14` / `207001`) via POSIX `/dev/shm` lock-free ring buffers (`1 MiB` ring slots, `80 MiB` bulk slots for single-pass 8K prefill `all_reduce`), 128-byte cache-line alignment, `aclrtLaunchHostFunc` ACL graph capture, and ARMv8.2-A FP16 NEON vectorized reduction.
   - Uses CANN `aclrtMallocHost` pinned DMA bounce buffers (`11.83 GiB/s` vs `8.09 GiB/s` unpinned), per-rank local NUMA tmpfs page first-touch, and coalesced single-DMA H2D `all_gather`.
   - Reduces 130× `all_reduce` latency per decode step from `71.47 ms` (`0.550 ms/op`) to **`11.42 ms` (`0.088 ms/op`, 6.26× speedup)**.

2. **Full `W8A8_DYNAMIC` GDN + `lm_head` + `mtp.fc` Quantization & On-the-Fly INT8 `embed_tokens`**
   - [`scripts/quantize_gdn_w8a8.py`](scripts/quantize_gdn_w8a8.py): Parallel 9-worker offline converter quantizing all 48 `linear_attn` (GDN `in_proj_qkvz` + `out_proj`) layers, `lm_head`, and `mtp.fc` from FP16 to per-channel symmetric INT8 (`W8A8_DYNAMIC`).
   - [`shm_hccl/vocab_parallel_embedding_310p.py`](shm_hccl/vocab_parallel_embedding_310p.py): Quantizes `embed_tokens` on-the-fly into INT8 packed inside `float16` views compatible with `aclnnEmbedding` and eliminates redundant NZ weight copies.
   - [`shm_hccl/llm_base_proposer.py`](shm_hccl/llm_base_proposer.py): Shares `lm_head` between the main model and the MTP Drafter.
   - Reduces per-card NPU weight memory from `10.99 GiB` to **`6.73 GiB`** (saving **`4.26 GiB/card`**, **`17.04 GiB` total** across 4 cards) and expands KV Cache capacity to **`265,045` tokens** (`128K` context + concurrency).

3. **5-Step Divide-and-Conquer Lower-Triangular Solver for Prefill (`shm_hccl/chunk_gated_delta_rule_310p.py`)**
   - Replaces the 63-step serial Python loop in GDN WY representation with a numerically bounded 5-step recursive block solver ($T_{21} = T_{22} A_{21} T_{11}$), avoiding FP16 Cube overflow on slow-decay heads while accelerating long-context prefill.

4. **Zero-Sync Single-Collective Sampler & Dual-Socket NUMA Binding (`shm_hccl/sampler_310p_v2.py`, `llm_base_proposer_v2.py`, `cpu_binding_310p.py`)**
   - Collapses sharded `greedy_sample` from 2× `dim=-1` `all_gather` into 1× transpose-free `dim=0` `[B, 2]` collective across both MTP draft and target verification, fuses `bonus + target` greedy verification into a single collective, and replaces `rejection_greedy_sample_pytorch` with a static-shape sync-free tensor kernel (eliminating 5 NPU-to-CPU stream syncs per step).
   - Fixes `apply_top_k_top_p` fallback when `self.top_k is None` on `temperature > 0` (avoiding `topk(62080)` and `9 MB` cross-card gathers) and coalesces candidate `(val, idx)` gathers.
   - Fixes physical Card ID (`NPU1,3,4,6`) to logical ID (`0,1,2,3`) mapping in `cpu_binding_310p.py` and prevents cross-socket `extend_numa`, pinning `TP0/1` to `NUMA 0` (`CPU 0-31`) and `TP2/3` to `NUMA 1` (`CPU 32-63`) with `migratepages`.
   - Scales MTP speculative decoding to `num_speculative_tokens = 5` with aligned `cudagraph_capture_sizes = [6, 12, 18, 24, 36, 48, 72, 96]` (`scripts/s1_gdn_w8a8_opt.sh`).

## Single-Stream Performance (`TP=4`, `20K` Context Benchmark)

Measured via [`shm_hccl/bench_20k.py`](shm_hccl/bench_20k.py) (`~20,440` prompt tokens, `250` output tokens):

| Scenario (`~20.4K` Prompt) | Baseline (`MTP=3`) | Optimized (`MTP=4`) | Optimized (`MTP=5`) | Speedup |
| :--- | :---: | :---: | :---: | :---: |
| **20K Uncached Greedy (`t=0`)** | `25.83 tok/s` | `29.11 tok/s` | **`32.48 tok/s`** (peak `35.24`) | **`+25.7%`** |
| **20K Uncached Default (`t=0.3`)** | `27.57 tok/s` | `31.25 tok/s` | **`32.88 tok/s`** (peak `33.06`) | **`+19.3%`** |
| **20K Prefix-Cached Greedy (`t=0`)** | `26.26 tok/s` | `30.93 tok/s` | **`33.96 tok/s`** (peak `34.63`) | **`+29.3%`** |
| **20K Prefix-Cached Default (`t=0.3`)** | `20.77 tok/s` | `26.89 tok/s` | **`32.78 tok/s`** (peak `36.45`) | **`+57.8%`** |

## Long-Context Baseline Sweep (`TP=4`, `MTP=3`)

| Context Length | Prompt Tokens | TTFT (Cold) | Decode TPS (`MTP=3`) | Mean Acceptance (`tok/chunk`) |
| :--- | :---: | :---: | :---: | :---: |
| **4K** | 4,117 | `6.24 s` | **`24.78 tok/s`** | `2.78` |
| **32K** | 32,789 | `50.81 s` | **`21.22 tok/s`** | `2.63` |
| **64K** | 65,557 | `114.23 s` | **`19.57 tok/s`** | `2.70` |
| **100K** | 102,421 | `201.01 s` | **`19.18 tok/s`** | `2.94` |

## Directory Structure

- `shm_hccl/`: C++ shared-memory HCCL library (`shm_hccl_v2.cpp`, `libshm_hccl_v2.so`), vLLM communicator/NUMA binding patches (`cpu_binding_310p.py`), 310P GDN/Embedding/W8A8 operators, and zero-sync sampler (`sampler_310p_v2.py`, `llm_base_proposer_v2.py`).
- `scripts/s1_gdn_w8a8_opt.sh`: Optimized `MTP=5` container launch script (`scripts/s1_gdn_w8a8.sh` retained as `MTP=3` baseline).
- `scripts/quantize_gdn_w8a8.py`: Parallel offline safetensors quantizer (`Qwen3.8-27B-w8a8-310p` -> `Qwen3.8-27B-w8a8-gdn-310p`).
- `model_config/`: Updated `quant_model_description.json`, `generation_config.json`, and `config.json`.
- `昇腾310P多卡vLLM部署与HCCL绕过实战指南.md`: Full technical guide and root-cause analysis.
