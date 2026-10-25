#!/bin/bash
set -e
docker stop qwen38-27b >/dev/null 2>&1 || true
docker rm -f qwen38-27b-async >/dev/null 2>&1 || true
rm -f /dev/shm/vllm_shm_comm*

docker run -d --name qwen38-27b-async \
  --network host --ipc host --pid host --shm-size 67108864 --workdir /workspace \
  --device /dev/davinci0 --device /dev/davinci2 \
  --device /dev/davinci4 --device /dev/davinci6 \
  --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
  -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /home/models:/home/models \
  -v /home/models/shm_hccl/sampler_310p.py:/vllm-workspace/vllm-ascend/vllm_ascend/_310p/sample/sampler.py \
  -v /home/models/shm_hccl/npu_communicator.py:/vllm-workspace/vllm-ascend/vllm_ascend/distributed/device_communicators/npu_communicator.py \
  -v /home/models/shm_hccl/patch_distributed.py:/vllm-workspace/vllm-ascend/vllm_ascend/patch/worker/patch_distributed.py \
  -v /home/models/shm_hccl/llm_base_proposer.py:/vllm-workspace/vllm-ascend/vllm_ascend/spec_decode/llm_base_proposer.py \
  -v /home/models/shm_hccl/vocab_parallel_embedding_310p.py:/vllm-workspace/vllm-ascend/vllm_ascend/_310p/ops/vocab_parallel_embedding.py \
  -v /home/models/shm_hccl/modelslim_config_310p.py:/vllm-workspace/vllm-ascend/vllm_ascend/_310p/quantization/modelslim_config.py \
  -v /home/models/shm_hccl/w8a8_dynamic_310p.py:/vllm-workspace/vllm-ascend/vllm_ascend/_310p/quantization/methods/w8a8_dynamic.py \
  -v /home/models/shm_hccl/chunk_gated_delta_rule_310p.py:/vllm-workspace/vllm-ascend/vllm_ascend/_310p/ops/fla/chunk_gated_delta_rule.py \
  -v /home/models/shm_hccl/fused_gdn_gating_310p.py:/vllm-workspace/vllm-ascend/vllm_ascend/_310p/ops/fla/fused_gdn_gating.py \
  -v /home/models/shm_hccl/gdn_310.py:/vllm-workspace/vllm-ascend/vllm_ascend/_310p/ops/fla/gdn_310.py \
  -e PYTORCH_NPU_ALLOC_CONF=expandable_segments:True \
  -e SHM_HCCL_LIB_PATH=/home/models/shm_hccl/libshm_hccl.so \
  quay.io/ascend/vllm-ascend:v0.23.0.post1-310p \
  vllm serve /home/models/Qwen3.8-27B-w8a8-gdn-310p \
    --served-model-name Qwen/Qwen3.8-27B Qwen3.8-27B \
    --host 0.0.0.0 --port 8000 \
    --tensor-parallel-size 4 \
    --quantization ascend \
    --dtype float16 \
    --mamba-ssm-cache-dtype float16 \
    --max-model-len 131072 \
    --kv-cache-memory 5368709120 \
    --enable-prefix-caching \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 16 \
    --async-scheduling \
    --speculative-config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 3}' \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]}' \
    --additional-config '{"enable_reduce_sample": true, "ascend_compilation_config": {"enable_npugraph_ex": false, "fuse_allreduce_rms": false}}' \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3

docker ps --format "{{.Names}}|{{.Status}}"
