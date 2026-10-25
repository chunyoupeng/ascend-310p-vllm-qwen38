#!/usr/bin/env bash
# Build a libshm_hccl variant. Usage: build_shm.sh <src.cpp> <out.so>
set -e
source /usr/local/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1
TORCH_NPU_LIB=$(python3 -c "import torch_npu, os; print(os.path.join(os.path.dirname(torch_npu.__file__), 'lib'))")
echo "TORCH_NPU_LIB=$TORCH_NPU_LIB"
g++ -O3 -march=armv8.2-a+fp16 -shared -fPIC -fopenmp "$1" \
    -L/usr/local/Ascend/ascend-toolkit/latest/lib64 \
    -L"$TORCH_NPU_LIB" \
    -lascendcl -ltorch_npu \
    -Wl,-rpath,/usr/local/Ascend/ascend-toolkit/latest/lib64:"$TORCH_NPU_LIB" \
    -o "$2"
echo "built $2"
