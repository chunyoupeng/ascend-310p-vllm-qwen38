import os
import torch
import torch_npu
import vllm_ascend.platform
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import (
    init_distributed_environment,
    initialize_model_parallel,
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_reduce_scatter,
    get_tp_group,
)

rank = int(os.environ['RANK'])
local_rank = int(os.environ['LOCAL_RANK'])
world_size = int(os.environ['WORLD_SIZE'])
torch.npu.set_device(local_rank)

with set_current_vllm_config(VllmConfig()):
    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        distributed_init_method='env://',
        local_rank=local_rank,
        backend='hccl',
    )
    initialize_model_parallel(tensor_model_parallel_size=world_size)

    x = torch.full((4, 1280), float(rank + 1), dtype=torch.float16, device=f'npu:{local_rank}')
    y = tensor_model_parallel_all_reduce(x)
    g = tensor_model_parallel_all_gather(x, dim=-1)
    rs = tensor_model_parallel_reduce_scatter(x, dim=-1)

    if rank == 0:
        d = {'a': torch.tensor([42.0, 99.0], dtype=torch.float16, device=f'npu:{local_rank}'), 'meta': 'ok'}
    else:
        d = None
    d_out = get_tp_group().broadcast_tensor_dict(d, src=0)

    # Test NPUGraph with tensor_model_parallel_all_reduce
    static_x = torch.full((2, 5120), float(rank + 1), dtype=torch.float16, device=f'npu:{local_rank}')
    s = torch.npu.Stream()
    with torch.npu.stream(s):
        _ = tensor_model_parallel_all_reduce(static_x)
    s.synchronize()

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=s):
        static_y = tensor_model_parallel_all_reduce(static_x)
    torch.npu.synchronize()

    static_x.fill_(float((rank + 1) * 10))
    torch.npu.synchronize()
    graph.replay()
    torch.npu.synchronize()

    if rank == 0:
        print('vLLM TP all_reduce:', float(y[0, 0].cpu()), 'expected 10.0')
        print('vLLM TP all_gather shape:', list(g.shape), 'vals:', [float(g[0, i*1280].cpu()) for i in range(world_size)])
        print('vLLM TP reduce_scatter shape:', list(rs.shape), 'val:', float(rs[0, 0].cpu()), 'expected 10.0')
        print('vLLM TP broadcast_tensor_dict:', d_out['a'].cpu().tolist(), d_out['meta'])
        print('vLLM TP NPUGraph replay:', float(static_y[0, 0].cpu()), 'expected 100.0')
