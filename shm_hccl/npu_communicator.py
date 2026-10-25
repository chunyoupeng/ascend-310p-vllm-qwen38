#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

import ctypes
import logging
import os
import uuid
import torch
import torch.distributed as dist
from vllm.distributed.device_communicators.base_device_communicator import DeviceCommunicatorBase

logger = logging.getLogger(__name__)

_SHM_LIB_PATH = os.getenv("SHM_HCCL_LIB_PATH", "/home/models/shm_hccl/libshm_hccl.so")
_LIB = ctypes.CDLL(_SHM_LIB_PATH)

_LIB.shm_comm_init.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
_LIB.shm_comm_init.restype = ctypes.c_void_p

_LIB.shm_comm_destroy.argtypes = [ctypes.c_void_p]
_LIB.shm_comm_destroy.restype = None

_LIB.shm_barrier.argtypes = [ctypes.c_void_p]
_LIB.shm_barrier.restype = ctypes.c_int

_LIB.shm_all_reduce.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
]
_LIB.shm_all_reduce.restype = ctypes.c_int

_LIB.shm_all_gather.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_void_p,
    ctypes.c_int,
]
_LIB.shm_all_gather.restype = ctypes.c_int

_LIB.shm_reduce_scatter.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
]
_LIB.shm_reduce_scatter.restype = ctypes.c_int

_LIB.shm_broadcast.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
]
_LIB.shm_broadcast.restype = ctypes.c_int

_LIB.shm_gather.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
]
_LIB.shm_gather.restype = ctypes.c_int

_DTYPE_MAP = {
    torch.float16: 0,
    torch.float32: 1,
    torch.int64: 2,
    torch.int32: 3,
    torch.int8: 4,
    torch.uint8: 5,
    torch.bool: 5,
    torch.bfloat16: 6,
}

_SHM_COMM_BY_RANKS: dict[tuple[int, ...], int] = {}
_COMM_BY_RANKS: dict[tuple[int, ...], "NPUCommunicator"] = {}
_COMM_BY_PG: dict[int, "NPUCommunicator"] = {}
_DEVICE_PG_IDS: set[int] = set()


def _dtype_code(dtype: torch.dtype) -> int:
    return _DTYPE_MAP.get(dtype, 4)


def _op_code(op) -> int:
    if op == dist.ReduceOp.MAX:
        return 1
    if op == dist.ReduceOp.MIN:
        return 2
    if op == dist.ReduceOp.PRODUCT:
        return 3
    return 0


def _is_capturing() -> int:
    try:
        return 1 if torch.npu.is_current_stream_capturing() else 0
    except Exception:
        return 0


class NPUCommunicator(DeviceCommunicatorBase):
    def __init__(
        self,
        cpu_group: dist.ProcessGroup,
        device: torch.device | None = None,
        device_group: dist.ProcessGroup | None = None,
        unique_name: str = "",
    ):
        super().__init__(cpu_group, device, device_group, unique_name)
        self.device = torch.npu.current_device()
        self.ca_comm = None
        self.shm_comm = None

        if self.world_size > 1:
            ranks_key = tuple(self.ranks)
            if ranks_key in _SHM_COMM_BY_RANKS:
                self.shm_comm = _SHM_COMM_BY_RANKS[ranks_key]
            else:
                if self.rank_in_group == 0:
                    ranks_tag = "_".join(str(r) for r in self.ranks)
                    shm_name = f"{os.getpid()}_{ranks_tag}_{uuid.uuid4().hex[:8]}"
                else:
                    shm_name = ""
                obj_list = [shm_name]
                dist.broadcast_object_list(obj_list, src=self.ranks[0], group=self.cpu_group)
                shm_name = obj_list[0]
                self.shm_comm = _LIB.shm_comm_init(
                    shm_name.encode("utf-8"),
                    self.world_size,
                    self.rank_in_group,
                    1,
                )
                if not self.shm_comm:
                    raise RuntimeError(f"Failed to init ShmComm for {shm_name}")
                _SHM_COMM_BY_RANKS[ranks_key] = self.shm_comm
                logger.info(
                    "Initialized Host SHM communicator %s for ranks %s (rank_in_group=%d)",
                    shm_name,
                    self.ranks,
                    self.rank_in_group,
                )

            _COMM_BY_RANKS[ranks_key] = self
            if self.device_group is not None:
                _COMM_BY_PG[id(self.device_group)] = self
                _DEVICE_PG_IDS.add(id(self.device_group))
            if self.cpu_group is not None:
                _COMM_BY_PG[id(self.cpu_group)] = self

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        if self.world_size <= 1:
            return input_
        x = input_.contiguous()
        out = torch.empty_like(x)
        _LIB.shm_all_reduce(
            self.shm_comm,
            out.data_ptr(),
            x.data_ptr(),
            x.numel(),
            _dtype_code(x.dtype),
            0,
            0,
            _is_capturing(),
        )
        return out

    def all_reduce_inplace(self, tensor: torch.Tensor, op=dist.ReduceOp.SUM) -> None:
        if self.world_size <= 1:
            return
        if tensor.is_contiguous():
            _LIB.shm_all_reduce(
                self.shm_comm,
                tensor.data_ptr(),
                tensor.data_ptr(),
                tensor.numel(),
                _dtype_code(tensor.dtype),
                _op_code(op),
                0,
                _is_capturing(),
            )
        else:
            tmp = tensor.contiguous()
            _LIB.shm_all_reduce(
                self.shm_comm,
                tmp.data_ptr(),
                tmp.data_ptr(),
                tmp.numel(),
                _dtype_code(tmp.dtype),
                _op_code(op),
                0,
                _is_capturing(),
            )
            tensor.copy_(tmp)

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        if self.world_size <= 1:
            return input_
        if dim < 0:
            dim += input_.dim()
        x = input_.contiguous()
        input_size = x.size()
        in_bytes = x.numel() * x.element_size()
        output_tensor = torch.empty(
            (self.world_size,) + input_size,
            dtype=x.dtype,
            device=x.device,
        )
        _LIB.shm_all_gather(
            self.shm_comm,
            output_tensor.data_ptr(),
            x.data_ptr(),
            in_bytes,
            0,
            _is_capturing(),
        )
        if dim != 0:
            output_tensor = output_tensor.movedim(0, dim)
        output_tensor = output_tensor.reshape(
            input_size[:dim] + (self.world_size * input_size[dim],) + input_size[dim + 1:]
        )
        return output_tensor

    def reduce_scatter(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        if self.world_size <= 1:
            return input_
        world_size = self.world_size
        if dim < 0:
            dim += input_.dim()
        if dim != 0:
            input_ = input_.movedim(dim, 0)
        assert input_.size(0) % world_size == 0
        chunks = input_.size(0) // world_size
        output_shape = (chunks,) + input_.size()[1:]
        x = input_.contiguous()
        output_tensor = torch.empty(
            output_shape,
            dtype=x.dtype,
            device=x.device,
        )
        _LIB.shm_reduce_scatter(
            self.shm_comm,
            output_tensor.data_ptr(),
            x.data_ptr(),
            x.numel(),
            _dtype_code(x.dtype),
            0,
            0,
            _is_capturing(),
        )
        if dim != 0:
            output_tensor = output_tensor.movedim(0, dim)
        return output_tensor

    def reduce_scatterv(
        self,
        input_: torch.Tensor,
        dim: int = -1,
        sizes: list[int] | None = None,
    ) -> torch.Tensor:
        if sizes is None:
            return self.reduce_scatter(input_, dim=dim)
        if dim < 0:
            dim += input_.dim()
        full = self.all_reduce(input_)
        splits = torch.split(full, sizes, dim=dim)
        return splits[self.rank_in_group].contiguous()

    def gather(self, input_: torch.Tensor, dst: int = 0, dim: int = -1) -> torch.Tensor | None:
        if self.world_size <= 1:
            return input_ if self.rank_in_group == dst else None
        if dim < 0:
            dim += input_.dim()
        x = input_.contiguous()
        input_size = x.size()
        in_bytes = x.numel() * x.element_size()
        if self.rank_in_group == dst:
            buf = torch.empty((self.world_size,) + input_size, dtype=x.dtype, device=x.device)
            out_ptr = buf.data_ptr()
        else:
            buf = None
            out_ptr = 0
        _LIB.shm_gather(self.shm_comm, out_ptr, x.data_ptr(), in_bytes, dst, 0, _is_capturing())
        if self.rank_in_group == dst:
            assert buf is not None
            if dim != 0:
                buf = buf.movedim(0, dim)
            return buf.reshape(
                input_size[:dim] + (self.world_size * input_size[dim],) + input_size[dim + 1:]
            )
        return None

    def broadcast_inplace(self, tensor: torch.Tensor, root_in_group: int = 0) -> None:
        if self.world_size <= 1:
            return
        nbytes = tensor.numel() * tensor.element_size()
        if tensor.is_contiguous():
            _LIB.shm_broadcast(self.shm_comm, tensor.data_ptr(), nbytes, root_in_group, 0, _is_capturing())
        else:
            tmp = tensor.contiguous()
            _LIB.shm_broadcast(self.shm_comm, tmp.data_ptr(), nbytes, root_in_group, 0, _is_capturing())
            if self.rank_in_group != root_in_group:
                tensor.copy_(tmp)

    def send(self, tensor: torch.Tensor, dst: int | None = None) -> None:
        if dst is None:
            dst = (self.rank_in_group + 1) % self.world_size
        self.broadcast_inplace(tensor, root_in_group=self.rank_in_group)

    def recv(self, size: torch.Size, dtype: torch.dtype, src: int | None = None) -> torch.Tensor:
        if src is None:
            src = (self.rank_in_group - 1) % self.world_size
        out = torch.empty(size, dtype=dtype, device=f"npu:{self.device}")
        self.broadcast_inplace(out, root_in_group=src)
        return out

    def all_to_all(
        self,
        input_: torch.Tensor,
        scatter_dim: int = 0,
        gather_dim: int = -1,
        scatter_sizes: list[int] | None = None,
        gather_sizes: list[int] | None = None,
    ) -> torch.Tensor:
        if self.world_size <= 1:
            return input_
        if scatter_dim < 0:
            scatter_dim += input_.dim()
        if gather_dim < 0:
            gather_dim += input_.dim()

        if scatter_sizes is not None and gather_sizes is not None:
            input_list = [t.contiguous() for t in torch.split(input_, scatter_sizes, scatter_dim)]
            output_list = []
            tensor_shape_base = input_list[self.rank_in_group].size()
            for i in range(self.world_size):
                tensor_shape = list(tensor_shape_base)
                tensor_shape[gather_dim] = gather_sizes[i]
                output_list.append(torch.empty(tensor_shape, dtype=input_.dtype, device=input_.device))
        else:
            input_list = [t.contiguous() for t in torch.tensor_split(input_, self.world_size, scatter_dim)]
            output_list = [torch.empty_like(input_list[i]) for i in range(self.world_size)]

        dist.all_to_all(output_list, input_list, group=self.device_group)
        output_tensor = torch.cat(output_list, dim=gather_dim).contiguous()
        return output_tensor

    def barrier(self) -> None:
        if self.shm_comm:
            _LIB.shm_barrier(self.shm_comm)

    def destroy(self) -> None:
        pass


class _DummyWork:
    def wait(self, timeout=None):
        return True

    def is_completed(self):
        return True

    def is_success(self):
        return True


def _get_shm_communicator(group=None) -> NPUCommunicator | None:
    if group is not None and id(group) in _COMM_BY_PG:
        return _COMM_BY_PG[id(group)]
    try:
        if group is None or group is dist.GroupMember.WORLD:
            ws = dist.get_world_size()
            if ws <= 1:
                return None
            ranks_key = tuple(range(ws))
        else:
            ranks = dist.get_process_group_ranks(group)
            if len(ranks) <= 1:
                return None
            ranks_key = tuple(ranks)
        comm = _COMM_BY_RANKS.get(ranks_key)
        if comm is not None and group is not None:
            _COMM_BY_PG[id(group)] = comm
        return comm
    except Exception:
        return None


def _is_npu_tensor(t) -> bool:
    return isinstance(t, torch.Tensor) and t.device.type == "npu"


def _install_torch_distributed_shm_hooks() -> None:
    import torch.distributed.distributed_c10d as c10d
    from functools import wraps

    if getattr(dist, "_shm_hccl_patched", False):
        return

    orig_all_reduce = dist.all_reduce
    orig_broadcast = dist.broadcast
    orig_all_gather_into_tensor = dist.all_gather_into_tensor
    orig_all_gather = dist.all_gather
    orig_reduce_scatter_tensor = dist.reduce_scatter_tensor
    orig_reduce_scatter = dist.reduce_scatter
    orig_gather = dist.gather
    orig_all_to_all = dist.all_to_all
    orig_all_to_all_single = dist.all_to_all_single
    orig_barrier = dist.barrier

    @wraps(orig_all_reduce)
    def shm_all_reduce(tensor, op=dist.ReduceOp.SUM, group=None, async_op=False):
        if _is_npu_tensor(tensor):
            comm = _get_shm_communicator(group)
            if comm is not None and comm.world_size > 1:
                comm.all_reduce_inplace(tensor, op=op)
            return _DummyWork() if async_op else None
        return orig_all_reduce(tensor, op=op, group=group, async_op=async_op)

    @wraps(orig_broadcast)
    def shm_broadcast(tensor, src=0, group=None, async_op=False, group_src=None):
        if _is_npu_tensor(tensor):
            comm = _get_shm_communicator(group)
            if comm is not None and comm.world_size > 1:
                if group_src is not None:
                    root_in_group = int(group_src)
                else:
                    root_in_group = comm.ranks.index(int(src))
                comm.broadcast_inplace(tensor, root_in_group=root_in_group)
            return _DummyWork() if async_op else None
        if group_src is not None:
            return orig_broadcast(tensor, group=group, async_op=async_op, group_src=group_src)
        return orig_broadcast(tensor, src=src, group=group, async_op=async_op)

    @wraps(orig_all_gather_into_tensor)
    def shm_all_gather_into_tensor(output_tensor, input_tensor, group=None, async_op=False):
        if _is_npu_tensor(input_tensor):
            comm = _get_shm_communicator(group)
            if comm is not None and comm.world_size > 1:
                x = input_tensor.contiguous()
                in_bytes = x.numel() * x.element_size()
                _LIB.shm_all_gather(
                    comm.shm_comm,
                    output_tensor.data_ptr(),
                    x.data_ptr(),
                    in_bytes,
                    0,
                    _is_capturing(),
                )
            else:
                output_tensor.copy_(input_tensor)
            return _DummyWork() if async_op else None
        return orig_all_gather_into_tensor(output_tensor, input_tensor, group=group, async_op=async_op)

    @wraps(orig_all_gather)
    def shm_all_gather(tensor_list, tensor, group=None, async_op=False):
        if _is_npu_tensor(tensor):
            comm = _get_shm_communicator(group)
            if comm is not None and comm.world_size > 1:
                x = tensor.contiguous()
                in_bytes = x.numel() * x.element_size()
                buf = torch.empty((comm.world_size,) + x.size(), dtype=x.dtype, device=x.device)
                _LIB.shm_all_gather(
                    comm.shm_comm,
                    buf.data_ptr(),
                    x.data_ptr(),
                    in_bytes,
                    0,
                    _is_capturing(),
                )
                for i, t in enumerate(tensor_list):
                    t.copy_(buf[i])
            else:
                tensor_list[0].copy_(tensor)
            return _DummyWork() if async_op else None
        return orig_all_gather(tensor_list, tensor, group=group, async_op=async_op)

    @wraps(orig_reduce_scatter_tensor)
    def shm_reduce_scatter_tensor(output, input, op=dist.ReduceOp.SUM, group=None, async_op=False):
        if _is_npu_tensor(input):
            comm = _get_shm_communicator(group)
            if comm is not None and comm.world_size > 1:
                x = input.contiguous()
                _LIB.shm_reduce_scatter(
                    comm.shm_comm,
                    output.data_ptr(),
                    x.data_ptr(),
                    x.numel(),
                    _dtype_code(x.dtype),
                    _op_code(op),
                    0,
                    _is_capturing(),
                )
            else:
                output.copy_(input)
            return _DummyWork() if async_op else None
        return orig_reduce_scatter_tensor(output, input, op=op, group=group, async_op=async_op)

    @wraps(orig_reduce_scatter)
    def shm_reduce_scatter(output, input_list, op=dist.ReduceOp.SUM, group=None, async_op=False):
        if input_list and _is_npu_tensor(input_list[0]):
            stacked = torch.cat([t.contiguous() for t in input_list], dim=0)
            return shm_reduce_scatter_tensor(output, stacked, op=op, group=group, async_op=async_op)
        return orig_reduce_scatter(output, input_list, op=op, group=group, async_op=async_op)

    @wraps(orig_gather)
    def shm_gather(tensor, gather_list=None, dst=0, group=None, async_op=False, group_dst=None):
        if _is_npu_tensor(tensor):
            comm = _get_shm_communicator(group)
            if comm is not None and comm.world_size > 1:
                dst_in_group = int(group_dst) if group_dst is not None else comm.ranks.index(int(dst))
                x = tensor.contiguous()
                in_bytes = x.numel() * x.element_size()
                if comm.rank_in_group == dst_in_group:
                    buf = torch.empty((comm.world_size,) + x.size(), dtype=x.dtype, device=x.device)
                    _LIB.shm_gather(comm.shm_comm, buf.data_ptr(), x.data_ptr(), in_bytes, dst_in_group, 0, _is_capturing())
                    if gather_list is not None:
                        for i, t in enumerate(gather_list):
                            t.copy_(buf[i])
                else:
                    _LIB.shm_gather(comm.shm_comm, 0, x.data_ptr(), in_bytes, dst_in_group, 0, _is_capturing())
            else:
                if gather_list:
                    gather_list[0].copy_(tensor)
            return _DummyWork() if async_op else None
        if group_dst is not None:
            return orig_gather(tensor, gather_list=gather_list, group=group, async_op=async_op, group_dst=group_dst)
        return orig_gather(tensor, gather_list=gather_list, dst=dst, group=group, async_op=async_op)

    @wraps(orig_all_to_all)
    def shm_all_to_all(output_tensor_list, input_tensor_list, group=None, async_op=False):
        if input_tensor_list and _is_npu_tensor(input_tensor_list[0]):
            comm = _get_shm_communicator(group)
            if comm is not None and comm.world_size > 1:
                ws = comm.world_size
                # Uneven splits would need all ranks to agree on each
                # broadcast's byte count, which this transport cannot negotiate.
                shapes = {tuple(t.shape) for t in input_tensor_list}
                shapes |= {tuple(t.shape) for t in output_tensor_list}
                if len(shapes) != 1:
                    raise NotImplementedError(
                        f"shm all_to_all needs uniform split shapes, got {sorted(shapes)}"
                    )
                for src_r in range(ws):
                    for dst_r in range(ws):
                        # Every rank must enter shm_broadcast: it carries an
                        # all-rank barrier, so letting the src==dst rank take a
                        # local-copy shortcut hangs every other rank forever.
                        if comm.rank_in_group == src_r:
                            buf = input_tensor_list[dst_r].contiguous()
                        else:
                            # Scratch, not output_tensor_list[src_r]:
                            # .contiguous() would alias it and a later iteration
                            # would overwrite the finished result.
                            buf = torch.empty_like(output_tensor_list[src_r])
                        nbytes = buf.numel() * buf.element_size()
                        _LIB.shm_broadcast(comm.shm_comm, buf.data_ptr(), nbytes, src_r, 0, _is_capturing())
                        if comm.rank_in_group == dst_r:
                            output_tensor_list[src_r].copy_(buf)
            else:
                output_tensor_list[0].copy_(input_tensor_list[0])
            return _DummyWork() if async_op else None
        return orig_all_to_all(output_tensor_list, input_tensor_list, group=group, async_op=async_op)

    @wraps(orig_all_to_all_single)
    def shm_all_to_all_single(
        output,
        input,
        output_split_sizes=None,
        input_split_sizes=None,
        group=None,
        async_op=False,
    ):
        if _is_npu_tensor(input):
            comm = _get_shm_communicator(group)
            if comm is not None and comm.world_size > 1:
                ws = comm.world_size
                in_splits = (
                    list(torch.split(input, input_split_sizes, dim=0))
                    if input_split_sizes is not None
                    else list(torch.tensor_split(input, ws, dim=0))
                )
                out_splits = (
                    list(torch.split(output, output_split_sizes, dim=0))
                    if output_split_sizes is not None
                    else list(torch.tensor_split(output, ws, dim=0))
                )
                return shm_all_to_all(out_splits, in_splits, group=group, async_op=async_op)
            output.copy_(input)
            return _DummyWork() if async_op else None
        return orig_all_to_all_single(
            output,
            input,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
            async_op=async_op,
        )

    @wraps(orig_barrier)
    def shm_barrier(group=None, async_op=False, device_ids=None):
        if group is None or group is dist.GroupMember.WORLD or id(group) in _DEVICE_PG_IDS:
            comm = _get_shm_communicator(group)
            if comm is not None and comm.shm_comm:
                comm.barrier()
                return _DummyWork() if async_op else None
        return orig_barrier(group=group, async_op=async_op, device_ids=device_ids)

    for mod in (dist, c10d):
        mod.all_reduce = shm_all_reduce
        mod.broadcast = shm_broadcast
        mod.all_gather_into_tensor = shm_all_gather_into_tensor
        mod.all_gather = shm_all_gather
        mod.reduce_scatter_tensor = shm_reduce_scatter_tensor
        mod.reduce_scatter = shm_reduce_scatter
        mod.gather = shm_gather
        mod.all_to_all = shm_all_to_all
        mod.all_to_all_single = shm_all_to_all_single
        mod.barrier = shm_barrier

    dist._shm_hccl_patched = True


_install_torch_distributed_shm_hooks()
