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

from __future__ import annotations

import logging
from functools import wraps
from typing import Any, cast

import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
import vllm
from torch.distributed import Backend
from vllm.distributed.parallel_state import GroupCoordinator, _get_unique_name, _register_group

from vllm_ascend.distributed.device_communicators.npu_communicator import (
    NPUCommunicator,
    _COMM_BY_PG,
    _COMM_BY_RANKS,
    _DEVICE_PG_IDS,
    _LIB,
    _dtype_code,
    _is_capturing,
    _op_code,
)
from vllm_ascend.patch.worker._hccl_pg_registry import HcclPgKey, HcclPgRegistry, make_hccl_pg_key
from vllm_ascend.utils import create_hccl_pg_options

_HCCL_PG_REGISTRY = HcclPgRegistry()
logger = logging.getLogger(__name__)


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


def _is_npu_tensor(t: Any) -> bool:
    return isinstance(t, torch.Tensor) and t.device.type == "npu"


def _install_torch_distributed_shm_hooks() -> None:
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
                    _LIB.shm_gather(comm.shm_comm, buf.data_ptr(), x.data_ptr(), in_bytes, dst_in_group, 0)
                    if gather_list is not None:
                        for i, t in enumerate(gather_list):
                            t.copy_(buf[i])
                else:
                    _LIB.shm_gather(comm.shm_comm, 0, x.data_ptr(), in_bytes, dst_in_group, 0)
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
                for src_r in range(comm.world_size):
                    for dst_r in range(comm.world_size):
                        if comm.rank_in_group == src_r and src_r == dst_r:
                            output_tensor_list[src_r].copy_(input_tensor_list[dst_r])
                        else:
                            buf = (
                                input_tensor_list[dst_r].contiguous()
                                if comm.rank_in_group == src_r
                                else output_tensor_list[src_r].contiguous()
                            )
                            nbytes = buf.numel() * buf.element_size()
                            _LIB.shm_broadcast(comm.shm_comm, buf.data_ptr(), nbytes, src_r, 0)
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


def _normalize_backend(backend: str | Backend) -> str:
    return str(backend)


def _resolve_reuse_domain(group_name: str) -> str:
    group_base_name = group_name.split(":")[0]
    if "eplb" in group_base_name or group_base_name == "mc2":
        return group_base_name
    return "shared"


def _create_device_group(
    ranks: list[int],
    backend: str,
    hccl_pg_options: object,
):
    return torch.distributed.new_group(
        ranks,
        backend=backend,
        pg_options=hccl_pg_options,
    )


def _acquire_hccl_group(
    *,
    ranks: list[int],
    backend: str,
    hccl_pg_options: object,
    reuse_domain: str,
):
    hccl_key = make_hccl_pg_key(ranks, backend, hccl_pg_options, reuse_domain)
    device_group = _HCCL_PG_REGISTRY.acquire(
        ranks=ranks,
        backend=backend,
        pg_options=hccl_pg_options,
        reuse_domain=reuse_domain,
        create_fn=lambda: _create_device_group(ranks, backend, hccl_pg_options),
    )
    return device_group, hccl_key


def _wrap_destroy_distributed_environment(destroy_fn):
    if getattr(cast(Any, destroy_fn), "_hccl_registry_clearing_wrapped", False) is True:
        return destroy_fn

    @wraps(destroy_fn)
    def wrapped(*args, **kwargs):
        try:
            return destroy_fn(*args, **kwargs)
        finally:
            _HCCL_PG_REGISTRY.clear()

    cast(Any, wrapped)._hccl_registry_clearing_wrapped = True
    return wrapped


def _patch_destroy_distributed_environment():
    destroy_fn = _wrap_destroy_distributed_environment(vllm.distributed.parallel_state.destroy_distributed_environment)
    vllm.distributed.parallel_state.destroy_distributed_environment = destroy_fn
    vllm.distributed.destroy_distributed_environment = destroy_fn


class GroupCoordinatorPatch(GroupCoordinator):
    def __init__(
        self,
        group_ranks: list[list[int]],
        local_rank: int,
        torch_distributed_backend: str | Backend,
        use_device_communicator: bool,
        use_message_queue_broadcaster: bool = False,
        group_name: str | None = None,
    ):
        group_name = group_name or "anonymous"
        self.unique_name = _get_unique_name(group_name)
        _register_group(self)

        self.rank = torch.distributed.get_rank()
        self.local_rank = local_rank
        self.backend = _normalize_backend(torch_distributed_backend)
        self._acquired_hccl_keys: list[HcclPgKey] = []
        self._unshared_hccl_groups: list[object] = []
        self.use_device_communicator = use_device_communicator
        self.device_communicator: NPUCommunicator | None = None
        self.mq_broadcaster = None
        self.cpu_group = None
        self.device_group = None
        self.device = None
        self.use_custom_op_call = True
        self.use_cpu_custom_send_recv = False
        self.group_name = group_name
        self.group_ranks = group_ranks

        try:
            self._init_device_groups(create_cpu_group=True)
            assert self.cpu_group is not None
            assert self.device_group is not None

            self._init_device_communicator()

            from vllm.distributed.device_communicators.shm_broadcast import MessageQueue

            if use_message_queue_broadcaster and self.world_size > 1:
                self.mq_broadcaster = MessageQueue.create_from_process_group(
                    self.cpu_group,
                    1 << 22,
                    6,
                )
        except Exception:
            try:
                self.destroy()
            except Exception:
                logger.exception("Failed to clean up partially initialized GroupCoordinatorPatch")
            raise

    def _init_device_groups(self, create_cpu_group: bool) -> None:
        reuse_domain = _resolve_reuse_domain(self.group_name)
        self_device_group = None
        for ranks in self.group_ranks:
            hccl_pg_options = create_hccl_pg_options(self.group_name)
            device_group, hccl_key = _acquire_hccl_group(
                ranks=ranks,
                backend=self.backend,
                hccl_pg_options=hccl_pg_options,
                reuse_domain=reuse_domain,
            )
            if hccl_key is not None:
                self._acquired_hccl_keys.append(hccl_key)
            elif self.backend == "hccl" and self.rank in ranks:
                self._unshared_hccl_groups.append(device_group)

            cpu_group = torch.distributed.new_group(ranks, backend="gloo") if create_cpu_group else None
            if self.rank in ranks:
                if create_cpu_group:
                    self.ranks = ranks
                    self.world_size = len(ranks)
                    self.rank_in_group = ranks.index(self.rank)
                    self.cpu_group = cpu_group
                self_device_group = device_group

        if self_device_group is not None:
            self.device_group = self_device_group

    def _init_device_communicator(self) -> None:
        self.device = torch.npu.current_device()
        if self.world_size > 1:
            self.device_communicator = NPUCommunicator(
                cpu_group=self.cpu_group,
                device=self.device,
                device_group=self.device_group,
                unique_name=self.unique_name,
            )

    def _release_hccl_resources(self) -> bool:
        destroyed = False
        device_communicator = getattr(self, "device_communicator", None)
        if device_communicator is not None:
            device_communicator.destroy()
            self.device_communicator = None
            destroyed = True

        if hasattr(self, "_acquired_hccl_keys"):
            for hccl_key in reversed(self._acquired_hccl_keys):
                _HCCL_PG_REGISTRY.release(hccl_key)
            self._acquired_hccl_keys = []
            destroyed = True

        if hasattr(self, "_unshared_hccl_groups"):
            for device_group in reversed(self._unshared_hccl_groups):
                torch.distributed.destroy_process_group(device_group)
            self._unshared_hccl_groups = []
            destroyed = True

        return destroyed

    def destroy(self):
        if getattr(self, "mq_broadcaster", None) is not None:
            self.mq_broadcaster = None

        self._release_hccl_resources()

        device_group = getattr(self, "device_group", None)
        if device_group is not None and self.backend != "hccl":
            torch.distributed.destroy_process_group(device_group)
        if hasattr(self, "device_group"):
            del self.device_group

        cpu_group = getattr(self, "cpu_group", None)
        if cpu_group is not None:
            torch.distributed.destroy_process_group(cpu_group)
        if hasattr(self, "cpu_group"):
            del self.cpu_group

    def destroy_hccl(self) -> bool:
        destroyed = self._release_hccl_resources()
        if hasattr(self, "device_group"):
            self.device_group = None
        return destroyed

    def restore_hccl(self) -> bool:
        if self.device_group is not None:
            return False
        self._init_device_groups(create_cpu_group=False)
        assert self.device_group is not None
        self._init_device_communicator()
        return True

    def all_to_all(
        self,
        input_: torch.Tensor,
        scatter_dim: int = 0,
        gather_dim: int = -1,
        scatter_sizes: list[int] | None = None,
        gather_sizes: list[int] | None = None,
    ) -> torch.Tensor:
        if self.world_size == 1:
            return input_
        assert -input_.dim() <= scatter_dim < input_.dim(), (
            f"Invalid scatter dim ({scatter_dim}) for input tensor with shape {input_.size()}"
        )
        assert -input_.dim() <= gather_dim < input_.dim(), (
            f"Invalid gather dim ({gather_dim}) for input tensor with shape {input_.size()}"
        )
        assert self.device_communicator is not None, "device_communicator should be initialized when world_size > 1"
        return self.device_communicator.all_to_all(input_, scatter_dim, gather_dim, scatter_sizes, gather_sizes)


vllm.distributed.parallel_state.GroupCoordinator = GroupCoordinatorPatch
_patch_destroy_distributed_environment()
_install_torch_distributed_shm_hooks()
