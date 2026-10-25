# 昇腾 310P（Atlas 300I Pro）多卡 vLLM 部署与 Host 共享内存通信实战指南

> **适用场景**：搭载多张华为 **Atlas 300I Pro（Ascend 310P3）** 单芯 PCIe 推理卡、无板间 HCCS/RoCE 互联、跨卡 PCIe P2P 因硬件 BAR 空间限制无法使用原生 HCCL 的服务器（如鲲鹏 920 + 4× 310P3 部署 27B/32B 级大模型 `TP=4` 推理）。

---

两江市岸ai服务器（远程统一用域名连接）域名：ljsan.7766.org
用户名：root
密码：sctr@2026
(远程登录用这个)用户名：hw
密码：sctr@2026
ai服务器内网ip：192.168.1.13
    远程端口内部：9221，外部：60001
ai服务器内网ip：192.168.1.14
    远程端口内部：9221，外部：60002
ai服务器内网ip：192.168.1.15
    远程端口内部：9221，外部：60003
ai服务器内网ip：192.168.1.16
    远程端口内部：9221，外部：60004
本机SSH已经都配置了免密登录，可以直接用。可以先读一下配置。
## 一、环境背景与问题全貌

### 1.1 硬件与软件栈
- **服务器架构**：KunLun 2280（2× Kunpeng-920 64 核 `aarch64`，250 GiB 内存，2 个 NUMA 节点：node0 = CPU 0-31，node1 = CPU 32-63；每路 L3 32 MiB，L3 cache line 128 B）
- **NPU 硬件**：4× Atlas 300I Pro（`Ascend 310P3`，单卡 21.5 GB 显存，4 卡共 86 GB；NPU 0/1 位于 NUMA node 0，NPU 2/3 位于 NUMA node 1（由 `/sys/bus/pci/devices/<BDF>/numa_node` 实测，早期版本笔记写的「NUMA 4」不存在），卡间仅通过 CPU Root Complex 下的 PCIe Gen4 x16 拓扑连接）
- **宿主机驱动**：`npu-smi 26.1.1`（Driver `26.1.1`）
- **容器镜像**：`quay.io/ascend/vllm-ascend:v0.23.0.post1-310p`（内置 CANN `9.1.0`、`torch 2.10.0`、`torch_npu 2.10.0.post4`、`vllm 0.23.0`）
- **部署目标**：4 卡张量并行（`TP=4`）运行 `Eco-Tech/Qwen3.8-27B-w8a8-310p`（W8A8 量化，约 28 GB 权重），开启 `FULL_DECODE_ONLY` 图捕获（`ACLGraph`）与 `qwen3_5_mtp` 投机采样。

### 1.2 核心矛盾
- 单卡 21.5 GB 显存无法放下 28 GB 的 W8A8 权重，必须使用 2 卡或 4 卡 `TP` 并行。
- 但只要启动多卡 `vllm serve`（或 2 卡以上 `torch.distributed` `hccl` 后端），所有集合通信立刻崩溃。

---

## 二、排查与调优过程中踩到的 11 个核心“坑点”与解决方案

### 坑 1：HCCL 初始化报 `207001`（`Failed to allocate 32 bytes device memory requested by the HCCL module`）

- **报错现象**：
  启动多卡任务调用 `HcclCommInitRootInfoConfig` 时，尽管每张卡有 20 GB 以上空闲显存，HCCL 依然报错：
  ```text
  Resource_Error_Insufficient_Device_Memory(EL0019): Failed to allocate 32 bytes device memory requested by the HCCL module.
  ```
- **无效尝试**：
  调整 `HCCL_BUFFSIZE`、`HCCL_INTRA_PCIE_ENABLE=0`、`HCCL_INTRA_SHM_ENABLE=1`、`HCCL_SOCKET_IFNAME`、`HCCL_WHITELIST_DISABLE=1` 等环境变量**全部无效**，报错发生在 `HcclCommInit` 最底层的 32 字节 Notify/Sync 空间分配阶段。
- **逆向根因**：
  通过反汇编 CANN 9.1.0 的 `libhccl_plf.so` 发现：
  1. `hrtMalloc(size, ptr)` 内部调用 `hccl::Is310PDevice()`。
  2. `Is310PDevice()` 读取全局变量 `hccl::g_is310PDevice`（位于 `.bss` 段），但在 CANN 9.1.0 的整个 `libhccl*.so` 中，**没有任何代码将该变量置为 `true`**（恒为 `0`）！
  3. 当 `Is310PDevice() == 0` 时，`hrtMalloc` 调用 `halMemAlloc(ptr, size, flag=0x10801)`，其中 bit 16（`0x10000`）向内核驱动请求 **P2P BAR 物理页池（`nid=1`）**；而在 310P3 驱动中该内存池容量为 `0 pages`，导致哪怕申请 32 字节也立即返回 `207001`（OOM）。

---

### 坑 2：Patch `Is310PDevice()=1` 后，跨卡拷贝报内核态 `drv_devmm_host.ko` `ret=-14`

- **报错现象**：
  将 `libhccl_plf.so` 中的 `Is310PDevice()` 二进制 Patch 为固定返回 `1` 后，`halMemAlloc` 改用普通 HBM 标志（`0x801`），`HcclCommInit` 的 32 字节分配不再报 OOM；但紧接着在 HCCL 建链 `ExchangeMemHandle` / `aclrtMemcpy(ACL_MEMCPY_DEVICE_TO_DEVICE)` 时报错，宿主机 `dmesg` 输出：
  ```text
  [devmm] [devmm_FillPcieBarPa 123] [devmm_d2d_pa_to_bar_dma] Pcie fill pa-to-barpa fail. (devid=0; dst_devid=1; ret=-14)
  ```
- **硬件根因**：
  通过 `lspci -vvv -s 01:00.0` 检查 PCIe 配置空间发现：
  - Atlas 300I Pro（单芯 `310P3`）仅暴露了 **BAR0（8 MB）**、**BAR2（128 MB）**、**BAR4（64 KB）**。
  - 它**没有**训练卡（910B）或双芯卡上的大尺寸 64-bit Prefetchable HBM BAR 窗口（通常需 32 GB~64 GB）。
  - 因此，两张单芯 310P3 卡之间**在硬件物理层面不支持 PCIe P2P BAR DMA 直拷（D2D）**，任何跨卡数据传输必须经由 Host 内存（`D2H -> Host 共享内存 -> H2D`）中转。

---

### 坑 3：C/ctypes 自定义通信库中 `all_gather` / `reduce_scatter` 及 `NPUGraph` 输出全为 `0.0`（`torch_npu` 异步 `TaskQueue` 陷阱）

- **报错现象**：
  编写 `/dev/shm` Host 共享内存通信库 `libshm_hccl.so` 后，在 Python 中执行：
  ```python
  x = torch.full((1, 1280), float(rank + 1), dtype=torch.float16, device=f"npu:{rank}")
  stream = torch.npu.current_stream().npu_stream
  lib.shm_all_gather(comm, out.data_ptr(), x.data_ptr(), nbytes, ctypes.c_void_p(stream), 0)
  ```
  尽管 C 函数内部第一行就调用了 `aclrtSynchronizeStream(stream)`，再调用 `aclrtMemcpy(..., ACL_MEMCPY_DEVICE_TO_HOST)`，拷回 Host 的数据却**全部是 `0.0`**！
- **核心根因**：
  1. `torch_npu` 为了掩盖算子下发开销，默认启用了 **C++ 异步主机下发队列（`c10_npu::Repository` / `TaskQueue`）**。
  2. 当 Python 执行 `torch.full(...)` 时，只是把算子任务塞进了 CPU 内存里的 `TaskQueue`，由后台消费线程异步调用 `aclopCompileAndExecute` 下发到 `aclrtStream`。
  3. Python 侧访问 `torch.npu.current_stream().npu_stream` 在底层调用的是 `c10_npu::NPUStream::stream(false)`——注意参数是 **`false`**，它只返回 `aclrtStream` 指针，**绝不会等待 `TaskQueue` 排空**！
  4. 因此，当 C 函数拿到 `stream` 立即调用 `aclrtSynchronizeStream(stream)` 或 `aclrtMemcpyAsync(..., stream)` 时，`torch.full` 还在 `torch_npu` 的 CPU 队列里排队，根本还没进入 `aclrtStream`，导致 `aclrtMemcpy` 先于 `torch.full` 执行，读出未初始化的全 `0` 显存！
- **终极解决方案**：
  在 `shm_hccl.cpp` 中直接链接 `libtorch_npu.so`，声明并调用 `c10_npu::getCurrentNPUStream(-1).stream(true)`（对应导出符号 `_ZNK7c10_npu9NPUStream6streamEb`）：
  ```cpp
  namespace c10_npu {
  struct NPUStream {
      uint64_t data[2];
      ~NPUStream() {}
      void* stream(bool need_empty) const;
  };
  NPUStream getCurrentNPUStream(signed char device_index);
  }

  static inline void* flush_npu_stream() {
      c10_npu::NPUStream s = c10_npu::getCurrentNPUStream(-1);
      return s.stream(true); // 强制排空 torch_npu 的 CPU TaskQueue，且不阻塞等待硬件流完成
  }
  ```
  **为什么用 `stream(true)` 而不是 `torch.npu.synchronize()`？**
  因为 `stream(true)` 只排空 CPU 侧 `TaskQueue`（确保前置算子已按序入队到 `aclrtStream`），不会在硬件流上调用阻塞式的 `aclrtSynchronizeStream`。这不仅解决了 Eager 模式的乱序问题，而且在 **`NPUGraph`（`ACLGraph`）图捕获模式**下也不会破坏流捕获状态！

---

### 坑 4：让 Host 共享内存通信原生支持 `NPUGraph`（`ACLGraph`）捕获与回放

- **问题分析**：
  vLLM 在开启 `--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'` 时，会将 Decode 阶段的整张计算图（包含多次 `all_reduce`）录制进 `NPUGraph`。在图捕获期间（`torch.npu.is_current_stream_capturing() == True`）：
  - **不能**调用同步的 `aclrtSynchronizeStream` 或 `aclrtMemcpy`；
  - 每次 `graph.replay()` 都在 C++ 层直接重放录制好的流任务，**不会**再次经过 Python 函数调用。
- **解决方案（`aclrtMemcpyAsync` + `aclrtLaunchHostFunc`）**：
  1. 在 `shm_comm_init` 中通过 `aclrtMallocHost(&comm->host_bounce, bounce_size)` 为每个进程分配一块 Pinned Host 内存作为异步 DMA 中转区。
  2. 当检测到 `is_capturing == 1` 时，向当前 `aclrtStream` 顺序录制以下 3 个节点：
     ```cpp
     aclrtMemcpyAsync(comm->host_bounce, total_bytes, in_dev_ptr, total_bytes, ACL_MEMCPY_DEVICE_TO_HOST, stream);
     aclrtLaunchHostFunc(stream, host_all_reduce_cb, t);
     aclrtMemcpyAsync(out_dev_ptr, total_bytes, comm->host_bounce, total_bytes, ACL_MEMCPY_HOST_TO_DEVICE, stream);
     ```
  3. CANN Runtime 原生支持在 `NPUGraph` 中录制 `aclrtLaunchHostFunc`！每次 `graph.replay()` 执行到该节点时，ACL 流会自动触发 `host_all_reduce_cb` 回调，在 Host 侧完成 `/dev/shm` 原子屏障同步与 OpenMP 多线程规约，随后由下一条 `aclrtMemcpyAsync(H2D)` 拷回 NPU 显存。
  4. **实测性能**：4 卡 `[1, 5120]` `float16` 的 `w8a8 matmul + all_reduce` 整图回放单步耗时仅 **`0.662 ms`**。

---

### 坑 5：`shm_comm_init` 多进程初始化竞态条件与 `/dev/shm` 文件残留

- **报错现象**：
  早期版本在 `shm_comm_init` 中写成：
  ```cpp
  if (create_fresh && rank == 0) {
      memset(comm->region, 0, sizeof(ShmSync));
  }
  ```
  当使用 `torchrun` 启动 4 个进程时，Rank 1/2/3 可能比 Rank 0 先完成 `shm_comm_init` 并进入第一个 `shm_all_reduce`，将 `arrive_seq[1..3]` 设为 `1`；随后姗姗来迟的 Rank 0 执行 `memset(..., 0)` 把 `arrive_seq[1..3]` 重新清零，导致 4 个进程永久死锁在 `wait_seq`！
- **解决方案**：
  1. 每个 Rank 在 `shm_comm_init` 中只归零属于自己的 `arrive_seq[rank]`、`mid_seq[rank]`、`done_seq[rank]`。
  2. 引入单调递增的 `init_flag[rank]` 原子屏障，所有 Rank 必须等待全员到达当前初始化代次后才能从 `shm_comm_init` 返回：
     ```cpp
     uint64_t next_init = __atomic_load_n(&comm->region->sync.init_flag[rank], __ATOMIC_SEQ_CST) + 1;
     __atomic_store_n(&comm->region->sync.init_flag[rank], next_init, __ATOMIC_SEQ_CST);
     wait_seq(comm->region->sync.init_flag, world_size, next_init);
     if (rank == 0 && create_fresh) {
         unlink(path); // 全员 mmap 完成后立即 unlink，进程退出后内核自动释放 /dev/shm 空间
     }
     ```

---

### 坑 6：`vllm-ascend` 中的 `torch.distributed` 旁路调用与 Hook 挂载时机陷阱

- **报错现象**：
  仅替换 `NPUCommunicator` 的 `all_reduce` / `all_gather` / `reduce_scatter` 后，运行到 `get_tp_group().broadcast_tensor_dict(...)` 时仍然触发 `HcclCommInitRootInfoConfig` 报 `207001` 错误。
- **根因分析**：
  1. `vllm` 的 `GroupCoordinator.broadcast_tensor_dict`、`gather` 等方法并不走 `self.device_communicator`，而是直接调用 `torch.distributed.broadcast(tensor, src=..., group=self.device_group, async_op=True)`。
  2. 如果把 `torch.distributed` 的 Monkey-Patch 放在 `vllm_ascend/patch/worker/patch_distributed.py` 中，当某些进程或测试只触发 `vllm_ascend.platform`（动态加载 `npu_communicator.py`）而尚未导入 `vllm_ascend.worker` 时，`patch_distributed.py` 根本还没被执行！
- **解决方案**：
  1. **将 `_install_torch_distributed_shm_hooks()` 直接放在 `npu_communicator.py` 模块底部**，只要 `npu_communicator` 被导入就立即替换 `torch.distributed` 和 `torch.distributed.distributed_c10d` 的 `all_reduce`、`broadcast`、`all_gather_into_tensor`、`all_gather`、`reduce_scatter_tensor`、`reduce_scatter`、`gather`、`all_to_all`、`all_to_all_single` 和 `barrier`。
  2. **按 `tuple(ranks)` 全局复用 `ShmComm`**：在 `GroupCoordinatorPatch._init_device_communicator` 中，只要 `world_size > 1`（包括 `use_device_communicator=False` 的 `_WORLD` 组），统一实例化 `NPUCommunicator` 并按 `tuple(self.ranks)` 共享同一个底层 `ShmComm` 指针。这样无论调用方传入 `group=None`（默认 `WORLD`）、`group=_WORLD.device_group` 还是 `group=_TP.device_group`，全部路由到同一个步调一致的 `ShmComm`，且节省了重复的 `/dev/shm` 和 Pinned Host 内存开销。

---

### 坑 7：`torch.compile` 图融合 Pass 隐式调用 `get_hccl_comm_name` 触发 HCCL 崩溃

- **报错现象**：
  在 `vllm-ascend` 开启 `torch.compile` 时，`vllm_ascend/compilation/graph_fusion_pass_manager.py` 默认启用 `fuse_allreduce_rms: True`，实例化 `MatmulAllReduceAddRMSNormPass`。该 Pass 在 `__init__` 中执行：
  ```python
  backend = device_group._get_backend(torch.device("npu"))
  self.tp_group_name = backend.get_hccl_comm_name(self.local_rank)
  ```
  调用 `ProcessGroupHCCL::getHcclCommName` 会立即在 C++ 层强制初始化硬件 HCCL 通信域并触发 `207001` 报错。
- **解决方案**：
  在 `vllm serve` 启动参数 `--additional-config` 中显式关闭 `fuse_allreduce_rms`：
  ```bash
  --additional-config '{"ascend_compilation_config": {"enable_npugraph_ex": false, "fuse_allreduce_rms": false}}'
  ```

---

### 坑 8：310P 上单独验证 `npu_quant_matmul`（W8A8）时输出全 `0.0`

- **报错现象**：
  编写独立脚本测试 `torch_npu.npu_quant_matmul(x_int8, w_nz, scale, output_dtype=torch.float16)` 时，如果传入 `scale = torch.ones((N,), dtype=torch.int64, device="npu")`，算子不报错但输出全是 `0.0`。
- **根因与解法**：
  1. 在 310P 上，`npu_quant_matmul` 要求右矩阵权重必须先转置再转换为 CANN 专用的 **`FRACTAL_NZ`（format 29）**：
     ```python
     w_nz = torch_npu.npu_format_cast(w_int8.t().contiguous(), 29)
     ```
  2. `deq_scale` 虽然是 `int64` 张量，但其低 32 位存储的是 **IEEE 754 `float32` 的二进制位模式（bit pattern）**！如果填入整数 `1`（`0x0000000000000001`），硬件将其低 32 位当作 `float32` 解析即 `1.4e-45`，乘完直接下溢为 `0.0`。
  3. 正确做法：使用 `torch_npu.npu_trans_quant_param(scale_fp32)` 转换，或在测试脚本中填入 `1.0f` 对应的十六进制位模式 `0x3f800000`：
     ```python
     scale = torch.full((N,), 0x3f800000, dtype=torch.int64, device="npu")
     ```

---

### 坑 9：`NPUGraph` 录制 3 节点导致 130 次 `all_reduce` 耗时 `71.47 ms/step`（单节点 `HostFunc` 直拷 + Ring Buffer + ARMv8.2-A NEON 提速 `6.26x`）

- **性能瓶颈现象**：
  模型成功跑通后，在 `Qwen3.8-27B`（64 层 + 1 层 MTP，每步 Decode 共 **130 次 `all_reduce`**，每次约 `20 KB`）下，单步 Decode 耗时约 `161~195 ms`。使用微基准脚本 `/home/models/shm_hccl/bench_comm.py` 拆解 `NPUGraph` 内的 130 次 `all_reduce` 发现：
  - 初版每条 `all_reduce` 录制 3 个图节点：`aclrtMemcpyAsync(D2H)` + `aclrtLaunchHostFunc` + `aclrtMemcpyAsync(H2D)`，130 次合计耗时高达 **`71.47 ms/step`（`0.550 ms/op`）**！
  - 单独测试发现：130 次 `aclrtMemcpyAsync`（D2H + H2D）在 NPU TS/SDMA 调度器上的任务下发与切换开销就占了 **`58.13 ms/step`（`0.447 ms/op`）**，而 130 次空 `aclrtLaunchHostFunc` 仅耗时 **`5.94 ms/step`（`0.046 ms/op`）**！
- **深度优化方案（4 项组合拳）**：
  1. **单图节点回调内同步直拷（消除 SDMA 任务调度开销）**：
     在 `is_capturing == 1` 时，**不再录制 `aclrtMemcpyAsync` 图节点**，而是每次集合通信**只向 `NPUGraph` 录制 1 个 `aclrtLaunchHostFunc` 节点**。在 `shm_comm_init` 时通过 `aclrtGetCurrentContext(&comm->acl_ctx)` 记录当前 NPU 上下文，在 `HostFunc` 回调线程中通过 TLS 缓存调用 `aclrtSetCurrentContext(comm->acl_ctx)` 绑定上下文后，直接调用同步的 `aclrtMemcpy(D2H)` 和 `aclrtMemcpy(H2D)`（底层直接走 PCIe DMA ioctl，单次 `20 KB` 拷贝仅耗时 `~15 μs`）。
  2. **无锁 Ring Buffer + 单次屏障（消除 `mid_seq` / `done_seq` 双重自旋等待）**：
     在 `ShmRegion` 中开辟 `ring_buf[RING_SLOTS][MAX_RANKS][RING_SLOT_BYTES]`。对于 `<= RING_SLOT_BYTES` 的张量（覆盖 Decode 阶段全部 `all_reduce` / `all_gather` / `reduce_scatter`），第 `step` 步直接使用第 `step % RING_SLOTS` 号槽位，每个 Rank 将自己的输入拷入 `slot_buf[rank]` 后，**只需等待 1 次 `wait_arrive` 屏障**，即可各自在本地缓存的 `host_bounce`（`posix_memalign` 分配的 L1/L2 可缓存内存，替代原先 Uncached 的 `aclrtMallocHost`）中完成全部规约并拷回 NPU，彻底省去了 `wait_mid` 和 `wait_done` 两次跨进程屏障！

     关于 `RING_SLOTS` 的取值：初版用了 64 槽（`ring_buf` 占 128 MB）。实际上 `wait_arrive` 是全员屏障，Rank B 要推进到第 `N+64` 步，必须先通过第 `N+1..N+63` 每一步的屏障，这要求 Rank A 的 `arrive_seq` 已经到 `N+63`，也就意味着 A 早就读完了第 `N` 步的槽位。**屏障本身就提供了保护，正确性只需要 2 槽**。现已降到 `RING_SLOTS = 4`，`ring_buf` 从 128 MB 降到 16 MB，`ShmRegion` 总量 640 MB → 528 MB。
     `RING_SLOT_BYTES` 则从 256 KB 提到 **512 KB**：`hidden = 5120` 的 fp16 张量在 `cudagraph_capture_sizes = 32`（`max_num_seqs 8` × MTP `1+3`）时是 320 KB，恰好越过 256 KB 的界,导致最大那张图的 130 次 `all_reduce` 全部退回 3 屏障的 `in_buf` 慢路径。
  3. **128 字节 Cache Line 对齐（消除鲲鹏 920 跨 NUMA False Sharing）**：
     将原先紧凑排列的 `arrive_seq[MAX_RANKS]` 改为每个 Rank 独占一个 **`__attribute__((aligned(128))) RankSync`** 结构体（鲲鹏 920 L3 Cache Line 为 128 字节），避免 4 个进程自旋写各自序号时触发跨 NUMA 缓存行乒乓失效。
  4. **ARMv8.2-A FP16/FP32 NEON SIMD 向量化规约（消除小张量 OpenMP 线程池唤醒开销）**：
     使用 `<arm_neon.h>`（编译加 `-march=armv8.2-a+fp16`）直接用 `vld1q_f16` / `vcvt_f32_f16` / `vaddq_f32` 对 4 卡/2 卡 `float16` 和 `float32` 做 SIMD 向量化累加，且对 `< 262144` 元素的张量关闭 OpenMP 多线程（避免 10K 元素张量唤醒 4 个线程反而变慢）。
- **优化实测对比（4 卡 130× `all_reduce [2, 5120] fp16`）**：
  - 优化前：`71.47 ms/step`（`0.550 ms/op`）
  - 优化后：**`11.42 ms/step`（`0.088 ms/op`）**——**通信提速 `6.26 倍`，每个 Decode 步直接节省 `60.05 ms`！**

---

### 坑 10：`worker_310p.py` 默认将可用 KV Cache 显存砍半（`// 2`），导致 4 卡 86 GB 显存下上下文受限

- **现象**：
  4 张 Atlas 300I Pro 共有 `86 GB` 显存，加载 `28 GB` 的 `Qwen3.8-27B-w8a8` 权重（每卡仅占 `6.96 GiB`）后，即使设置 `--gpu-memory-utilization 0.85`，日志却显示 `Available KV cache memory: 1.70 GiB`，单卡显存只用了 `15.2 GB / 21.5 GB`，每张卡白白闲置了 `~6 GB` 显存！
- **源码根因**：
  检查 `/vllm-workspace/vllm-ascend/vllm_ascend/_310p/worker/worker_310p.py` 中的 `NPUWorker310.determine_available_memory()` 发现：
  ```python
  available_kv_cache_memory = int(
      total_npu_memory * self.cache_config.gpu_memory_utilization
      - peak_memory
      - non_torch_memory
  ) // 2  # <--- 310P Worker 硬编码将剩余可用显存直接除以 2！
  ```
  因为这个 `// 2`，原本每卡剩余可用的 `~3.4~5.5 GiB` KV Cache 空间被强行砍半到了 `1.70 GiB`。
- **解决方案**：
  在 `vllm serve` 启动参数中显式指定 `--kv-cache-memory 5368709120`（每卡固定分配 **5.0 GiB**，4 卡合计 **20.0 GiB KV Cache**），这会直接绕过 `determine_available_memory()` 的砍半逻辑！同时将 `--max-model-len` 从 `32768` 提升至 **`131072`（128K 上下文）**：
  - 优化前：每卡 KV Cache `1.70 GiB`（总容量 `95,495` tokens），最大上下文 `32K`。
  - 优化后：每卡 KV Cache **`5.00 GiB`**（总容量 **`146,580` tokens**），最大上下文 **`128K`（`131,072` tokens）**，单卡显存利用率提升至 `18,710 / 21,524 MB`（预留 2.8 GB 安全裕量防长序列激活峰值 OOM）。

---

### 坑 11：`vllm-ascend` 310P 随机采样器（`sampler_310p.py`）CPU 回退耗时 `30 ms/step` 及 `torch.topk` AICPU 退化陷阱

- **性能瓶颈现象**：
  开启随机采样（`temperature > 0`）时，每步 Decode 比 Greedy（`temperature = 0`）慢了整整 **`33 ms/step`**！
- **源码根因**：
  1. 检查 `/vllm-workspace/vllm-ascend/vllm_ascend/_310p/sample/sampler.py` 发现：为了规避多卡随机种子不同步问题，`fill_exponential_310p` 在每次采样时都在 CPU 上分配 `[B, 248320]`（词表大小 `248,320`，`float32` 约 `1~2 MB`）张量，在 CPU 单线程执行 `q_cpu.exponential_()`，再通过 PCIe 拷回 NPU。由于开启了 MTP 投机采样，每步 Decode 会调用 **3 次**（Target 采样 + Rejection 采样 + MTP Draft 采样），仅 CPU 端生成指数分布随机数就白白消耗了 **`30~33 ms/step`**！
  2. **千万不要盲目开启 `enable_reduce_sample: true`**：在 `vllm_ascend/sample/sampler.py` 中，若开启 `enable_reduce_sample: true`，采样器会先对 `float32` logits 调用 `torch.topk(logits, k=20)`。然而在昇腾 310P 上，`float32` 的 `torch.topk` 没有 AI Core 硬件算子支持，会**回退到 AICPU 执行，单次耗时高达 `467.13 ms`**！而默认的 `probs.sort(dim=-1, descending=False)` 在 AI Core 上仅需 **`2.23 ms`**。
- **终极解决方案（`/home/models/shm_hccl/sampler_310p.py`）**：
  实测在 310P 的 NPU 上直接创建固定相同初始种子的 `torch.Generator(device="npu").manual_seed(20260417)` 并调用 `q.exponential_(generator=npu_gen)`，不仅 4 张卡生成的随机数**逐比特完全一致（`max diff = 0.0`）**，而且单次耗时从 `10+ ms` 降至 **`1.14 ms`**！我们将优化后的 `/home/models/shm_hccl/sampler_310p.py` 挂载覆盖容器内的 `/vllm-workspace/vllm-ascend/vllm_ascend/_310p/sample/sampler.py`，每步随机采样直接再省下 **`~27 ms`**。

---

## 二点五、通信开销的实测分解，以及三个被证伪的优化假设

坑 9 把 Decode 的 `all_reduce` 从 `0.550 ms/op` 压到 `0.085 ms/op` 之后，再想往下压就得先知道这 `85 μs` 花在哪。用 `probe.cpp`（单进程单卡，只往 `NPUGraph` 里录空回调或纯拷贝回调）量出来的分解是：

```
130 次空 aclrtLaunchHostFunc          3.20 ms/step   24.6 μs/op   <- 派发硬地板
130 次 回调内 aclrtMemcpy D2H  20 KB   5.20 ms/step   40.0 μs/op
130 次 回调内 D2H + H2D       20 KB   7.36 ms/step   56.6 μs/op
130 次 回调内 D2H + H2D      320 KB  20.69 ms/step  159.2 μs/op
```

对照完整 `all_reduce` 的 `85 μs/op`（20 KB）：`24.6 μs` 是 `aclrtLaunchHostFunc` 的派发开销，`32 μs` 是两次 `aclrtMemcpy` 的固定 ioctl 开销（单次约 `15 μs`，与拷贝多少字节几乎无关），剩下约 `28 μs` 才是屏障加规约。

**结论：Decode 小张量上 67% 的时间是派发与 ioctl 的固定开销，和数据量无关。** 要再快就只能减少 Host 回调的次数，而 65 层 × 2 次 `all_reduce` 是模型结构决定的，减不掉。所以 Decode 这条路径已经接近这个架构的下限，不要再在上面投入。

真正还有空间的是 Prefill。`max-num-batched-tokens 8192` 时每次 `all_reduce` 的张量是 `8192 × 5120 × 2 = 80 MB`，远超 `SLOT_SIZE`，走的是 `in_buf` 分块循环，而那条路径上的 `reduce_slice()` 原先完全没有 NEON 特化，只有 `reduce_ring_to_dst()` 有。补上 NEON 并把 OpenMP 线程数按规模分级（`>= 4M` 元素开 16 线程，而非原先无论多大都只开 4 线程）后的实测：

| 单次 `all_reduce` 张量 | 优化前 | 优化后 | 变化 |
| :--- | :--- | :--- | :--- |
| `512 × 5120` fp16（5 MiB） | `3.01 ms` | `2.47 ms` | −17.9% |
| `2048 × 5120` fp16（20 MiB） | `9.99 ms` | `8.52 ms` | −14.7% |
| `8192 × 5120` fp16（80 MiB，生产配置） | `38.73 ms` | `30.35 ms` | **−21.6%** |
| `16384 × 5120` fp16（160 MiB） | `80.22 ms` | `58.58 ms` | **−27.0%** |

按生产配置换算：每个 8192-token 的 Prefill chunk 有 130 次 `all_reduce`，`38.73 ms × 130 = 5.03 s` → `30.35 ms × 130 = 3.95 s`。100K 上下文约 13 个 chunk，TTFT 中这部分从约 `65 s` 降到约 `51 s`，即 220 s 的冷启动 Prefill 能省下约 `14 s`（约 6%）。

### 三个被实测证伪的假设（别再走这几条路）

**一、「`/dev/shm` 不是 pinned 内存，D2H 会多一次 staging 拷贝」——错。**
`probe2.cpp` 直接对比了三种 Host 内存的 D2H 带宽（32 MiB / 256 MiB）：

```
posix_memalign（未注册）   10.59 / 11.13 GiB/s
aclrtMallocHost（pinned）  11.74 / 12.10 GiB/s
tmpfs mmap（ShmRegion）    11.37 / 12.01 GiB/s
```

tmpfs mmap 和 `aclrtMallocHost` 基本同速，CANN 对非 pinned 目标处理得很好。 **这条路径上不存在值得消除的双拷贝。**

**二、「用 `aclrtHostRegister` 把 shm 区域注册成 pinned 可以提速」——这个 API 在 310P 上不可用。**
`libascendcl.so` 里符号是有的（`aclrtHostRegister` / `aclrtHostRegisterV2` / `aclrtHostUnregister`），头文件签名是 `aclrtHostRegister(void *ptr, uint64_t size, aclrtHostRegisterType type, void **devPtr)`，但对 `posix_memalign` 内存和 tmpfs mmap、32 MiB 和 256 MiB，全部返回 `207000`。这段代码已经从 `shm_hccl.cpp` 里删掉了。

**三、「把 `HostFunc` 回调线程绑到 NPU 所在的 NUMA node 能减少跨 die 访存」——实测更慢。**
CANN 的回调线程池确实不受 `enable_cpu_binding` 控制（实测 worker 的 affinity 是 `0-63`，也就是说 `enable_cpu_binding: true` 其实没绑住）。但用 `sched_setaffinity` 把回调绑到本 socket 后，除 320 KB 一档外每档都退化：

```
                  不绑        绑到本 socket
20 KB    10.90 ms/step   12.17 ms/step
240 KB   51.13 ms/step   57.61 ms/step
320 KB   81.90 ms/step   78.70 ms/step
640 KB  101.35 ms/step  116.55 ms/step
```

规约要读全部 4 个 Rank 的槽位，其中两个无论如何都在对面 socket，绑核并不能改变这一点，只是剥夺了调度器的回避空间，还会和 vLLM 自己的线程抢核。这段代码也已删掉。

> 「测量方法提示」：服务在跑的情况下同配置两次测量能差 15%（320 KB 一档测到过 70.86 和 81.90）。做 A/B 必须交替多轮取各自最小值，单次对比在这个噪声下没有意义。

### 两个潜伏的正确性问题（已修）

**`shm_broadcast` / `shm_gather` 的 ctypes 参数个数与 C 签名不一致。**
C 侧是 6 个和 7 个参数（末位 `is_capturing`），Python 侧 `argtypes` 只声明了 5 个和 6 个，调用也少传一个。aarch64 AAPCS 下 `is_capturing` 落在 `w5`/`w6`，读到的是上一次调用残留的寄存器值。这两个函数体恰好都没用这个参数，所以一直没炸——纯粹靠运气。现已补齐，并在两个函数入口加了捕获期告警。

**`all_to_all` 在 `src_r == dst_r` 时必然死锁。**
原逻辑里 `rank == src_r == dst_r` 的那个进程走本地 `copy_` 捷径、不调 `shm_broadcast`，而其余进程都调了；`shm_broadcast` 内有全员 `wait_arrive` 屏障，于是剩下 3 个进程永久自旋。Dense + MTP 的 `Qwen3.8-27B` 不走 MoE，这条路现在没人踩，但 3.5 节是写给迁移复用的，下一个人换 MoE 模型就会撞上。顺带还有一个别名 bug：非 root 分支用 `output_tensor_list[src_r].contiguous()` 当接收缓冲，而 `.contiguous()` 对已连续张量返回原张量本身，后续迭代会把已经写好的结果覆盖掉。两者都已修，并对不等分 split 显式抛 `NotImplementedError` 而不是假装支持。

---

## 三、完整落地方案、性能对比与第二台服务器快速复用手册

### 3.1 优化前后性能与容量对比（4× Atlas 300I Pro 部署 `Qwen3.8-27B-w8a8`）

| 指标项 | 优化前（初版 SHM + 默认配置） | 优化后（单节点直拷 Ring + NPU 采样 + Prefix Cache + Async + MTP=3 + ReduceSample） | 提升幅度 |
| :--- | :--- | :--- | :--- |
| **最大上下文长度 (`--max-model-len`)** | `32,768` (32K) | **`102,400` (100K) / `131,072` (128K)** | **3.1x ~ 4.0x** |
| **KV + Mamba 缓存分配显存** | `1.70 GiB`/卡（共 `6.8 GiB`，`95,495` tokens） | **`5.00 GiB`/卡（共 `20.0 GiB`，`142,809 ~ 167,424` tokens 物理块池）** | **2.94x** |
| **4 卡 130× `all_reduce` 图回放耗时** | `71.47 ms/step` (`0.550 ms/op`) | **`11.42 ms/step` (`0.088 ms/op`)** | **6.26x 提速（省 `60.05 ms/step`）** |
| **10K 多轮对话次轮 Prefill 耗时 (TTFT)** | `19.27 s` (`515 tok/s`，无前缀缓存) | **`2.32 s`（等效吞吐 `4,291 tok/s`，开启 `--enable-prefix-caching`）** | **8.31x 提速（立省 `16.95 s`）** |
| **Greedy 推理速度 (`temperature=0.0`)** | `10.34 tok/s` (`161.21 ms/step`) | **`21.02 ~ 25.78 tok/s`**（`--async-scheduling` + MTP `num_speculative_tokens: 3` + `enable_reduce_sample`） | **+103.3% ~ +149.3%（最高约 2.5 倍）** |
| **Thinking 推理速度 (`temperature=0.6`)** | `~9.1 tok/s` (`~190 ms/step`) | **`14.50 ~ 19.95 tok/s`** | **+59.3% ~ +119.2% 提速** |
| **高随机推理速度 (`temperature=1.0`)** | `8.55 tok/s` (`194.88 ms/step`) | **`13.74 tok/s` (`112.84 ms/step`)** | **+60.7% 提速** |

> **调用提示**：模型目录 `generation_config.json` 已默认设置为 `"temperature": 0.0, "do_sample": false`。由于 `Qwen3_5MTP` 投机采样使用 Greedy 生成候选草稿词，客户端请求时传入 `"temperature": 0.0`（代码/视觉 OCR/确定性问答，可达 **`21.0 ~ 25.8 tok/s`**）或 `"temperature": 0.6` 可获得最高的投机采样接受率与吞吐表现。同时开启 `--enable-prefix-caching` 与 `--async-scheduling` 后，Mamba 层自动工作于 512-token 对齐缓存模式（`mamba_cache_mode='align'`），多轮 Agent 工具调用无需重复 Prefill 历史上下文。
> **关于 `W8A8SC` 稀疏压缩量化的说明**：经核查华为最新 `msmodelslim 26.1.0` 源码与 `vllm-ascend` 实现，`W8A8S/W8A8SC` 仅在旧版 `modelslim_v0` 引擎中支持纯文本 `Qwen2/Qwen3-Dense`，而 `Qwen3.5/Qwen3.8-27B`（混合 Mamba `GatedDeltaNet` + ViT + MTP）采用的 `multimodal_vlm_modelslim_v1` 引擎尚未实现 `W8A8S` IR；且 `vllm-ascend` 下 `W8A8SC`（`--load-format sharded_state`）要求强制关闭 `--enable-prefix-caching` 且不支持 `qwen3_5_mtp` 投机解码。因此 `W8A8_DYNAMIC + Prefix Caching + Async + MTP=3` 是当前 310P 上综合延迟与吞吐最优的方案。

### 3.1.1 长文本（`10K → 100K`）Prefill 与 Decode 性能实测表（单并发冷启动）

| 目标长度 | 实际输入 Token | 冷启动 Prefill 耗时 (TTFT) | Prefill 吞吐 (`tok/s`) | 输出 Token | Decode 耗时 | Decode 速度 (`tok/s`, MTP=1 基线) | 单步延迟 TPOT (`ms/tok`) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **10K** | `9,935` | `16.79 s`（前缀缓存命中时 **`2.32 s`**） | **591.7 tok/s**（缓存命中 **`4,291 tok/s`**） | `100` | `5.47 s` | **18.09 tok/s**（MTP=2/3 达 **`22.69 tok/s`**） | `55.28 ms`（MTP=2/3 达 **`44.07 ms`**） |
| **20K** | `19,955` | `34.48 s` | **578.7 tok/s** | `100` | `6.02 s` | **16.45 tok/s** | `60.80 ms` |
| **30K** | `29,975` | `53.81 s` | **557.0 tok/s** | `100` | `6.23 s` | **15.90 tok/s** | `62.90 ms` |
| **40K** | `39,935` | `73.73 s` | **541.6 tok/s** | `100` | `6.45 s` | **15.34 tok/s** | `65.20 ms` |
| **50K** | `49,955` | `95.52 s` | **523.0 tok/s** | `100` | `6.45 s` | **15.36 tok/s** | `65.11 ms` |
| **60K** | `59,975` | `117.63 s` | **509.9 tok/s** | `100` | `7.05 s` | **14.04 tok/s** | `71.24 ms` |
| **70K** | `69,935` | `143.18 s` | **488.4 tok/s** | `100` | `7.00 s` | **14.15 tok/s** | `70.68 ms` |
| **80K** | `79,955` | `166.46 s` | **480.3 tok/s** | `100` | `7.21 s` | **13.73 tok/s** | `72.82 ms` |
| **90K** | `89,975` | `193.48 s` | **465.0 tok/s** | `100` | `7.64 s` | **12.95 tok/s** | `77.22 ms` |
| **100K** | `99,935` | `220.67 s` | **452.9 tok/s** | `100` | `7.65 s` | **12.94 tok/s** | `77.27 ms` |

### 3.1.2 多模态识图性能实测表（客户端自动 `PNG → JPG` 压缩 + `--async-scheduling` + `MTP=3`）

| 测试场景 | 原始体积 → 发送体积 | 视觉+文本 Token | 优化前 TTFT | 优化后 TTFT (`PNG→JPG`) | 优化前 Decode | 优化后 Decode (`Async+MTP=3`) | 识别准确率 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1. 插画与艺术字 (`夏日汽水与猫.png`)** | `2180.8 KB PNG` → **`500.6 KB JPG`** | `1,564` | `11.83 s` | **`6.55 s` (-44.6%)** | `18.30 tok/s` | **`21.02 tok/s`** | **100%** |
| **2. 书法大字与极小字 (`破昼问天.png`)** | `1888.5 KB PNG` → **`362.0 KB JPG`** | `1,562` | `12.42 s` | **`5.48 s` (-55.9%)** | `19.45 tok/s` | **`24.57 tok/s`** | **100%** |
| **3. 软件 UI 参数提取 (`screenshot.png`)** | `90.6 KB JPG` → **`90.6 KB JPG`** | `791` | `3.05 s` | **`2.86 s` (-6.2%)** | `20.52 tok/s` | **`25.78 tok/s`** | **100%** |
| **4. 高速监控多目标 (`20260815...jpg`)** | `332.4 KB JPG` → **`332.4 KB JPG`** | `2,124` | `5.86 s` | **`5.86 s`** | `18.16 tok/s` | **`22.75 tok/s`** | **100%** |
| **5. 13 宫格高密海报 (`poster-programmatic.png`)** | `2483.3 KB PNG` → **`776.8 KB JPG`** | `1,937` | `13.73 s` | **`7.58 s` (-44.8%)** | `19.34 tok/s` | **`24.19 tok/s`** | **96%+** |

---

### 3.2 核心文件清单（`/home/models/shm_hccl/`）

在 `ljsan.7766.org:60001` 上，所有代码与编译产物均已持久化保存在 `/home/models/shm_hccl/` 目录下：

| 文件路径 | 作用说明 |
| :--- | :--- |
| `/home/models/shm_hccl/shm_hccl.cpp` | Host 共享内存集合通信 C++ 源码（含 `flush_npu_stream`、单节点 `HostFunc` 直拷、4 槽 × 512 KB Ring Buffer、128B 对齐，`reduce_ring_to_dst` 与 `reduce_slice` 双路 ARMv8.2-A FP16 NEON） |
| `/home/models/shm_hccl/libshm_hccl.so` | 编译后的动态库（链接 `libascendcl.so` 与 `libtorch_npu.so`） |
| `/home/models/shm_hccl/npu_communicator.py` | 挂载替换容器内 `/vllm-workspace/vllm-ascend/vllm_ascend/distributed/device_communicators/npu_communicator.py` |
| `/home/models/shm_hccl/patch_distributed.py` | 挂载替换容器内 `/vllm-workspace/vllm-ascend/vllm_ascend/patch/worker/patch_distributed.py` |
| `/home/models/shm_hccl/sampler_310p.py` | 挂载替换容器内 `/vllm-workspace/vllm-ascend/vllm_ascend/_310p/sample/sampler.py`（NPU 原生指数分布采样） |
| `/home/models/shm_hccl/bench_comm.py` | 4 卡 130× `all_reduce` `NPUGraph` 微基准性能测试脚本 |
| `/home/models/shm_hccl/test_vllm_comm.py` | 4 卡全算子与 `NPUGraph` 功能快速验收脚本 |
| `/home/models/shm_hccl/test_shm_matrix.py` | dtype × op × 尺寸 × 算子的正确性矩阵（89 项），尺寸刻意跨过 `RING_SLOT_BYTES` 与 `SLOT_SIZE` 两个阈值，确保每条内部路径都被覆盖。**改 `shm_hccl.cpp` 后必须先过这个再换 `.so`** |
| `/home/models/shm_hccl/test_bcast_a2a.py` | 覆盖矩阵测不到的 `shm_broadcast`（4 个 root）、`shm_gather`（4 个 dst）与 `all_to_all` 4×4，用真实的 6/7 参数签名调用 |
| `/home/models/shm_hccl/bench_prefill.py` | Prefill 形状（`512/2048/8192/16384 × 5120` fp16）的 `all_reduce` 基准，5 轮取最小值 |
| `/home/models/shm_hccl/bench_shapes.py` | Decode 形状扫描（`2/8/24/32/64` tokens）的 130× `NPUGraph` `all_reduce` 基准 |
| `/home/models/shm_hccl/probe.cpp` / `probe2.cpp` | 开销分解探针：`aclrtLaunchHostFunc` 派发地板、回调内纯拷贝成本；以及三种 Host 内存的 D2H 带宽对比 |
| `/home/models/shm_hccl/build_shm.sh` | 编译脚本，`build_shm.sh <src.cpp> <out.so>`，自动解析 `torch_npu` 的 lib 路径 |
| `/home/models/run_vllm.sh` | 4 卡 `qwen38-27b` 生产环境容器启动脚本（128K 上下文 + 20 GiB KV Cache + Prefix Caching + MTP=2） |

### 3.3 编译 `libshm_hccl.so` 的标准命令（容器内执行）
如果后续升级镜像版本需要重新编译 `libshm_hccl.so`，可在容器内执行（注意加上 `-march=armv8.2-a+fp16` 开启鲲鹏 920 FP16 NEON 指令集）：
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
TORCH_NPU_LIB=$(python3 -c "import torch_npu, os; print(os.path.join(os.path.dirname(torch_npu.__file__), 'lib'))")
g++ -O3 -march=armv8.2-a+fp16 -shared -fPIC -fopenmp /home/models/shm_hccl/shm_hccl.cpp \
    -L/usr/local/Ascend/ascend-toolkit/latest/lib64 \
    -L${TORCH_NPU_LIB} \
    -lascendcl -ltorch_npu \
    -Wl,-rpath,/usr/local/Ascend/ascend-toolkit/latest/lib64:${TORCH_NPU_LIB} \
    -o /home/models/shm_hccl/libshm_hccl.so
```

> **升级镜像后必做的 ABI 校验**：`shm_hccl.cpp` 里手写了 `c10_npu::NPUStream` 的结构体布局（`uint64_t data[2]`）来调用内部符号 `_ZNK7c10_npu9NPUStream6streamEb`，这是在猜 `torch_npu` 的私有 ABI。一旦 `torch_npu` 改了这个类的大小或布局，返回值传递就会错位，而且很可能不崩溃、只是静默返回错误的 stream——表现又是输出全 `0.0`，等于坑 3 原地复发。所以换镜像后先验证两件事再上线：
> ```bash
> # 1. 符号还在吗
> nm -D $(python3 -c "import torch_npu,os;print(os.path.dirname(torch_npu.__file__))")/lib/libtorch_npu.so \
>     | grep _ZNK7c10_npu9NPUStream6streamEb
> # 2. 正确性矩阵过不过
> torchrun --nproc_per_node=4 test_shm_matrix.py && torchrun --nproc_per_node=4 test_bcast_a2a.py
> ```

### 3.4 标准容器启动脚本（`/home/models/run_vllm.sh`）
```bash
#!/usr/bin/env bash
set -e
sudo docker rm -f qwen38-27b >/dev/null 2>&1 || true
sudo docker run -d --name qwen38-27b \
    --restart unless-stopped \
    --pid=host \
    --net=host \
    --ipc=host \
    --device /dev/davinci0 \
    --device /dev/davinci2 \
    --device /dev/davinci4 \
    --device /dev/davinci6 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /home/models:/home/models \
    -v /home/models/shm_hccl/npu_communicator.py:/vllm-workspace/vllm-ascend/vllm_ascend/distributed/device_communicators/npu_communicator.py \
    -v /home/models/shm_hccl/patch_distributed.py:/vllm-workspace/vllm-ascend/vllm_ascend/patch/worker/patch_distributed.py \
    -v /home/models/shm_hccl/sampler_310p.py:/vllm-workspace/vllm-ascend/vllm_ascend/_310p/sample/sampler.py \
    -e PYTORCH_NPU_ALLOC_CONF=expandable_segments:True \
    -e SHM_HCCL_LIB_PATH=/home/models/shm_hccl/libshm_hccl.so \
    quay.io/ascend/vllm-ascend:v0.23.0.post1-310p \
    vllm serve /home/models/Qwen3.8-27B-w8a8-310p \
    --served-model-name Qwen/Qwen3.8-27B Qwen3.8-27B \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 4 \
    --quantization ascend \
    --dtype float16 \
    --mamba-ssm-cache-dtype float16 \
    --max-model-len 131072 \
    --kv-cache-memory 5368709120 \
    --enable-prefix-caching \
    --async-scheduling \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 8 \
    --speculative-config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 3}' \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]}' \
    --additional-config '{"enable_cpu_binding": true, "enable_reduce_sample": true, "ascend_compilation_config": {"enable_npugraph_ex": false, "fuse_allreduce_rms": false}}' \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3
```

### 3.5 迁移到第二台服务器的操作步骤

四台机器统一用域名 `ljsan.7766.org` 连接，远程登录用户 `hw`，各机内网 IP 与外部端口对应关系：

| 内网 IP | 外部端口 | 说明 |
| :--- | :--- | :--- |
| `192.168.1.13` | `60001` | 主力机，当前跑 `qwen38-27b` |
| `192.168.1.14` | `60002` | 第二台（早期版本笔记误记为 `192.168.1.11`） |
| `192.168.1.15` | `60003` | |
| `192.168.1.16` | `60004` | |

当需要在另一台同配置服务器上部署 4 卡模型时，无需重新排查，只需执行以下三步：
1. 将 `192.168.1.13` 上的 `/home/models/shm_hccl` 目录及 `/home/models/run_vllm.sh` 同步过去：
   ```bash
   scp -r /home/models/shm_hccl /home/models/run_vllm.sh hw@192.168.1.14:/home/models/
   ```
2. 确保启动容器时包含 `--ipc=host`（使多进程共享大容量 `/dev/shm`）并挂载 `npu_communicator.py`、`patch_distributed.py` 和 `sampler_310p.py`。
3. 确保包含 `--kv-cache-memory 5368709120` 与 `--additional-config '{"ascend_compilation_config": {"enable_npugraph_ex": false, "fuse_allreduce_rms": false}}'`。

---

## 四、核心源码完整参考（可直接复制使用）

以下为已在 4× Atlas 300I Pro（`Ascend 310P3`）上通过压测与 `Qwen3.8-27B` 端到端验证的完整源码，下次遇到同类环境可直接复制创建。

### 4.1 `/home/models/shm_hccl/shm_hccl.cpp`（C++ 核心通信库）

```cpp
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sched.h>
#include <omp.h>
#if defined(__aarch64__)
#include <arm_neon.h>
#endif

#define MAX_RANKS 8
#define SLOT_SIZE (32UL * 1024UL * 1024UL)
// The all-rank wait_arrive barrier already orders step N against step N-1, so
// two ring slots suffice for correctness; 4 keeps margin while leaving the whole
// ring L3-resident (Kunpeng-920: 32 MiB L3 per socket, 128 B lines).
#define RING_SLOTS 4
// Must cover the largest tensor the decode graph reduces, or the capture
// callback silently degrades to the 3-barrier in_buf path. hidden=5120 fp16 at
// cudagraph_capture_sizes=32 (max_num_seqs 8 x MTP 1+3) is 320 KiB.
#define RING_SLOT_BYTES (512UL * 1024UL)

typedef struct __attribute__((aligned(128))) {
    volatile uint64_t init_seq;
    volatile uint64_t arrive_seq;
    volatile uint64_t mid_seq;
    volatile uint64_t done_seq;
    uint8_t pad[128 - 32];
} RankSync;

typedef struct {
    RankSync rank_sync[MAX_RANKS];
    uint8_t pad[4096 - sizeof(RankSync) * MAX_RANKS];
    uint8_t __attribute__((aligned(128))) ring_buf[RING_SLOTS][MAX_RANKS][RING_SLOT_BYTES];
    uint8_t __attribute__((aligned(128))) in_buf[MAX_RANKS][SLOT_SIZE];
    uint8_t __attribute__((aligned(128))) out_buf[SLOT_SIZE * MAX_RANKS];
} ShmRegion;

static_assert(sizeof(RankSync) == 128, "RankSync must be exactly one 128 B L3 line");
static_assert(sizeof(RankSync) * MAX_RANKS < 4096, "rank_sync[] must leave room for pad[]");
static_assert((RING_SLOTS & (RING_SLOTS - 1)) == 0, "RING_SLOTS must be a power of two");

typedef void (*aclrtHostFunc)(void *args);

// Internal torch_npu symbol to drain its C++ async task queue
namespace c10_npu {
struct NPUStream {
    uint64_t data[2];
    ~NPUStream() {}
    void* stream(bool need_empty) const;
};
NPUStream getCurrentNPUStream(signed char device_index);
}

static inline void* flush_npu_stream() {
    c10_npu::NPUStream s = c10_npu::getCurrentNPUStream(-1);
    return s.stream(true);
}

extern "C" {

extern int aclrtSynchronizeStream(void *stream);
extern int aclrtMemcpy(void *dst, size_t destMax, const void *src, size_t count, int kind);
extern int aclrtMemcpyAsync(void *dst, size_t destMax, const void *src, size_t count, int kind, void *stream);
extern int aclrtMallocHost(void **hostPtr, size_t size);
extern int aclrtFreeHost(void *hostPtr);
extern int aclrtLaunchHostFunc(void *stream, aclrtHostFunc fn, void *args);
extern int aclrtGetCurrentContext(void **context);
extern int aclrtSetCurrentContext(void *context);

#define ACL_MEMCPY_HOST_TO_HOST 0
#define ACL_MEMCPY_HOST_TO_DEVICE 1
#define ACL_MEMCPY_DEVICE_TO_HOST 2
#define ACL_MEMCPY_DEVICE_TO_DEVICE 3

static inline void wait_init(RankSync *rs, int world_size, uint64_t step) {
    for (int i = 0; i < world_size; i++) {
        uint64_t spins = 0;
        while (__atomic_load_n(&rs[i].init_seq, __ATOMIC_ACQUIRE) < step) {
            if (++spins > 200000) {
                sched_yield();
                spins = 0;
            } else {
                __asm__ volatile("yield" ::: "memory");
            }
        }
    }
}

static inline void wait_arrive(RankSync *rs, int world_size, uint64_t step) {
    for (int i = 0; i < world_size; i++) {
        uint64_t spins = 0;
        while (__atomic_load_n(&rs[i].arrive_seq, __ATOMIC_ACQUIRE) < step) {
            if (++spins > 200000) {
                sched_yield();
                spins = 0;
            } else {
                __asm__ volatile("yield" ::: "memory");
            }
        }
    }
}

static inline void wait_mid(RankSync *rs, int world_size, uint64_t step) {
    for (int i = 0; i < world_size; i++) {
        uint64_t spins = 0;
        while (__atomic_load_n(&rs[i].mid_seq, __ATOMIC_ACQUIRE) < step) {
            if (++spins > 200000) {
                sched_yield();
                spins = 0;
            } else {
                __asm__ volatile("yield" ::: "memory");
            }
        }
    }
}

static inline void wait_done(RankSync *rs, int world_size, uint64_t step) {
    for (int i = 0; i < world_size; i++) {
        uint64_t spins = 0;
        while (__atomic_load_n(&rs[i].done_seq, __ATOMIC_ACQUIRE) < step) {
            if (++spins > 200000) {
                sched_yield();
                spins = 0;
            } else {
                __asm__ volatile("yield" ::: "memory");
            }
        }
    }
}

// NOTE: ShmGraphTask allocations below are deliberately never freed. The
// recorded NPUGraph node holds the pointer and every graph.replay() dereferences
// it, so freeing it in the callback would hand the second replay a dangling
// pointer. Bounded by capture_sizes x collectives per process (tens of KiB).
typedef struct {
    ShmRegion *region;
    void *host_bounce;
    void *acl_ctx;
    size_t bounce_size;
    uint64_t step;
    int world_size;
    int rank;
} ShmComm;

static __thread void *tls_acl_ctx = NULL;
// aclrtLaunchHostFunc callbacks run on a CANN runtime thread, not the vLLM
// worker thread, so they do not inherit its ACL context.
// Measured and rejected: pinning this thread to the NPU-local socket via
// sched_setaffinity was slower at every size but 320 KiB.
static inline void ensure_acl_ctx(ShmComm *comm) {
    if (comm->acl_ctx != NULL && tls_acl_ctx != comm->acl_ctx) {
        aclrtSetCurrentContext(comm->acl_ctx);
        tls_acl_ctx = comm->acl_ctx;
    }
}

// A collective too large for the capture path falls through to the synchronous
// branch, which calls aclrtSynchronizeStream on a capturing stream and silently
// corrupts the graph. Make that loud rather than mysterious.
static void warn_capture_unsupported(const char *fn, size_t bytes) {
    static int warned = 0;
    if (__atomic_exchange_n(&warned, 1, __ATOMIC_SEQ_CST)) return;
    fprintf(stderr,
            "[shm_hccl] %s: %zu bytes while the stream is capturing, but the "
            "graph path accepts at most %zu and broadcast/gather have no graph "
            "path at all. The synchronous fallback will corrupt the captured "
            "graph.\n",
            fn, bytes, (size_t)SLOT_SIZE);
    fflush(stderr);
}

ShmComm* shm_comm_init(const char *name, int world_size, int rank, int create_fresh) {
    char path[256];
    snprintf(path, sizeof(path), "/dev/shm/vllm_shm_%s", name);
    int fd = open(path, O_RDWR | O_CREAT, 0666);
    if (fd < 0) return NULL;
    size_t total_size = sizeof(ShmRegion);
    if (ftruncate(fd, total_size) != 0) {
        close(fd);
        return NULL;
    }
    void *ptr = mmap(NULL, total_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (ptr == MAP_FAILED) return NULL;

    ShmComm *comm = (ShmComm*)malloc(sizeof(ShmComm));
    comm->region = (ShmRegion*)ptr;
    comm->step = 0;
    comm->world_size = world_size;
    comm->rank = rank;
    comm->bounce_size = SLOT_SIZE * (size_t)world_size;
    comm->host_bounce = NULL;
    comm->acl_ctx = NULL;
    aclrtGetCurrentContext(&comm->acl_ctx);
    if (posix_memalign(&comm->host_bounce, 128, comm->bounce_size) != 0) comm->host_bounce = NULL;


    comm->region->rank_sync[rank].arrive_seq = 0;
    comm->region->rank_sync[rank].mid_seq = 0;
    comm->region->rank_sync[rank].done_seq = 0;
    uint64_t next_init = __atomic_load_n(&comm->region->rank_sync[rank].init_seq, __ATOMIC_SEQ_CST) + 1;
    __atomic_store_n(&comm->region->rank_sync[rank].init_seq, next_init, __ATOMIC_SEQ_CST);
    wait_init(comm->region->rank_sync, world_size, next_init);

    if (rank == 0 && create_fresh) {
        unlink(path);
    }
    return comm;
}

void shm_comm_destroy(ShmComm *comm) {
    if (!comm) return;
    if (comm->host_bounce) {
        free(comm->host_bounce);
        comm->host_bounce = NULL;
    }
    if (comm->region) {
        munmap(comm->region, sizeof(ShmRegion));
        comm->region = NULL;
    }
    free(comm);
}

int shm_barrier(ShmComm *comm) {
    if (!comm) return 0;
    ShmRegion *reg = comm->region;
    int rank = comm->rank;
    int ws = comm->world_size;
    uint64_t step = __atomic_add_fetch(&comm->step, 1, __ATOMIC_SEQ_CST);
    __atomic_store_n(&reg->rank_sync[rank].arrive_seq, step, __ATOMIC_RELEASE);
    wait_arrive(reg->rank_sync, ws, step);
    return 0;
}

static size_t dtype_size(int dtype) {
    switch (dtype) {
        case 0: return 2; // float16
        case 1: return 4; // float32
        case 2: return 8; // int64
        case 3: return 4; // int32
        case 4: return 1; // int8
        case 5: return 1; // uint8 / bool
        case 6: return 2; // bfloat16
        default: return 1;
    }
}

static inline float bf16_to_fp32(uint16_t u) {
    uint32_t bits = ((uint32_t)u) << 16;
    float f;
    memcpy(&f, &bits, sizeof(float));
    return f;
}

static inline uint16_t fp32_to_bf16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(float));
    uint32_t rounding_bias = ((bits >> 16) & 1) + 0x7FFF;
    return (uint16_t)((bits + rounding_bias) >> 16);
}

// Fast ring reduction directly into dst_ptr[0 .. (end_elem - start_elem)]
static inline void reduce_ring_to_dst(
    uint8_t *dst_ptr,
    uint8_t slot_buf[MAX_RANKS][RING_SLOT_BYTES],
    int world_size,
    size_t start_elem,
    size_t end_elem,
    int dtype,
    int op
) {
    if (start_elem >= end_elem) return;
    size_t count = end_elem - start_elem;

#if defined(__aarch64__)
    if (dtype == 0 && op == 0) { // float16 SUM
        float16_t *dst = (float16_t*)dst_ptr;
        if (world_size == 4) {
            const float16_t *s0 = (const float16_t*)slot_buf[0] + start_elem;
            const float16_t *s1 = (const float16_t*)slot_buf[1] + start_elem;
            const float16_t *s2 = (const float16_t*)slot_buf[2] + start_elem;
            const float16_t *s3 = (const float16_t*)slot_buf[3] + start_elem;
            size_t i = 0;
            for (; i + 8 <= count; i += 8) {
                float16x8_t v0 = vld1q_f16(s0 + i);
                float16x8_t v1 = vld1q_f16(s1 + i);
                float16x8_t v2 = vld1q_f16(s2 + i);
                float16x8_t v3 = vld1q_f16(s3 + i);
                float32x4_t lo = vaddq_f32(
                    vaddq_f32(vcvt_f32_f16(vget_low_f16(v0)), vcvt_f32_f16(vget_low_f16(v1))),
                    vaddq_f32(vcvt_f32_f16(vget_low_f16(v2)), vcvt_f32_f16(vget_low_f16(v3)))
                );
                float32x4_t hi = vaddq_f32(
                    vaddq_f32(vcvt_f32_f16(vget_high_f16(v0)), vcvt_f32_f16(vget_high_f16(v1))),
                    vaddq_f32(vcvt_f32_f16(vget_high_f16(v2)), vcvt_f32_f16(vget_high_f16(v3)))
                );
                vst1q_f16(dst + i, vcombine_f16(vcvt_f16_f32(lo), vcvt_f16_f32(hi)));
            }
            for (; i < count; i++) {
                float acc = (float)s0[i] + (float)s1[i] + (float)s2[i] + (float)s3[i];
                dst[i] = (float16_t)acc;
            }
            return;
        } else if (world_size == 2) {
            const float16_t *s0 = (const float16_t*)slot_buf[0] + start_elem;
            const float16_t *s1 = (const float16_t*)slot_buf[1] + start_elem;
            size_t i = 0;
            for (; i + 8 <= count; i += 8) {
                float16x8_t v0 = vld1q_f16(s0 + i);
                float16x8_t v1 = vld1q_f16(s1 + i);
                float32x4_t lo = vaddq_f32(vcvt_f32_f16(vget_low_f16(v0)), vcvt_f32_f16(vget_low_f16(v1)));
                float32x4_t hi = vaddq_f32(vcvt_f32_f16(vget_high_f16(v0)), vcvt_f32_f16(vget_high_f16(v1)));
                vst1q_f16(dst + i, vcombine_f16(vcvt_f16_f32(lo), vcvt_f16_f32(hi)));
            }
            for (; i < count; i++) {
                dst[i] = (float16_t)((float)s0[i] + (float)s1[i]);
            }
            return;
        }
    } else if (dtype == 1 && op == 0 && world_size == 4) { // float32 SUM
        float *dst = (float*)dst_ptr;
        const float *s0 = (const float*)slot_buf[0] + start_elem;
        const float *s1 = (const float*)slot_buf[1] + start_elem;
        const float *s2 = (const float*)slot_buf[2] + start_elem;
        const float *s3 = (const float*)slot_buf[3] + start_elem;
        size_t i = 0;
        for (; i + 4 <= count; i += 4) {
            float32x4_t v = vaddq_f32(
                vaddq_f32(vld1q_f32(s0 + i), vld1q_f32(s1 + i)),
                vaddq_f32(vld1q_f32(s2 + i), vld1q_f32(s3 + i))
            );
            vst1q_f32(dst + i, v);
        }
        for (; i < count; i++) {
            dst[i] = s0[i] + s1[i] + s2[i] + s3[i];
        }
        return;
    }
#endif

    if (dtype == 0) { // float16
        __fp16 *dst = (__fp16*)dst_ptr;
        for (size_t idx = 0; idx < count; idx++) {
            size_t i = start_elem + idx;
            float acc = (float)(((__fp16*)slot_buf[0])[i]);
            for (int r = 1; r < world_size; r++) {
                float v = (float)(((__fp16*)slot_buf[r])[i]);
                if (op == 0) acc += v;
                else if (op == 1) { if (v > acc) acc = v; }
                else if (op == 2) { if (v < acc) acc = v; }
                else if (op == 3) acc *= v;
            }
            dst[idx] = (__fp16)acc;
        }
    } else if (dtype == 1) { // float32
        float *dst = (float*)dst_ptr;
        for (size_t idx = 0; idx < count; idx++) {
            size_t i = start_elem + idx;
            float acc = ((float*)slot_buf[0])[i];
            for (int r = 1; r < world_size; r++) {
                float v = ((float*)slot_buf[r])[i];
                if (op == 0) acc += v;
                else if (op == 1) { if (v > acc) acc = v; }
                else if (op == 2) { if (v < acc) acc = v; }
                else if (op == 3) acc *= v;
            }
            dst[idx] = acc;
        }
    } else if (dtype == 2) { // int64
        int64_t *dst = (int64_t*)dst_ptr;
        for (size_t idx = 0; idx < count; idx++) {
            size_t i = start_elem + idx;
            int64_t acc = ((int64_t*)slot_buf[0])[i];
            for (int r = 1; r < world_size; r++) {
                int64_t v = ((int64_t*)slot_buf[r])[i];
                if (op == 0) acc += v;
                else if (op == 1) { if (v > acc) acc = v; }
                else if (op == 2) { if (v < acc) acc = v; }
                else if (op == 3) acc *= v;
            }
            dst[idx] = acc;
        }
    } else if (dtype == 3) { // int32
        int32_t *dst = (int32_t*)dst_ptr;
        for (size_t idx = 0; idx < count; idx++) {
            size_t i = start_elem + idx;
            int32_t acc = ((int32_t*)slot_buf[0])[i];
            for (int r = 1; r < world_size; r++) {
                int32_t v = ((int32_t*)slot_buf[r])[i];
                if (op == 0) acc += v;
                else if (op == 1) { if (v > acc) acc = v; }
                else if (op == 2) { if (v < acc) acc = v; }
                else if (op == 3) acc *= v;
            }
            dst[idx] = acc;
        }
    } else if (dtype == 4 || dtype == 5) { // int8 / uint8
        uint8_t *dst = (uint8_t*)dst_ptr;
        for (size_t idx = 0; idx < count; idx++) {
            size_t i = start_elem + idx;
            uint8_t acc = slot_buf[0][i];
            for (int r = 1; r < world_size; r++) {
                uint8_t v = slot_buf[r][i];
                if (op == 0) acc += v;
                else if (op == 1) { if (v > acc) acc = v; }
                else if (op == 2) { if (v < acc) acc = v; }
            }
            dst[idx] = acc;
        }
    } else if (dtype == 6) { // bfloat16
        uint16_t *dst = (uint16_t*)dst_ptr;
        for (size_t idx = 0; idx < count; idx++) {
            size_t i = start_elem + idx;
            float acc = bf16_to_fp32(((uint16_t*)slot_buf[0])[i]);
            for (int r = 1; r < world_size; r++) {
                float v = bf16_to_fp32(((uint16_t*)slot_buf[r])[i]);
                if (op == 0) acc += v;
                else if (op == 1) { if (v > acc) acc = v; }
                else if (op == 2) { if (v < acc) acc = v; }
                else if (op == 3) acc *= v;
            }
            dst[idx] = fp32_to_bf16(acc);
        }
    }
}

static void reduce_slice(
    uint8_t *out_ptr,
    uint8_t in_buf[MAX_RANKS][SLOT_SIZE],
    int world_size,
    size_t start_elem,
    size_t end_elem,
    int dtype,
    int op
) {
    if (start_elem >= end_elem) return;
    size_t count = end_elem - start_elem;
    // Prefill reduces ~8 M elements per slice; 4 threads leaves most of the
    // socket idle. Decode stays single-threaded so a 10 K tensor never pays for
    // a thread-pool wakeup.
    int nthreads = 1;
    if (count >= 4194304) nthreads = 16;
    else if (count >= 1048576) nthreads = 8;
    else if (count >= 262144) nthreads = 4;

#if defined(__aarch64__)
    // reduce_ring_to_dst has a NEON kernel but this function did not, so every
    // reduction above RING_SLOT_BYTES -- i.e. all of prefill -- ran scalar with
    // the op test in the innermost loop.
    if (op == 0 && dtype == 0) { // float16 SUM
        float16_t *dst = (float16_t*)out_ptr;
        #pragma omp parallel for schedule(static) num_threads(nthreads) if(nthreads > 1)
        for (size_t blk = 0; blk < count; blk += 8) {
            size_t i = start_elem + blk;
            size_t rem = count - blk;
            if (rem >= 8) {
                float32x4_t lo = vdupq_n_f32(0.0f);
                float32x4_t hi = vdupq_n_f32(0.0f);
                for (int r = 0; r < world_size; r++) {
                    float16x8_t v = vld1q_f16((const float16_t*)in_buf[r] + i);
                    lo = vaddq_f32(lo, vcvt_f32_f16(vget_low_f16(v)));
                    hi = vaddq_f32(hi, vcvt_f32_f16(vget_high_f16(v)));
                }
                vst1q_f16(dst + i, vcombine_f16(vcvt_f16_f32(lo), vcvt_f16_f32(hi)));
            } else {
                for (size_t k = 0; k < rem; k++) {
                    float acc = 0.0f;
                    for (int r = 0; r < world_size; r++) acc += (float)(((const float16_t*)in_buf[r])[i + k]);
                    dst[i + k] = (float16_t)acc;
                }
            }
        }
        return;
    }
    if (op == 0 && dtype == 1) { // float32 SUM
        float *dst = (float*)out_ptr;
        #pragma omp parallel for schedule(static) num_threads(nthreads) if(nthreads > 1)
        for (size_t blk = 0; blk < count; blk += 4) {
            size_t i = start_elem + blk;
            size_t rem = count - blk;
            if (rem >= 4) {
                float32x4_t acc = vdupq_n_f32(0.0f);
                for (int r = 0; r < world_size; r++) acc = vaddq_f32(acc, vld1q_f32((const float*)in_buf[r] + i));
                vst1q_f32(dst + i, acc);
            } else {
                for (size_t k = 0; k < rem; k++) {
                    float a = 0.0f;
                    for (int r = 0; r < world_size; r++) a += ((const float*)in_buf[r])[i + k];
                    dst[i + k] = a;
                }
            }
        }
        return;
    }
#endif

    if (dtype == 0) { // float16
        __fp16 *dst = (__fp16*)out_ptr;
        #pragma omp parallel for schedule(static) num_threads(nthreads) if(count >= 262144)
        for (size_t i = start_elem; i < end_elem; i++) {
            float acc = (float)(((__fp16*)in_buf[0])[i]);
            for (int r = 1; r < world_size; r++) {
                float v = (float)(((__fp16*)in_buf[r])[i]);
                if (op == 0) acc += v;
                else if (op == 1) { if (v > acc) acc = v; }
                else if (op == 2) { if (v < acc) acc = v; }
                else if (op == 3) acc *= v;
            }
            dst[i] = (__fp16)acc;
        }
    } else if (dtype == 1) { // float32
        float *dst = (float*)out_ptr;
        #pragma omp parallel for schedule(static) num_threads(nthreads) if(count >= 262144)
        for (size_t i = start_elem; i < end_elem; i++) {
            float acc = ((float*)in_buf[0])[i];
            for (int r = 1; r < world_size; r++) {
                float v = ((float*)in_buf[r])[i];
                if (op == 0) acc += v;
                else if (op == 1) { if (v > acc) acc = v; }
                else if (op == 2) { if (v < acc) acc = v; }
                else if (op == 3) acc *= v;
            }
            dst[i] = acc;
        }
    } else if (dtype == 2) { // int64
        int64_t *dst = (int64_t*)out_ptr;
        #pragma omp parallel for schedule(static) num_threads(nthreads) if(count >= 262144)
        for (size_t i = start_elem; i < end_elem; i++) {
            int64_t acc = ((int64_t*)in_buf[0])[i];
            for (int r = 1; r < world_size; r++) {
                int64_t v = ((int64_t*)in_buf[r])[i];
                if (op == 0) acc += v;
                else if (op == 1) { if (v > acc) acc = v; }
                else if (op == 2) { if (v < acc) acc = v; }
                else if (op == 3) acc *= v;
            }
            dst[i] = acc;
        }
    } else if (dtype == 3) { // int32
        int32_t *dst = (int32_t*)out_ptr;
        #pragma omp parallel for schedule(static) num_threads(nthreads) if(count >= 262144)
        for (size_t i = start_elem; i < end_elem; i++) {
            int32_t acc = ((int32_t*)in_buf[0])[i];
            for (int r = 1; r < world_size; r++) {
                int32_t v = ((int32_t*)in_buf[r])[i];
                if (op == 0) acc += v;
                else if (op == 1) { if (v > acc) acc = v; }
                else if (op == 2) { if (v < acc) acc = v; }
                else if (op == 3) acc *= v;
            }
            dst[i] = acc;
        }
    } else if (dtype == 4 || dtype == 5) { // int8 / uint8
        uint8_t *dst = (uint8_t*)out_ptr;
        for (size_t i = start_elem; i < end_elem; i++) {
            uint8_t acc = in_buf[0][i];
            for (int r = 1; r < world_size; r++) {
                uint8_t v = in_buf[r][i];
                if (op == 0) acc += v;
                else if (op == 1) { if (v > acc) acc = v; }
                else if (op == 2) { if (v < acc) acc = v; }
            }
            dst[i] = acc;
        }
    } else if (dtype == 6) { // bfloat16
        uint16_t *dst = (uint16_t*)out_ptr;
        #pragma omp parallel for schedule(static) num_threads(nthreads) if(count >= 262144)
        for (size_t i = start_elem; i < end_elem; i++) {
            float acc = bf16_to_fp32(((uint16_t*)in_buf[0])[i]);
            for (int r = 1; r < world_size; r++) {
                float v = bf16_to_fp32(((uint16_t*)in_buf[r])[i]);
                if (op == 0) acc += v;
                else if (op == 1) { if (v > acc) acc = v; }
                else if (op == 2) { if (v < acc) acc = v; }
                else if (op == 3) acc *= v;
            }
            dst[i] = fp32_to_bf16(acc);
        }
    }
}

typedef struct {
    ShmComm *comm;
    const void *in_dev_ptr;
    void *out_dev_ptr;
    size_t numel;
    size_t bytes;
    int dtype;
    int op;
} ShmGraphTask;

static void host_all_reduce_cb(void *arg) {
    ShmGraphTask *t = (ShmGraphTask*)arg;
    ShmComm *comm = t->comm;
    ensure_acl_ctx(comm);
    int rank = comm->rank;
    int ws = comm->world_size;
    ShmRegion *reg = comm->region;
    size_t chunk_elems = t->numel;
    size_t chunk_bytes = t->bytes;

    uint64_t step = __atomic_add_fetch(&comm->step, 1, __ATOMIC_SEQ_CST);
    if (chunk_bytes <= RING_SLOT_BYTES) {
        uint8_t (*slot_buf)[RING_SLOT_BYTES] = reg->ring_buf[step & (RING_SLOTS - 1)];
        aclrtMemcpy(slot_buf[rank], chunk_bytes, t->in_dev_ptr, chunk_bytes, ACL_MEMCPY_DEVICE_TO_HOST);
        __atomic_store_n(&reg->rank_sync[rank].arrive_seq, step, __ATOMIC_RELEASE);
        wait_arrive(reg->rank_sync, ws, step);
        reduce_ring_to_dst((uint8_t*)comm->host_bounce, slot_buf, ws, 0, chunk_elems, t->dtype, t->op);
        aclrtMemcpy(t->out_dev_ptr, chunk_bytes, comm->host_bounce, chunk_bytes, ACL_MEMCPY_HOST_TO_DEVICE);
        return;
    }

    aclrtMemcpy(reg->in_buf[rank], chunk_bytes, t->in_dev_ptr, chunk_bytes, ACL_MEMCPY_DEVICE_TO_HOST);

    __atomic_store_n(&reg->rank_sync[rank].arrive_seq, step, __ATOMIC_RELEASE);
    wait_arrive(reg->rank_sync, ws, step);

    size_t per_rank = (chunk_elems + ws - 1) / ws;
    size_t s = rank * per_rank;
    size_t e = s + per_rank;
    if (s > chunk_elems) s = chunk_elems;
    if (e > chunk_elems) e = chunk_elems;

    reduce_slice(reg->out_buf, reg->in_buf, ws, s, e, t->dtype, t->op);

    __atomic_store_n(&reg->rank_sync[rank].mid_seq, step, __ATOMIC_RELEASE);
    wait_mid(reg->rank_sync, ws, step);

    aclrtMemcpy(t->out_dev_ptr, chunk_bytes, reg->out_buf, chunk_bytes, ACL_MEMCPY_HOST_TO_DEVICE);

    __atomic_store_n(&reg->rank_sync[rank].done_seq, step, __ATOMIC_RELEASE);
    wait_done(reg->rank_sync, ws, step);
}

static void host_all_gather_cb(void *arg) {
    ShmGraphTask *t = (ShmGraphTask*)arg;
    ShmComm *comm = t->comm;
    ensure_acl_ctx(comm);
    int rank = comm->rank;
    int ws = comm->world_size;
    ShmRegion *reg = comm->region;
    size_t in_bytes = t->bytes;
    size_t out_bytes = in_bytes * (size_t)ws;

    uint64_t step = __atomic_add_fetch(&comm->step, 1, __ATOMIC_SEQ_CST);
    if (in_bytes <= RING_SLOT_BYTES) {
        uint8_t (*slot_buf)[RING_SLOT_BYTES] = reg->ring_buf[step & (RING_SLOTS - 1)];
        aclrtMemcpy(slot_buf[rank], in_bytes, t->in_dev_ptr, in_bytes, ACL_MEMCPY_DEVICE_TO_HOST);
        __atomic_store_n(&reg->rank_sync[rank].arrive_seq, step, __ATOMIC_RELEASE);
        wait_arrive(reg->rank_sync, ws, step);
        for (int r = 0; r < ws; r++) {
            aclrtMemcpy((uint8_t*)t->out_dev_ptr + (size_t)r * in_bytes, in_bytes, slot_buf[r], in_bytes, ACL_MEMCPY_HOST_TO_DEVICE);
        }
        return;
    }

    aclrtMemcpy(reg->in_buf[rank], in_bytes, t->in_dev_ptr, in_bytes, ACL_MEMCPY_DEVICE_TO_HOST);

    __atomic_store_n(&reg->rank_sync[rank].arrive_seq, step, __ATOMIC_RELEASE);
    wait_arrive(reg->rank_sync, ws, step);

    for (int r = 0; r < ws; r++) {
        memcpy((uint8_t*)comm->host_bounce + (size_t)r * in_bytes, reg->in_buf[r], in_bytes);
    }
    aclrtMemcpy(t->out_dev_ptr, out_bytes, comm->host_bounce, out_bytes, ACL_MEMCPY_HOST_TO_DEVICE);

    __atomic_store_n(&reg->rank_sync[rank].done_seq, step, __ATOMIC_RELEASE);
    wait_done(reg->rank_sync, ws, step);
}

static void host_reduce_scatter_cb(void *arg) {
    ShmGraphTask *t = (ShmGraphTask*)arg;
    ShmComm *comm = t->comm;
    ensure_acl_ctx(comm);
    int rank = comm->rank;
    int ws = comm->world_size;
    ShmRegion *reg = comm->region;
    size_t total_elems = t->numel;
    size_t total_bytes = t->bytes;
    size_t out_elems = total_elems / ws;
    size_t out_bytes = total_bytes / ws;

    uint64_t step = __atomic_add_fetch(&comm->step, 1, __ATOMIC_SEQ_CST);
    if (total_bytes <= RING_SLOT_BYTES) {
        uint8_t (*slot_buf)[RING_SLOT_BYTES] = reg->ring_buf[step & (RING_SLOTS - 1)];
        aclrtMemcpy(slot_buf[rank], total_bytes, t->in_dev_ptr, total_bytes, ACL_MEMCPY_DEVICE_TO_HOST);
        __atomic_store_n(&reg->rank_sync[rank].arrive_seq, step, __ATOMIC_RELEASE);
        wait_arrive(reg->rank_sync, ws, step);
        size_t s = rank * out_elems;
        size_t e = s + out_elems;
        reduce_ring_to_dst((uint8_t*)comm->host_bounce, slot_buf, ws, s, e, t->dtype, t->op);
        aclrtMemcpy(t->out_dev_ptr, out_bytes, comm->host_bounce, out_bytes, ACL_MEMCPY_HOST_TO_DEVICE);
        return;
    }

    aclrtMemcpy(reg->in_buf[rank], total_bytes, t->in_dev_ptr, total_bytes, ACL_MEMCPY_DEVICE_TO_HOST);

    __atomic_store_n(&reg->rank_sync[rank].arrive_seq, step, __ATOMIC_RELEASE);
    wait_arrive(reg->rank_sync, ws, step);

    size_t s = rank * out_elems;
    size_t e = s + out_elems;
    reduce_slice(reg->out_buf, reg->in_buf, ws, s, e, t->dtype, t->op);

    size_t el_sz = dtype_size(t->dtype);
    aclrtMemcpy(t->out_dev_ptr, out_bytes, reg->out_buf + s * el_sz, out_bytes, ACL_MEMCPY_HOST_TO_DEVICE);

    __atomic_store_n(&reg->rank_sync[rank].done_seq, step, __ATOMIC_RELEASE);
    wait_done(reg->rank_sync, ws, step);
}

static inline void ensure_bounce(ShmComm *comm) {
    if (comm->host_bounce == NULL) {
        if (posix_memalign(&comm->host_bounce, 128, comm->bounce_size) != 0) comm->host_bounce = NULL;
    }
}

int shm_all_reduce(ShmComm *comm, void *out_dev_ptr, const void *in_dev_ptr, size_t numel, int dtype, int op, void *stream, int is_capturing) {
    stream = flush_npu_stream();
    size_t el_sz = dtype_size(dtype);
    size_t total_bytes = numel * el_sz;
    if (is_capturing && total_bytes <= SLOT_SIZE) {
        ensure_bounce(comm);
        if (comm->host_bounce != NULL) {
             if (comm->acl_ctx == NULL) aclrtGetCurrentContext(&comm->acl_ctx);
            ShmGraphTask *t = (ShmGraphTask*)malloc(sizeof(ShmGraphTask));
            t->comm = comm;
            t->in_dev_ptr = in_dev_ptr;
            t->out_dev_ptr = out_dev_ptr;
            t->numel = numel;
            t->bytes = total_bytes;
            t->dtype = dtype;
            t->op = op;
            aclrtLaunchHostFunc(stream, host_all_reduce_cb, t);
            return 0;
        }
    }
    if (is_capturing) warn_capture_unsupported("shm_all_reduce", total_bytes);

    int rank = comm->rank;
    int ws = comm->world_size;
    ShmRegion *reg = comm->region;

    if (total_bytes <= RING_SLOT_BYTES) {
        ensure_bounce(comm);
        if (comm->host_bounce != NULL) {
            aclrtSynchronizeStream(stream);
            uint64_t step = __atomic_add_fetch(&comm->step, 1, __ATOMIC_SEQ_CST);
            uint8_t (*slot_buf)[RING_SLOT_BYTES] = reg->ring_buf[step & (RING_SLOTS - 1)];
            aclrtMemcpy(slot_buf[rank], total_bytes, in_dev_ptr, total_bytes, ACL_MEMCPY_DEVICE_TO_HOST);
            __atomic_store_n(&reg->rank_sync[rank].arrive_seq, step, __ATOMIC_RELEASE);
            wait_arrive(reg->rank_sync, ws, step);
            reduce_ring_to_dst((uint8_t*)comm->host_bounce, slot_buf, ws, 0, numel, dtype, op);
            aclrtMemcpy(out_dev_ptr, total_bytes, comm->host_bounce, total_bytes, ACL_MEMCPY_HOST_TO_DEVICE);
            return 0;
        }
    }

    size_t max_elems = SLOT_SIZE / el_sz;
    aclrtSynchronizeStream(stream);

    size_t offset_elems = 0;
    while (offset_elems < numel) {
        size_t chunk_elems = numel - offset_elems;
        if (chunk_elems > max_elems) chunk_elems = max_elems;
        size_t chunk_bytes = chunk_elems * el_sz;
        const uint8_t *cur_in = (const uint8_t*)in_dev_ptr + offset_elems * el_sz;
        uint8_t *cur_out = (uint8_t*)out_dev_ptr + offset_elems * el_sz;

        uint64_t step = __atomic_add_fetch(&comm->step, 1, __ATOMIC_SEQ_CST);
        aclrtMemcpy(reg->in_buf[rank], chunk_bytes, cur_in, chunk_bytes, ACL_MEMCPY_DEVICE_TO_HOST);

        __atomic_store_n(&reg->rank_sync[rank].arrive_seq, step, __ATOMIC_RELEASE);
        wait_arrive(reg->rank_sync, ws, step);

        size_t per_rank = (chunk_elems + ws - 1) / ws;
        size_t s = rank * per_rank;
        size_t e = s + per_rank;
        if (s > chunk_elems) s = chunk_elems;
        if (e > chunk_elems) e = chunk_elems;

        reduce_slice(reg->out_buf, reg->in_buf, ws, s, e, dtype, op);

        __atomic_store_n(&reg->rank_sync[rank].mid_seq, step, __ATOMIC_RELEASE);
        wait_mid(reg->rank_sync, ws, step);

        aclrtMemcpy(cur_out, chunk_bytes, reg->out_buf, chunk_bytes, ACL_MEMCPY_HOST_TO_DEVICE);

        __atomic_store_n(&reg->rank_sync[rank].done_seq, step, __ATOMIC_RELEASE);
        wait_done(reg->rank_sync, ws, step);

        offset_elems += chunk_elems;
    }
    return 0;
}

int shm_all_gather(ShmComm *comm, void *out_dev_ptr, const void *in_dev_ptr, size_t in_bytes, void *stream, int is_capturing) {
    stream = flush_npu_stream();
    if (is_capturing && in_bytes <= SLOT_SIZE) {
        ensure_bounce(comm);
        if (comm->host_bounce != NULL) {
            if (comm->acl_ctx == NULL) aclrtGetCurrentContext(&comm->acl_ctx);
            ShmGraphTask *t = (ShmGraphTask*)malloc(sizeof(ShmGraphTask));
            t->comm = comm;
            t->in_dev_ptr = in_dev_ptr;
            t->out_dev_ptr = out_dev_ptr;
            t->numel = 0;
            t->bytes = in_bytes;
            t->dtype = 4;
            t->op = 0;
            aclrtLaunchHostFunc(stream, host_all_gather_cb, t);
            return 0;
        }
    }
    if (is_capturing) warn_capture_unsupported("shm_all_gather", in_bytes);

    int rank = comm->rank;
    int ws = comm->world_size;
    ShmRegion *reg = comm->region;

    if (in_bytes <= RING_SLOT_BYTES) {
        ensure_bounce(comm);
        if (comm->host_bounce != NULL) {
            size_t out_bytes = in_bytes * (size_t)ws;
            aclrtSynchronizeStream(stream);
            uint64_t step = __atomic_add_fetch(&comm->step, 1, __ATOMIC_SEQ_CST);
            uint8_t (*slot_buf)[RING_SLOT_BYTES] = reg->ring_buf[step & (RING_SLOTS - 1)];
            aclrtMemcpy(slot_buf[rank], in_bytes, in_dev_ptr, in_bytes, ACL_MEMCPY_DEVICE_TO_HOST);
            __atomic_store_n(&reg->rank_sync[rank].arrive_seq, step, __ATOMIC_RELEASE);
            wait_arrive(reg->rank_sync, ws, step);
            for (int r = 0; r < ws; r++) {
                aclrtMemcpy((uint8_t*)out_dev_ptr + (size_t)r * in_bytes, in_bytes, slot_buf[r], in_bytes, ACL_MEMCPY_HOST_TO_DEVICE);
            }
            return 0;
        }
    }

    aclrtSynchronizeStream(stream);

    size_t offset = 0;
    while (offset < in_bytes) {
        size_t chunk = in_bytes - offset;
        if (chunk > SLOT_SIZE) chunk = SLOT_SIZE;

        uint64_t step = __atomic_add_fetch(&comm->step, 1, __ATOMIC_SEQ_CST);
        aclrtMemcpy(reg->in_buf[rank], chunk, (const uint8_t*)in_dev_ptr + offset, chunk, ACL_MEMCPY_DEVICE_TO_HOST);

        __atomic_store_n(&reg->rank_sync[rank].arrive_seq, step, __ATOMIC_RELEASE);
        wait_arrive(reg->rank_sync, ws, step);

        for (int r = 0; r < ws; r++) {
            uint8_t *dst = (uint8_t*)out_dev_ptr + (size_t)r * in_bytes + offset;
            aclrtMemcpy(dst, chunk, reg->in_buf[r], chunk, ACL_MEMCPY_HOST_TO_DEVICE);
        }

        __atomic_store_n(&reg->rank_sync[rank].done_seq, step, __ATOMIC_RELEASE);
        wait_done(reg->rank_sync, ws, step);

        offset += chunk;
    }
    return 0;
}

int shm_reduce_scatter(ShmComm *comm, void *out_dev_ptr, const void *in_dev_ptr, size_t total_numel, int dtype, int op, void *stream, int is_capturing) {
    stream = flush_npu_stream();
    size_t el_sz = dtype_size(dtype);
    size_t total_bytes = total_numel * el_sz;
    int ws = comm->world_size;
    int rank = comm->rank;
    size_t out_bytes = total_bytes / ws;
    size_t out_numel = total_numel / ws;

    if (is_capturing && total_bytes <= SLOT_SIZE) {
        ensure_bounce(comm);
        if (comm->host_bounce != NULL) {
            if (comm->acl_ctx == NULL) aclrtGetCurrentContext(&comm->acl_ctx);
            ShmGraphTask *t = (ShmGraphTask*)malloc(sizeof(ShmGraphTask));
            t->comm = comm;
            t->in_dev_ptr = in_dev_ptr;
            t->out_dev_ptr = out_dev_ptr;
            t->numel = total_numel;
            t->bytes = total_bytes;
            t->dtype = dtype;
            t->op = op;
            aclrtLaunchHostFunc(stream, host_reduce_scatter_cb, t);
            return 0;
        }
    }
    if (is_capturing) warn_capture_unsupported("shm_reduce_scatter", total_bytes);

    ShmRegion *reg = comm->region;

    if (total_bytes <= RING_SLOT_BYTES) {
        ensure_bounce(comm);
        if (comm->host_bounce != NULL) {
            aclrtSynchronizeStream(stream);
            uint64_t step = __atomic_add_fetch(&comm->step, 1, __ATOMIC_SEQ_CST);
            uint8_t (*slot_buf)[RING_SLOT_BYTES] = reg->ring_buf[step & (RING_SLOTS - 1)];
            aclrtMemcpy(slot_buf[rank], total_bytes, in_dev_ptr, total_bytes, ACL_MEMCPY_DEVICE_TO_HOST);
            __atomic_store_n(&reg->rank_sync[rank].arrive_seq, step, __ATOMIC_RELEASE);
            wait_arrive(reg->rank_sync, ws, step);
            size_t s = rank * out_numel;
            size_t e = s + out_numel;
            reduce_ring_to_dst((uint8_t*)comm->host_bounce, slot_buf, ws, s, e, dtype, op);
            aclrtMemcpy(out_dev_ptr, out_bytes, comm->host_bounce, out_bytes, ACL_MEMCPY_HOST_TO_DEVICE);
            return 0;
        }
    }

    aclrtSynchronizeStream(stream);

    if (total_bytes <= SLOT_SIZE) {
        uint64_t step = __atomic_add_fetch(&comm->step, 1, __ATOMIC_SEQ_CST);
        aclrtMemcpy(reg->in_buf[rank], total_bytes, in_dev_ptr, total_bytes, ACL_MEMCPY_DEVICE_TO_HOST);

        __atomic_store_n(&reg->rank_sync[rank].arrive_seq, step, __ATOMIC_RELEASE);
        wait_arrive(reg->rank_sync, ws, step);

        size_t s = rank * out_numel;
        size_t e = s + out_numel;
        reduce_slice(reg->out_buf, reg->in_buf, ws, s, e, dtype, op);

        aclrtMemcpy(out_dev_ptr, out_bytes, reg->out_buf + s * el_sz, out_bytes, ACL_MEMCPY_HOST_TO_DEVICE);

        __atomic_store_n(&reg->rank_sync[rank].done_seq, step, __ATOMIC_RELEASE);
        wait_done(reg->rank_sync, ws, step);
        return 0;
    }

    for (int target_r = 0; target_r < ws; target_r++) {
        size_t offset_elems = 0;
        size_t max_elems = SLOT_SIZE / el_sz;
        while (offset_elems < out_numel) {
            size_t chunk_elems = out_numel - offset_elems;
            if (chunk_elems > max_elems) chunk_elems = max_elems;
            size_t chunk_bytes = chunk_elems * el_sz;
            size_t global_elem_offset = (size_t)target_r * out_numel + offset_elems;

            uint64_t step = __atomic_add_fetch(&comm->step, 1, __ATOMIC_SEQ_CST);
            aclrtMemcpy(reg->in_buf[rank], chunk_bytes, (const uint8_t*)in_dev_ptr + global_elem_offset * el_sz, chunk_bytes, ACL_MEMCPY_DEVICE_TO_HOST);

            __atomic_store_n(&reg->rank_sync[rank].arrive_seq, step, __ATOMIC_RELEASE);
            wait_arrive(reg->rank_sync, ws, step);

            size_t per_rank = (chunk_elems + ws - 1) / ws;
            size_t s = rank * per_rank;
            size_t e = s + per_rank;
            if (s > chunk_elems) s = chunk_elems;
            if (e > chunk_elems) e = chunk_elems;
            reduce_slice(reg->out_buf, reg->in_buf, ws, s, e, dtype, op);

            __atomic_store_n(&reg->rank_sync[rank].mid_seq, step, __ATOMIC_RELEASE);
            wait_mid(reg->rank_sync, ws, step);

            if (rank == target_r) {
                aclrtMemcpy((uint8_t*)out_dev_ptr + offset_elems * el_sz, chunk_bytes, reg->out_buf, chunk_bytes, ACL_MEMCPY_HOST_TO_DEVICE);
            }

            __atomic_store_n(&reg->rank_sync[rank].done_seq, step, __ATOMIC_RELEASE);
            wait_done(reg->rank_sync, ws, step);

            offset_elems += chunk_elems;
        }
    }
    return 0;
}

int shm_broadcast(ShmComm *comm, void *dev_ptr, size_t total_bytes, int root, void *stream, int is_capturing) {
    stream = flush_npu_stream();
    if (is_capturing) warn_capture_unsupported("shm_broadcast", total_bytes);
    int rank = comm->rank;
    int ws = comm->world_size;
    ShmRegion *reg = comm->region;

    aclrtSynchronizeStream(stream);

    size_t offset = 0;
    while (offset < total_bytes) {
        size_t chunk = total_bytes - offset;
        if (chunk > SLOT_SIZE) chunk = SLOT_SIZE;

        uint64_t step = __atomic_add_fetch(&comm->step, 1, __ATOMIC_SEQ_CST);
        if (rank == root) {
            aclrtMemcpy(reg->out_buf, chunk, (const uint8_t*)dev_ptr + offset, chunk, ACL_MEMCPY_DEVICE_TO_HOST);
        }

        __atomic_store_n(&reg->rank_sync[rank].arrive_seq, step, __ATOMIC_RELEASE);
        wait_arrive(reg->rank_sync, ws, step);

        if (rank != root) {
            aclrtMemcpy((uint8_t*)dev_ptr + offset, chunk, reg->out_buf, chunk, ACL_MEMCPY_HOST_TO_DEVICE);
        }

        __atomic_store_n(&reg->rank_sync[rank].done_seq, step, __ATOMIC_RELEASE);
        wait_done(reg->rank_sync, ws, step);

        offset += chunk;
    }
    return 0;
}

int shm_gather(ShmComm *comm, void *out_dev_ptr, const void *in_dev_ptr, size_t in_bytes, int dst, void *stream, int is_capturing) {
    stream = flush_npu_stream();
    if (is_capturing) warn_capture_unsupported("shm_gather", in_bytes);
    int rank = comm->rank;
    int ws = comm->world_size;
    ShmRegion *reg = comm->region;

    aclrtSynchronizeStream(stream);

    size_t offset = 0;
    while (offset < in_bytes) {
        size_t chunk = in_bytes - offset;
        if (chunk > SLOT_SIZE) chunk = SLOT_SIZE;

        uint64_t step = __atomic_add_fetch(&comm->step, 1, __ATOMIC_SEQ_CST);
        aclrtMemcpy(reg->in_buf[rank], chunk, (const uint8_t*)in_dev_ptr + offset, chunk, ACL_MEMCPY_DEVICE_TO_HOST);

        __atomic_store_n(&reg->rank_sync[rank].arrive_seq, step, __ATOMIC_RELEASE);
        wait_arrive(reg->rank_sync, ws, step);

        if (rank == dst && out_dev_ptr != NULL) {
            for (int r = 0; r < ws; r++) {
                uint8_t *dst_ptr = (uint8_t*)out_dev_ptr + (size_t)r * in_bytes + offset;
                aclrtMemcpy(dst_ptr, chunk, reg->in_buf[r], chunk, ACL_MEMCPY_HOST_TO_DEVICE);
            }
        }

        __atomic_store_n(&reg->rank_sync[rank].done_seq, step, __ATOMIC_RELEASE);
        wait_done(reg->rank_sync, ws, step);

        offset += chunk;
    }
    return 0;
}

} // extern "C"
```

---

### 4.2 `/home/models/shm_hccl/npu_communicator.py`（通信器替换与 `torch.distributed` 钩子）

```python
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
```

---

### 4.3 `/home/models/shm_hccl/patch_distributed.py`（`GroupCoordinatorPatch` 补丁）

> 核心改动点：在 `GroupCoordinatorPatch._init_device_communicator` 中，将 `if self.use_device_communicator and self.world_size > 1:` 改为 `if self.world_size > 1:`，使 `_WORLD` 组在 `init_distributed_environment` 阶段即初始化并注册 `ShmComm`。

```python
    def _init_device_communicator(self) -> None:
        self.device = torch.npu.current_device()
        if self.world_size > 1:
            self.device_communicator = NPUCommunicator(
                cpu_group=self.cpu_group,
                device=self.device,
                device_group=self.device_group,
                unique_name=self.unique_name,
            )
```

---

### 4.4 `/home/models/shm_hccl/sampler_310p.py`（310P NPU 原生指数分布采样补丁）

> 挂载替换容器内 `/vllm-workspace/vllm-ascend/vllm_ascend/_310p/sample/sampler.py`，消除每次采样 `30 ms/step` 的 CPU 侧 `exponential_()` 拷贝与计算开销。

```python
import torch
import vllm.envs as envs

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.sample.sampler import (
    DEFAULT_LOGPROBS_MODE,
    AscendSampler,
    AscendTopKTopPSampler,
)

_DEFAULT_NPU_GEN_310P: torch.Generator | None = None
_NPU_GENERATOR_CACHE_310P: dict[int, tuple[torch.Generator, int]] = {}


def _get_default_npu_generator_310p(device: torch.device) -> torch.Generator:
    global _DEFAULT_NPU_GEN_310P
    if _DEFAULT_NPU_GEN_310P is None:
        _DEFAULT_NPU_GEN_310P = torch.Generator(device=device)
        _DEFAULT_NPU_GEN_310P.manual_seed(20260417)
    return _DEFAULT_NPU_GEN_310P


def _get_npu_generator_310p(i: int, generator: torch.Generator, device: torch.device) -> torch.Generator:
    cache_entry = _NPU_GENERATOR_CACHE_310P.get(i)
    if cache_entry is None or cache_entry[1] != id(generator):
        npu_generator = torch.Generator(device=device)
        try:
            npu_generator.manual_seed(generator.initial_seed())
        except Exception:
            npu_generator.manual_seed(20260417 + i)
        cache_entry = (npu_generator, id(generator))
        _NPU_GENERATOR_CACHE_310P[i] = cache_entry
    return cache_entry[0]


def fill_exponential_310p(
    q: torch.Tensor,
    generators: dict[int, torch.Generator],
    has_draft_mask: torch.Tensor | None = None,
) -> None:
    """Fill q with exponential values directly on NPU using synchronized NPU RNG."""
    default_gen = _get_default_npu_generator_310p(q.device)
    if not generators:
        q.exponential_(generator=default_gen)
        return
    if len(generators) != q.shape[0] or has_draft_mask is not None:
        q.exponential_(generator=default_gen)
    if has_draft_mask is not None:
        has_draft_mask_cpu = has_draft_mask.cpu()
        for i, generator in generators.items():
            npu_gen = _get_npu_generator_310p(i, generator, q.device)
            if bool(has_draft_mask_cpu[i]):
                q[i].exponential_(generator=npu_gen)
    else:
        for i, generator in generators.items():
            npu_gen = _get_npu_generator_310p(i, generator, q.device)
            q[i].exponential_(generator=npu_gen)


def _random_sample_310p(
    probs: torch.Tensor,
    generators: dict[int, torch.Generator],
) -> torch.Tensor:
    """310P-specific random sampling with NPU exponential generation for q."""
    q = torch.empty_like(probs)
    fill_exponential_310p(q, generators)
    return probs.div_(q).argmax(dim=-1).view(-1)


class AscendTopKTopPSampler310(AscendTopKTopPSampler):
    def forward_native(self, logits, generators, k, p):
        if envs.VLLM_BATCH_INVARIANT:
            return super().forward_native(logits, generators, k, p)
        if get_ascend_config().enable_reduce_sample:
            cand_logits, cand_idx = self.apply_top_k_top_p(logits, k, p, self.top_k)
            logits_to_return = None
            if self.logprobs_mode == "processed_logits":
                logits_to_return = cand_logits
            elif self.logprobs_mode == "processed_logprobs":
                logits_to_return = cand_logits.log_softmax(dim=-1, dtype=torch.float32)

            probs = torch.softmax(cand_logits, dim=-1)
            pos = _random_sample_310p(probs, generators)  # [B]

            next_token = cand_idx.gather(dim=1, index=pos.unsqueeze(1)).squeeze(1)  # [B]
            return next_token, logits_to_return
        else:
            logits = self.apply_top_k_top_p(logits, k, p)
            logits_to_return = None
            if self.logprobs_mode == "processed_logits":
                logits_to_return = logits
            elif self.logprobs_mode == "processed_logprobs":
                logits_to_return = logits.log_softmax(dim=-1, dtype=torch.float32)

            probs = logits.softmax(dim=-1, dtype=torch.float32)
            return _random_sample_310p(probs, generators), logits_to_return


class AscendSampler310(AscendSampler):
    def __init__(self, logprobs_mode=DEFAULT_LOGPROBS_MODE):
        super().__init__(logprobs_mode=logprobs_mode)
        self.topk_topp_sampler = AscendTopKTopPSampler310(logprobs_mode=logprobs_mode)
```

---

### 4.5 `/home/models/shm_hccl/test_vllm_comm.py`（4 卡全链路快速验收脚本）

在启动几十 GB 的大模型前，可用此脚本在 20 秒内一次性验证 4 卡 `all_reduce`、`all_gather`、`reduce_scatter`、`broadcast_tensor_dict` 以及 `NPUGraph` 图捕获回放：

```python
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

rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])
torch.npu.set_device(local_rank)

with set_current_vllm_config(VllmConfig()):
    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        distributed_init_method="env://",
        local_rank=local_rank,
        backend="hccl",
    )
    initialize_model_parallel(tensor_model_parallel_size=world_size)

    x = torch.full((4, 1280), float(rank + 1), dtype=torch.float16, device=f"npu:{local_rank}")
    y = tensor_model_parallel_all_reduce(x)
    g = tensor_model_parallel_all_gather(x, dim=-1)
    rs = tensor_model_parallel_reduce_scatter(x, dim=-1)

    if rank == 0:
        d = {"a": torch.tensor([42.0, 99.0], dtype=torch.float16, device=f"npu:{local_rank}"), "meta": "ok"}
    else:
        d = None
    d_out = get_tp_group().broadcast_tensor_dict(d, src=0)

    # 验证 NPUGraph 捕获与回放
    static_x = torch.full((2, 5120), float(rank + 1), dtype=torch.float16, device=f"npu:{local_rank}")
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
        print("vLLM TP all_reduce:", float(y[0, 0].cpu()), "expected 10.0")
        print("vLLM TP all_gather shape:", list(g.shape), "vals:", [float(g[0, i * 1280].cpu()) for i in range(world_size)])
        print("vLLM TP reduce_scatter shape:", list(rs.shape), "val:", float(rs[0, 0].cpu()), "expected 10.0")
        print("vLLM TP broadcast_tensor_dict:", d_out["a"].cpu().tolist(), d_out["meta"])
        print("vLLM TP NPUGraph replay:", float(static_y[0, 0].cpu()), "expected 100.0")
```
