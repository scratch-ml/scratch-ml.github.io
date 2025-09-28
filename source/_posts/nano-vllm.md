---
title: nano-vllm
date: 2025-09-18 11:51:13
tags:
hide: true
---
# 把 vLLM 压缩到 1,200 行：nano-vLLM 的技术解读

## 项目简介

nano-vLLM 是一个轻量级的高吞吐推理引擎，旨在为低资源环境提供高性能的推理服务。它基于 vLLM 的架构，但进行了大幅简化，仅保留了核心功能，并针对低资源环境进行了优化。

本文会在通读 nano-vLLM 源码的基础上，梳理其工程骨架，并详细阐述其关键优化，包括但不限于：Paged Attention、Prefix Caching、Tensor Parallelism、Torch Compile、CUDA Graph、FlashAttention 等。nano-vLLM 虽然项目体量很小，仅有约 1,200 行 Python 代码，但是却覆盖了一个现代高吞吐推理引擎的核心要素。如果希望在较短时间内理解一个推理引擎的关键路径，或把这些优化点移植到自己的系统，可以参考 nano-vLLM 项目以及本文的解读。

本文将按结构→细节→实践建议的顺序展开。

## 项目一图流（结构与职责）

当一个请求发送给 nano-vLLM 后，推理流程如下：提示词（prompts）被分词成序列（Sequences），由 LLMEngine 管理并加入等待队列；调度器（scheduler）优先处理 prefill 阶段的 Sequence，打包批次受 max_num_batched_tokens 和可用 KV blocks 约束； model runner 加载权重、分配 KV 缓存（block_manager.py），准备上下文（如 block_table、slot_mapping、cu_seqlens）；执行模型 forward 动作（qwen3.py 调用 layers/ 的 attention、linear 等），利用 FlashAttention 计算 logits；采样器（sampler.py）应用温度缩放生成新 token 并追加到序列（sequence.py）；重复 decode 步骤直到遇到 EOS 或达到 max_tokens，最终 postprocess 回收块并返回输出文本。

上述全部功能所对应的所有文件协作实现高效推理：
- engine/ 目录编排整体流程
- layers/ 提供优化算子
- models/ 定义模型架构
- utils/ 辅助加载和上下文

更细致的功能结构图如下：

```
/data/nano-vllm
├── nanovllm/
│   ├── __init__.py                # 对外导出 `LLM`, `SamplingParams`
│   ├── llm.py                     # 薄封装，继承 `LLMEngine`
│   ├── config.py                  # 运行配置（模型路径、并行规模、KV 块大小等）
│   ├── sampling_params.py         # 采样参数（temperature / max_tokens / ignore_eos）
│   ├── engine/
│   │   ├── llm_engine.py         # 引擎入口：管理请求队列、调度与生成主循环
│   │   ├── model_runner.py       # 多进程/多卡、权重加载、KVCache 分配、CUDA Graph 捕获
│   │   ├── scheduler.py          # 动态批调度（prefill 优先、decode 抢占）
│   │   ├── sequence.py           # 序列状态与 block table（块视图）
│   │   └── block_manager.py      # KV 块分配/回收 + 前缀哈希缓存（xxhash）
│   ├── layers/
│   │   ├── attention.py          # Flash-Attn + Triton 写 KVCache（prefill/decode 二合一）
│   │   ├── rotary_embedding.py   # RoPE（预计算 cos/sin 缓存）
│   │   ├── linear.py             # 张量并行线性层（列并行/行并行/QKV 合并）
│   │   ├── layernorm.py          # RMSNorm（含 residual 融合分支）
│   │   ├── embed_head.py         # 词嵌入/LM Head 的词表并行
│   │   └── sampler.py            # 温度缩放 + Gumbel Top-1 近似采样
│   ├── models/
│   │   └── qwen3.py              # Qwen3 解码器与 CausalLM 封装
│   └── utils/
│       ├── context.py            # 推理上下文（prefill/decode、块表、cu_seqlens 等）
│       └── loader.py             # safetensors 权重装载（含打包权重映射）
├── example.py                    # 快速上手样例（聊天模板/批量）
├── bench.py                      # 吞吐 benchmark 脚本
├── README.md                     
└── pyproject.toml                # 依赖（torch/triton/transformers/flash-attn/xxhash）
```
nano-vLLM 对外 API 极简：
- `LLM`: 推理入口，负责模型/分布式初始化、请求接入与调度、生成主循环与输出解码
- `SamplingParams`: 采样与停止配置，描述“怎么生成”，如 `temperature`定义采样强度，`max_tokens`/`ignore_eos`定义停止条件

推理动作由 `LLM.generate` 触发，背后是调度器打包请求 → 模型前向（prefill 或 decode） → 采样（按 `temperature`） → 写回新 token；遇到 `eos`（且未忽略）或达到 `max_tokens` 时标记完成并回收 KV 块。

## 核心技术：把“该有的能力”做全

以下按模块与执行阶段，结合关键张量形状与内存公式做细化说明，便于对照源码迁移到其它推理框架。

### 1) Paged Attention（分页注意力）
> 文件：`layers/attention.py`，`engine/model_runner.py`，`engine/block_manager.py`

Paged Attention 将每条序列的 KVCache 按固定大小划分为“块”（blocks ≈ OS 的页），用“块表”将序列的逻辑连续块映射到物理非连续显存块。注意力计算时通过块表高效收集对应 K/V，从而只在末尾块产生少量浪费（<~4%），避免大规模过度预留与碎片；且可以支持按需分配与复用物理块，命中前缀直接共享（引用计数 + 写时复制）。
- **关键数据结构**
  - **物理块（Physical Block）**：容量固定，存放多 token 的 K/V。
  - **逻辑块（Logical Block）**：序列视角上的连续区间。
  - **块表（Block Table）**：逻辑块 → 物理块映射；允许不同序列映射到同一物理块以实现前缀共享。
  - **引用计数与 COW**：共享块在写入时触发写时复制，保证并发安全。
  - **空闲列表/分配器**：按需分配与回收，降低显存碎片。

- 关键数据形状：
  - `block_table: [B, max_num_blocks] (int32)`：每条序列的页表，-1 填充；由 `prepare_block_tables` 生成（`engine/model_runner.py`）。
  - `slot_mapping: [N] (int32)`：本步需写入的全局槽位（prefill 为所有新增 token；decode 为每序列 1 个）；槽位计算为 `slot = block_id * 256 + offset`。
  - `k_cache/v_cache`（单层视图）：`[num_blocks, 256, H_kv, D]`，源自一次性分配的 `[2, L, num_kvcache_blocks, 256, H_kv, D]`


- 与其他模块的关系：
  - 页表来源：`engine/block_manager.py` 负责按 `block_size=256` 维护 `block_table`；
  - 全局 KVCache 的预算与一次性大块分配见“2) KV Cache 预算与布局”；
  - 新增 token 的 KV 写入由“4) Triton 写 KV”完成；
  - 注意力计算用“5) FlashAttention”的变长/paged 接口。


- 执行路径：
  - Prefill
    - 命中前缀时：`cu_seqlens_k.sum() > cu_seqlens_q.sum()`，构造 `block_table` 并将 `k/v` 直接指向全局 KVCache，仅对新增 token 追加写入。
    - 调用：`flash_attn_varlen_func(q, k_or_kcache, v_or_vcache, cu_seqlens_q/k, max_seqlen_q/k, block_table=..., causal=True)`
  - Decode
    - 每序列仅对最后一个 token 做注意力；上下文长度 `context_lens` 与 `block_table` 指明可见范围。
    - 调用：`flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache, cache_seqlens=context_lens, block_table=..., causal=True)`
  - KV 写入
    - 使用 Triton kernel 依据 `slot_mapping` 将 `[num_kv_heads*head_dim]` 的 K/V 向量写到全局 KVCache 对应槽位。
- 边界与注意：
  - `block_table` 需对齐每条序列当前实际页数，右侧用 -1 填充；
  - `slot_mapping` 顺序必须与本步 token 的生成顺序一致；
  - 不同序列共享相同 `block_id` 页时，读路径通过 `block_table` 自然实现复用；写路径只写新增部分，避免覆盖共享历史。

### 2) KV Cache 预算与布局

> 文件：`engine/model_runner.py`

我们首先要理解一些理论计算的公式：
- 单 token KV 所占用字节数目为：`2 * L * H_kv * D * dtype_size`；
- 每个 block 存储 256 token：`block_bytes = 2 * L * 256 * H_kv * D * dtype_size`。
- 上述公式中各个参数的意义如下：
  - L: 层数
  - H_kv: KV 头数量（通常是 num_key_value_heads）
  - D: 每个头的维度（head_dim）
  - dtype_size: 数据类型的字节数（如 fp16/bf16 = 2 字节，fp32 = 4 字节）

nano-vLLM 基于上述算式，结合 `mem_get_info()` 的 free/total 与 `memory_stats()` 的 peak/current，按 `gpu_memory_utilization` 求可分配块数 `num_kvcache_blocks`。
- 实际分配：

```
[2, L, num_kvcache_blocks, 256, H_kv, D] @ torch_dtype, cuda
```

核心分配逻辑是把每层注意力的 `k_cache/v_cache` 绑定为上述大张量的视图以供写入。每一层注意力里的 k_cache/v_cache 不再单独分配，而是“指向”这个大张量各自的切片（view），写入时直接写到共享缓冲区，避免碎片、提升效率。

小例子:假设 L=24, H_kv=8, D=128, dtype=fp16(2B)：
- 单 token ≈ 2×24×8×128×2 = 98,304 B ≈ 96 KiB
- 每块 256 token ≈ 24 MiB
- 若可分 200 块，KV Cache ≈ 4.8 GiB

这样做的好处：
- 一次性大块分配，降低碎片
- 视图切片，便于分层/分块管理
- 易于根据显存水位动态控制块数

因此，整体流程未变，但内存布局与管理方式有优化。
- 先用公式估算每 token /每 block 的字节
- 用显存信息+利用率算能分配多少块
- 实际用一个形状为 [2, L, num_kvcache_blocks, 256, H_kv, D] 的大张量承载全部 KV，并让各层拿视图来写

这是主流推理引擎（按块分页 KV Cache）的典型做法，便于扩展到分页、回收与调度。

### 3) 前缀缓存（Prefix Caching）

> 文件：`engine/block_manager.py`，`engine/model_runner.py`

- 分块与哈希：序列按 `block_size=256` 切分，对块序列做滚动哈希（`xxhash.xxh64`）。第 i 块哈希的输入包含“第 i 块的 token bytes + 第 i-1 块的哈希”，从而 O(1) 判断“前 i 块是否一致”。
- 分配（allocate）：命中哈希且内容一致的块直接增加 `ref_count` 并计入 `num_cached_tokens`；miss 则从 `free_block_ids` 取新块，记录到 `seq.block_table`。
- 追加（may_append）：长度从 256k 增至 256k+1 先预分配新块；当变为整除 256 时，对“最后块”计算哈希并登记。
- 与注意力联动：prefill 阶段向 kernel 传递 `block_tables: [B, max_num_blocks] (int32)`，命中前缀时直接将 `k/v` 指向 KVCache，避免重复计算历史 K/V。

更具体一些：

- 关键数据结构：
  - `Block`：`{block_id, ref_count, hash, token_ids}`；空闲块在 `free_block_ids`，使用中在 `used_block_ids`。
  - `hash_to_block_id: Dict[int, int]`：滚动哈希到块 id 的映射（命中需二次比对 `token_ids` 防碰撞）。
  - `Sequence.block_table: List[int]`：一条序列在 KVCache 中的块 id 列表；`num_cached_tokens` 记录命中的前缀 token 数。

- allocate 伪代码（摘意）：
```python
h = -1
for i in range(seq.num_blocks):
    token_ids = seq.block(i)
    h = compute_hash(token_ids, h) if len(token_ids) == block_size else -1
    block_id = hash_to_block_id.get(h, -1)
    cache_hit = (block_id != -1 and blocks[block_id].token_ids == token_ids)
    if not cache_hit:
        block_id = free_block_ids[0]; block = _allocate_block(block_id)
    else:
        seq.num_cached_tokens += block_size
        block = blocks[block_id]
        block.ref_count = block.ref_count + 1 if block_id in used_block_ids else (_allocate_block(block_id)).ref_count
    if h != -1:
        block.update(h, token_ids); hash_to_block_id[h] = block_id
    seq.block_table.append(block_id)
```

- may_append 逻辑：
  - 若 `len(seq) % block_size == 1`，说明新 token 落在“新块”的第一个位置，需要预分配一个空块（`hash == -1`）。
  - 若 `len(seq) % block_size == 0`，说明当前最后块被填满，计算其哈希并登记到 `hash_to_block_id`。

- 与 FlashAttention 的交互：
  - Prefill 场景：如果 `cu_seqlens_k.sum() > cu_seqlens_q.sum()`（说明命中历史前缀），`block_tables` 会被拼成 `[B, max_num_blocks]` 传入 kernel，K/V 直接引用 KVCache。
  - Decode 场景：通过 `slot_mapping: [B] (int32)` 指明“这个 step 写入/读取的 KV 槽位”，形如 `slot = block_table[last] * 256 + (last_block_num_tokens-1)`。

- 形状对照：
  - `block_tables`: `[B, max_num_blocks] (int32)`
  - `slot_mapping`: `[N] (int32)`，N 为本 step 写入的 token 数（prefill 为所有新增 token，decode 为 B）
  - `k_cache/v_cache`: `[num_layers, num_blocks, 256, num_kv_heads, head_dim]` 的视图（底层来自 `[2, L, num_kvcache_blocks, 256, H_kv, D]`）

- 一个小例子（B=2，block_size=4，演示方便）：
  - seq A：tokens `a0 a1 a2 a3 | a4 a5 ...` → `block_table: [3, 7, ...]`
  - seq B：tokens `a0 a1 a2 a3 | b4 b5 ...`，前 4 个与 A 相同 → 第一块哈希命中，直接共享 `block_id=3`，`ref_count += 1`；第二块不同，分配新块。

- 边界与注意事项：
  - 哈希碰撞：通过“哈希命中后再比对 `token_ids`”规避错误共享；
  - 内存回收：`deallocate(seq)` 需要从后往前遍历 block table，逐块 `ref_count -= 1`，为 0 才回到 `free_block_ids`；
  - 写入顺序：Triton kernel 逐 token 写入 KVCache，所以 `slot_mapping` 必须与本步的 token 顺序一致；
  - block_size 选择：实现中要求 `kvcache_block_size % 256 == 0`，默认 256；块太小哈希/元数据开销偏大，块太大命中粒度变粗，二者需权衡。

### 4) Triton 写 KV：轻量、刚好够用

> 文件：`layers/attention.py`

- kernel 网格：`(N,)`，N 为本次需写入 token 数；
- 输入约束：`key.stride(-1)==1`、`key.stride(1)==head_dim` 保证内存布局有利于矢量化；
- 逻辑：用 `slot_mapping[idx]` 计算 KV 全局偏移，将 `[num_heads*head_dim]` 的 key/value 一次性写入 `k_cache/v_cache` 对应槽位。

### 5) FlashAttention：变长 prefill + 增量 decode

> 文件：`layers/attention.py`

- Prefill：`flash_attn_varlen_func(q, k, v, cu_seqlens_q/k, max_seqlen_q/k, block_table, causal=True)`；命中前缀时 `k/v` 直接引用 KVCache；
- Decode：`flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache, cache_seqlens=context_lens, block_table, causal=True)`；仅对最后 token 做注意力；
- 缩放：`scale = 1/sqrt(D)`，与常规实现一致。

Q、K、V 的形状（单步视角）：
- `q`: `[-1, num_heads, head_dim]`
- `k/v`: `[-1, num_kv_heads, head_dim]`

### 6) 张量并行（Tensor Parallelism）

> 文件：`layers/linear.py`，`layers/embed_head.py`

- 列并行：切输出维，局部 `F.linear`，无需通信；
- 行并行：切输入维，前向后 `all_reduce` 聚合（bias 只在 rank0 加）；
- QKV 合并：计算 Q/K/V 子区间偏移，加载权重时按 `loaded_shard_id in {q,k,v}` 精确落位；
- Gate/Up 合并：一张大矩阵，运行时 `chunk` 成两支；
- 词表并行：Embedding/LMHead 切 vocab 维；Embedding 前向按 mask 只算本 rank 负责区间；LMHead 在 rank0 `gather` 拼回 logits。

权重装载（`utils/loader.py`）依赖各参数注册的 `weight_loader`，以 safetensors 的 key 为导向把 shard 切到位。

### 7) CUDA Graph：把 decode 静态化

> 文件：`engine/model_runner.py`

- 预捕获一组 batch size（`[1,2,4,8,16..,512]`）的图；
- 捕获前准备静态形状的张量缓冲（`input_ids/positions/slot_mapping/context_lens/block_tables/outputs`），进入 `with torch.cuda.graph(...)` 执行一次前向；
- 运行期将真实输入拷到缓冲里，调用 `graph.replay()`。

触发条件：prefill / `enforce_eager=True` / `bs>512` 时走 eager；其余 decode 优先复用图。

### 8) 调度器：prefill 优先，decode 可抢占

> 文件：`engine/scheduler.py`

- Prefill：受 `max_num_batched_tokens`、`max_num_seqs` 与可用 KV 块约束，尽可能装满一个 step；命中前缀会减少实际计算量；
- Decode：若无法“追加一块”，会从运行队列尾部 `preempt` 一条，回到等待队列；
- Postprocess：写回新 token，遇 `eos`（且未忽略）或达上限 `max_tokens` 则完成并回收块。

### 9) 采样器（Sampler）

> 文件：`layers/sampler.py`

- 近似 Gumbel-Top1：对 `logits/temperature` 取 softmax 得到 `p`，用 `Exp(1)` 噪声做 `p / exp(e)` 再 `argmax`，等价于 `argmax(log p - e)` 的技巧，计算量小、实现简洁。

### 10) RoPE 与 RMSNorm 的“小心思”

- RoPE：预计算 `[max_pos, 1, D]` 的 cos/sin 缓存，按位置索引后做旋转；
- RMSNorm：提供 residual 融合路径 `add_rms_forward(x, residual)`，减少一次读写；二者均加 `@torch.compile` 便于算子融合。

### 11) 多进程与轻量 RPC

- `LLMEngine` 启动 `tensor_parallel_size-1` 个子进程；
- `ModelRunner` 初始化 NCCL、绑定 CUDA 设备；
- rank0 用共享内存+事件广播方法调用（控制面），各 rank 收到后执行同名方法；数据面仍走分布式张量与通信原语。

## 端到端执行路径（概览）

1) 初始化：`LLMEngine.__init__` 产出 `Config`、启动子进程、在 rank0 构建 `ModelRunner`、用 `AutoTokenizer` 设置 `eos`；
2) 预热与图捕获：`warmup_model()`、`allocate_kv_cache()`、`capture_cudagraph()`（条件触发）；
3) 主循环：`schedule()` 决定 prefill/decode 与本步序列；`prepare_*` 组装输入并 `set_context(...)`；`run_model()` 前向产出 logits；rank0 `Sampler` 采样；`postprocess()` 追加 token 与收尾回收。

返回为 `[{"text": str, "token_ids": List[int]}]` 的列表。


## 快速上手
```python
from nanovllm import LLM, SamplingParams

llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
outputs = llm.generate(["Hello, Nano-vLLM."], SamplingParams(temperature=0.6, max_tokens=256))

print(outputs[0]["text"])
```

运行基准：

```bash
python bench.py
```

## 实践建议

- 调试优先：先开 `enforce_eager=True`，再逐步启用图捕获；
- Prefill 吞吐：适当增大 `max_num_batched_tokens`（显存允许时）；
- 并行规模：TP>1 时注意 NCCL 与 rank0 聚合（LM Head logits 聚合在 rank0）；
- 负载特征：多轮/长前缀复用场景，前缀缓存收益最明显。

## 结语

nano-vLLM 以极小工程量呈现了现代推理引擎的关键路径，适合作为“源码导读”与“工程移植参考”。建议先读 `engine/model_runner.py` 与 `layers/attention.py`，一个负责“把图跑起来”，一个负责“把算子跑快”。希望这份解读能帮助读者更快地建立整体心智模型。
