---
title: attention
date: 2025-09-22 14:26:27
tags:
hide: true
---

> 面向具有基础机器学习知识的读者，这篇文章用一文读懂三种常见的注意力（Attention）设计：Multi-Head Attention（MHA）、Multi-Query / Grouped-Query Attention（MQA/GQA）与 Multi-Head Latent Attention（MLA）。我们将从概念、实现细节到工程实践逐步展开，并给出可运行的 PyTorch 参考实现片段与选型建议。

小预告：三者的差异，核心都在“Key/Value 的形状与缓存（KV Cache）如何设计”。这会直接影响显存占用、带宽与解码吞吐。

---

### MHA

MHA（Multi-Head Attention）是 Transformer 的标准注意力。它把隐藏维度 `D_model` 切成 `num_heads` 个头，每个头的维度为 `head_dim = D_model / num_heads`，分别计算自注意力，再拼回输出。

- 基本公式（单层、忽略 bias）：
  - 线性变换：`Q = X · W_Q`, `K = X · W_K`, `V = X · W_V`
  - 形状：`X: [B, T, D_model]`，`Q/K/V: [B, T, num_heads, head_dim]`
  - 打分与归一化：`Attn = softmax((Q · K^T) / sqrt(head_dim))`
  - 聚合：`Y_head = Attn · V`，拼接各头后：`Y = Concat(Y_head) · W_O`

#### 基本公式与符号说明

公式（自注意力，单层、忽略偏置项）：

- 生成投影：`Q = X · W_Q`，`K = X · W_K`，`V = X · W_V`
- 注意力权重：`A = softmax((Q · K^T) / sqrt(D_h) + mask)`
- 聚合输出：`Y_head = A · V`，`Y = Concat(Y_head) · W_O`

符号与含义：

- `X`：输入序列表示，形状 `[B, T, D_model]`
- `B / T / D_model`：分别是 batch 大小 / 序列长度 / 模型隐藏维
- `num_heads = H`：头数；`head_dim = D_h = D_model / H`
- `W_Q, W_K, W_V`：投影矩阵，形状分别为 `[D_model, H·D_h]`
- `W_O`：输出映射矩阵，形状 `[H·D_h, D_model]`
- `Q, K, V`：投影后的查询/键/值，重排后形状为 `[B, H, T, D_h]`
- `K^T`：在最后两维上转置，形状 `[B, H, D_h, T]`
- `A`（或 `Attn`）：注意力权重，`A = softmax((Q · K^T)/sqrt(D_h) + mask)`，形状 `[B, H, T, T]`
- `mask`：掩码项；自回归解码使用因果掩码以禁止看见未来位置
- `sqrt(D_h)`：缩放因子，稳定梯度与数值范围
- `Y_head = A · V`：聚合后的每个头的输出，形状 `[B, H, T, D_h]`
- `Concat(Y_head)`：沿头维拼接回 `[B, T, H·D_h]`

直观理解：`Q` 表示“当前位在找谁”，`K` 表示“我是谁（可被匹配的特征）”，`V` 是“被带走的内容”。`A` 是对序列位置的分布，表示关注强度；`Y` 是对 `V` 的加权求和后，再经 `W_O` 融合各头的信息。

- 复杂度与内存：
  - 计算复杂度（自回归、单步解码忽略前向）：主要在打分与乘积上，和 `num_heads` 成正比。
  - KV Cache（推理时缓存 K/V 以便后续解码复用）：单 token 占用近似为 `2 * L * H * D_h * dtype_size` 字节（2 表示 K 与 V，`L` 为层数，`H` 为注意力头数，`D_h` 为每头维度）。

- PyTorch 参考实现（解码掩码的标准 MHA）：
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

B, T, D_model = 2, 16, 1024
num_heads = 16
head_dim = D_model // num_heads

x = torch.randn(B, T, D_model)
W_qkv = nn.Linear(D_model, 3 * D_model, bias=False)
W_o = nn.Linear(D_model, D_model, bias=False)

qkv = W_qkv(x)  # [B, T, 3*D]
q, k, v = qkv.split(D_model, dim=-1)

def split_heads(t):
    return t.view(B, T, num_heads, head_dim).transpose(1, 2)  # [B, H, T, Dh]

q = split_heads(q)
k = split_heads(k)
v = split_heads(v)

scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)  # [B, H, T, T]
causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
scores = scores.masked_fill(causal_mask, float('-inf'))
attn = F.softmax(scores, dim=-1)
y = torch.matmul(attn, v)  # [B, H, T, Dh]
y = y.transpose(1, 2).contiguous().view(B, T, D_model)
out = W_o(y)
```

- 什么时候用 MHA：
  - 训练与开源大模型默认配置；
  - 推理对延迟容忍度较高、显存充足；
  - 不追求极致的 KV Cache 压缩。

---

### MQA/GQA

MQA（Multi-Query Attention）与 GQA（Grouped-Query Attention）的核心思想：减少 Key/Value 的“头数”，让多组 Query 共享更少组的 K/V，从而显著降低 KV Cache 占用与带宽，提升解码吞吐。

- 概念对比：
  - MQA：`num_kv_heads = 1`，所有 Query 头共享同一组 K/V。
  - GQA：`num_kv_heads = H_kv < num_heads`，每 `group_size = num_heads / H_kv` 个 Query 头共享一组 K/V。

- 形状与映射：
  - `Q: [B, H, T, Dh]`
  - `K/V: [B, H_kv, T, Dh]`，按组广播到 `H` 个头。

- KV Cache 占用（每 token）：`2 * L * H_kv * Dh * dtype_size`。相对 MHA 的缩放比约为 `H_kv / H`。

-#### 基本公式与符号说明

以 GQA 为例（MQA 是 `H_kv=1` 的特例）：

- 生成投影：`Q = X · W_Q`，`K = X · W_K`，`V = X · W_V`
- 头与组：`H` 为 Query 头数，`H_kv` 为 KV 头数，`group_size = H / H_kv`
- 重排形状：`Q: [B, H, T, D_h]`；`K/V: [B, H_kv, T, D_h]`
- 广播映射：将每一组 `K/V` 复制到其对应的 `group_size` 个 Query 头，得到 `K', V'` 形状 `[B, H, T, D_h]`
- 注意力：`A = softmax((Q · K'^T) / sqrt(D_h) + mask)`，`A: [B, H, T, T]`
- 聚合：`Y_head = A · V'`，拼接：`Y = Concat(Y_head) · W_O`

符号与含义：
- `H`/`H_kv`/`group_size`：Query 头数 / KV 头数 / 每组包含的 Query 头数
- `D_h`：每个 Query 头维度；通常与 K/V 的维度一致
- `W_Q/W_K/W_V/W_O`：线性投影与输出矩阵
- `mask` 与 `sqrt(D_h)`：与 MHA 相同，分别做因果遮盖与缩放稳定化
- `KV Cache`：仅缓存 `H_kv` 组的 `K/V`，存储开销与带宽按比例下降

- PyTorch 参考实现（以 GQA 为例，MQA 是令 `H_kv=1` 的特例）：
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

B, T, D_model = 2, 16, 1024
num_heads = 16
num_kv_heads = 4  # GQA: 4 组；若设为 1 即 MQA
head_dim = D_model // num_heads

x = torch.randn(B, T, D_model)
W_q = nn.Linear(D_model, num_heads * head_dim, bias=False)
W_k = nn.Linear(D_model, num_kv_heads * head_dim, bias=False)
W_v = nn.Linear(D_model, num_kv_heads * head_dim, bias=False)
W_o = nn.Linear(D_model, D_model, bias=False)

q = W_q(x).view(B, T, num_heads, head_dim).transpose(1, 2)      # [B, H,  T, Dh]
k = W_k(x).view(B, T, num_kv_heads, head_dim).transpose(1, 2)   # [B, Hkv,T, Dh]
v = W_v(x).view(B, T, num_kv_heads, head_dim).transpose(1, 2)   # [B, Hkv,T, Dh]

expand = num_heads // num_kv_heads
k = k.unsqueeze(2).expand(B, num_kv_heads, expand, T, head_dim).reshape(B, num_heads, T, head_dim)
v = v.unsqueeze(2).expand(B, num_kv_heads, expand, T, head_dim).reshape(B, num_heads, T, head_dim)

scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)  # [B, H, T, T]
causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
scores = scores.masked_fill(causal_mask, float('-inf'))
attn = F.softmax(scores, dim=-1)
y = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, D_model)
out = W_o(y)
```

- 工程实践要点：
  - 解码更快：减少了从 KV Cache 读取的通道与显存带宽占用；
  - 兼容性：很多开源权重已经内置 GQA/MQA；推理框架（如 FlashAttention、vLLM 等）天然支持；
  - 质量影响：通常 MQA 最省但可能有更大质量回撤；GQA 是较稳妥折衷（例如 `H=16, H_kv=4`）。

- 小算例（fp16，`L=32, H=16, Dh=64`）：
  - MHA：`2*32*16*64*2 = 131072 B ≈ 128 KiB / token`
  - GQA（`H_kv=4`）：约 `32 KiB / token`（节省 4×）
  - MQA（`H_kv=1`）：约 `8 KiB / token`（节省 16×）

---

### MLA

MLA（Multi-Head Latent Attention）进一步在“维度”上压缩 K/V：把 K/V 投影到更低维的“latent”空间并缓存，仅在计算或输出阶段再通过小型“expander”还原或映射到输出空间。直观理解：在 GQA 减少“头数”的基础上，MLA 再减少“每头维度”。

- 典型设计（简化描述）：
  - 先常规得到 `Q`；
  - 用较少的 latent 头与更小的维度得到 `K_lat, V_lat` 并写入 KV Cache；
  - 将 `Q` 线性投影到 latent 维度，与 `K_lat` 做注意力；
  - 用一个轻量的 `value_expander` 把 `V_lat` 的聚合结果还原回 `head_dim`；
  - 最终与各头拼接并线性映射输出。

- 形状示意：
  - `Q: [B, H, T, Dh]`
  - `K_lat/V_lat: [B, H_lat, T, D_lat]`，`H_lat <= H` 且 `D_lat << Dh`
  - `Q_lat = Q · W_q2lat`，`Q_lat: [B, H, T, D_lat]`

-#### 基本公式与符号说明

- 生成投影：`Q = X · W_Q`；`K_lat = X · W_{K,lat}`；`V_lat = X · W_{V,lat}`
- 维度压缩：`H_lat <= H`，`D_lat << D_h`
- Query 降维：`Q_lat = Q · W_{q→lat}`，形状 `[B, H, T, D_lat]`
- 广播映射：将 `K_lat/V_lat` 按组广播到 `H` 个 Query 头，得到 `K_b/V_b: [B, H, T, D_lat]`
- 注意力：`A = softmax((Q_lat · K_b^T)/sqrt(D_lat) + mask)`，`A: [B, H, T, T]`
- 值还原：`Y_lat = A · V_b`，`Y_head = W_{val→out}(Y_lat)` 还原至 `D_h`
- 输出映射：`Y = Concat(Y_head) W_O`

符号与含义：
- `H/H_lat`：Query 头数 / latent KV 头数
- `D_h/D_lat`：每头维度 / latent 维度
- `W_{K,lat}/W_{V,lat}`：将输入映射到 latent KV 空间的权重
- `W_{q→lat}`：将每个 Query 头从 `D_h` 映射到 `D_lat`
- `W_{val→out}`：将 latent 聚合结果还原到 `D_h`
- `sqrt(D_lat)`：缩放因子在 latent 维度上计算
- `KV Cache`：仅缓存 `K_lat/V_lat`，显存与带宽按 `(H_lat/H)·(D_lat/D_h)` 比例下降

- PyTorch 参考实现（演示用，忽略分组卷积等更高效写法）：
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

B, T, D_model = 2, 16, 1024
H, H_lat = 16, 4
Dh, D_lat = 64, 16

x = torch.randn(B, T, D_model)
W_q = nn.Linear(D_model, H * Dh, bias=False)
W_k_lat = nn.Linear(D_model, H_lat * D_lat, bias=False)
W_v_lat = nn.Linear(D_model, H_lat * D_lat, bias=False)
W_q2lat = nn.Linear(Dh, D_lat, bias=False)      # 将每个 head 的 Q 映射到 latent 维
W_val_expand = nn.Linear(D_lat, Dh, bias=False) # 将 latent 聚合结果还原到 Dh
W_o = nn.Linear(D_model, D_model, bias=False)

q = W_q(x).view(B, T, H, Dh).transpose(1, 2)                # [B, H,    T, Dh]
k_lat = W_k_lat(x).view(B, T, H_lat, D_lat).transpose(1, 2) # [B, Hlat, T, Dlat]
v_lat = W_v_lat(x).view(B, T, H_lat, D_lat).transpose(1, 2) # [B, Hlat, T, Dlat]

# 将每个 Q 头映射到 latent 维
q_lat = W_q2lat(q)  # [B, H, T, Dlat]

# 按组把 K_lat/V_lat 广播到 H 个头（H 是 H_lat 的倍数）
expand = H // H_lat
kb = k_lat.unsqueeze(2).expand(B, H_lat, expand, T, D_lat).reshape(B, H, T, D_lat)
vb = v_lat.unsqueeze(2).expand(B, H_lat, expand, T, D_lat).reshape(B, H, T, D_lat)

scores = torch.matmul(q_lat, kb.transpose(-2, -1)) / (D_lat ** 0.5)  # [B, H, T, T]
causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
scores = scores.masked_fill(causal_mask, float('-inf'))
attn = F.softmax(scores, dim=-1)

y_lat = torch.matmul(attn, vb)  # [B, H, T, Dlat]
y = W_val_expand(y_lat).transpose(1, 2).contiguous().view(B, T, H * Dh)
out = W_o(y)
```

- 为什么有效：
  - KV Cache 写入与读取都发生在更小的 `H_lat × D_lat` 空间，显存与带宽显著下降；
  - 相比 MQA/GQA，除了“少头”，还“少维”，进一步压缩；
  - 额外代价是多了两层小线性（`W_q2lat` 和 `W_val_expand`）。

- KV Cache 占用（每 token）：`2 * L * H_lat * D_lat * dtype_size`。相对 MHA 的缩放比约为 `(H_lat/H) * (D_lat/Dh)`。

- 小算例（fp16，`L=32, H=16, Dh=64, H_lat=4, D_lat=16`）：`2*32*4*16*2 = 8192 B ≈ 8 KiB / token`，较 MHA 节省 16×，与 MQA 在此参数下相当，但带来更灵活的折衷空间。

- 工程注意：
  - 训练/权重兼容：MLA 通常需要在训练时就引入相应投影与损失，纯推理时很难对“已训练的 MHA/GQA 权重”直接替换；
  - 推理实现：需要推理引擎在 KV 写入/读取路径上支持 latent 形状与广播；
  - 质量与速度：一般会在几乎不损失质量前提下进一步降显存；具体取决于 `H_lat, D_lat` 的配置与训练策略。

---

### 工程实现：FlashAttention 与 Paged Attention 的关系

这一节不引入新的注意力“机制”，而是解释两种几乎成为标配的工程实现如何与上文的 MHA/MQA/GQA/MLA 协同：

- FlashAttention（FA）：一种 IO-aware 的精确 Softmax Attention 实现。通过块化（tiling）与在 SRAM/寄存器中累积 `softmax` 的分子/分母，减少 HBM 访问与中间张量的写回，从而显著降低显存占用与提高吞吐。它不改变数学公式，仅改变计算顺序与内存访问路径。
  - 输入/输出：与标准注意力一致，可直接用于 MHA/GQA/MQA/MLA 的打分与聚合阶段（需相应 kernel 支持 Q/K/V 的形状变体，如 `num_kv_heads ≤ num_heads`、`D_lat` 等）。
  - Prefill 与 Decode：FA 通常提供变长序列的 prefill 接口（使用 `cu_seqlens`/`block_table`）与带 KV Cache 的 decode 接口（`with_kvcache`）。
  - 实战建议：优先启用 FA；若有批量与形状限制，可回退到 eager 实现或按批分组。

- Paged Attention（PA）：面向推理的 KV Cache 管理策略。将每条序列的 K/V 缓存按固定大小分页（如 256 tokens 一页），用“块表（block table）”将逻辑连续映射到物理非连续显存页，结合“引用计数 + 写时复制”，实现前缀共享与低碎片化。
  - 角色定位：PA 主要解决“存什么、存在哪里、如何高效复用”的问题；
  - 与 FA 的关系：FA 负责“如何高效算”，PA 负责“如何高效存与取”。两者互补，经常一起使用：prefill 时通过 `block_table` 直接引用已存在的历史 K/V；decode 时按 `slot_mapping` 精确写入新 token 的 K/V 槽位，FA 的内核读取这些分页后的 K/V 完成注意力计算。

小结：
- 若把注意力看作“数学公式 × 工程实现 × 运行时存储”，则 MHA/MQA/GQA/MLA 属于“公式/架构”，FA 属于“高效算子的实现”，PA 属于“运行时的存储与复用”。在现代推理系统中，FA+PA 几乎是默认组合。

实践清单：
- 选择算子：优先使用 FlashAttention v2/v3 提供的变长/with-kvcache 接口。
- 设计缓存：采用 Paged Attention 的分页 KV 设计（页大小常取 256），并实现前缀哈希/共享。
- 形状对齐：
  - GQA/MQA：确保 `num_kv_heads` 与组广播在 kernel 支持集合内；
- 调度配合：prefill 优先、decode 稳态复用 CUDA Graph；block table 与 slot mapping 在两个阶段的构造方式不同，需要与内核接口对齐。

参考文档：
- FlashAttention（论文）：[FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- FlashAttention-2（论文）：[FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)
- FlashAttention（实现仓库）：[HazyResearch/flash-attention](https://github.com/HazyResearch/flash-attention)
- PagedAttention（论文）：[Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
- vLLM（实现仓库，含 PagedAttention）：[vllm-project/vllm](https://github.com/vllm-project/vllm)

### 对比、选型与实践建议

- 指标对比（越小越省）：

| 方案 | KV Cache（/token） | 带宽（decode） | 延迟 | 兼容性 |
| --- | --- | --- | --- | --- |
| MHA | `~2 * L * H * Dh * dtype` | 高 | 中 | 最强（默认） |
| GQA | `~2 * L * H_kv * Dh * dtype` | 中 | 低 | 强（广泛支持） |
| MQA | `~2 * L * 1 * Dh * dtype` | 低 | 低 | 强（很多模型） |
| MLA | `~2 * L * H_lat * D_lat * dtype` | 最低 | 低 | 需模型原生支持 |

- 选型建议：
  - 训练阶段：若追求推理效率，建议直接采用 GQA 或 MLA 的体系结构进行训练；
  - 只做推理：优先选用已发布的 GQA/MQA 权重；若模型原生支持 MLA，优先开启；
  - 长上下文与低显存：GQA 是“稳妥省”，MLA 是“更省但需原生支持”，MQA 是“最省但可能牺牲更多质量”。

- 实战清单：
  - 计算你的 KV 预算：`bytes_per_token = 2 * L * H_kv * D_eff * dtype_size`（MHA/GQA 的 `D_eff=Dh`，MLA 的 `D_eff=D_lat`）；
  - 结合推理引擎（如 vLLM/nano-vLLM/FlashAttention）确认是否支持 GQA/MQA/MLA 的张量形状；
  - Prefill/Decode 分离优化：Prefill 受算力约束，Decode 受带宽与 KV 形状影响更大；
  - 建议配合 Paged Attention 与 Prefix Caching 一起使用，以进一步压缩显存与提升吞吐。

---

### 常见问题（FAQ）

- MQA 和 GQA 会不会影响模型质量？
  - 相比 MHA，理论上表达力减少。实践中 GQA（如 4 或 8 组）通常影响较小，MQA 可能更明显，需以评测为准。

- MLA 与 GQA 的关系？
  - MLA 可视作在“GQA 少头”的基础上再“降维”，两者可以叠加；若模型原生采用 MLA，则 KV Cache 维度与头数均更小。

- 这些优化需要改训练吗？
  - GQA/MQA 可直接在架构中更改并从头训练，或使用已有权重；MLA 通常需要在训练期就参与，否则难以在推理期无痛替换。

---

参考阅读（按关键词检索）：
- Scaled Dot-Product Attention, Multi-Head Attention（Transformer 原论文）
- Multi-Query Attention（MQA）与 Grouped-Query Attention（GQA）
- FlashAttention 与 Paged Attention（高效注意力实现）
- Multi-Head Latent Attention（MLA）与 KV Cache 压缩相关工作
 - FlashAttention（实现与论文）：见上文“工程实现”章节参考链接
 - PagedAttention（论文与实现）：见上文“工程实现”章节参考链接