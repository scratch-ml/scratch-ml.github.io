---
title: torch
date: 2025-09-19 02:24:50
tags:
hide: true
---
## 讲座概览（PyTorch 与资源核算）
- 目标：自底向上梳理从张量到模型、优化器再到训练循环的全部原语，并始终进行资源核算。
- 两类关键资源：
  - Memory（GB）：参数、梯度、优化器状态、激活。
  - Compute（FLOPs）：前向、反向主导于大矩阵乘。
- 不展开 Transformer 细节，聚焦通用原语与心智模型。


## Tensors Basics
###  张量与内存（dtype 与动态范围）

- 创建：`tensor/zeros/ones/randn/empty`；`empty` 仅分配不初始化，便于自定义初始化（如 `nn.init.trunc_normal_`）。
- dtype 与内存：float32(4B)、float16(2B)、bfloat16(2B)、fp8(1B)。
  - float16 小数动态范围差，易下溢：如 `torch.tensor([1e-8], dtype=torch.float16) == 0`。
  - bfloat16 动态范围≈float32，数值更稳；分辨率差些但对 DL 影响小。
  - FP8（H100 支持 E4M3/E5M2）需配合库（如 Transformer Engine）谨慎使用。
- 张量内存：`x.numel() * x.element_size()`；大型线性层权重可达 GB 量级。

### Tensors on GPUs
- 设备：`device = 'cuda' if torch.cuda.is_available() else 'cpu'`；`x.to(device, non_blocking=True)`。
- 直接在 GPU 上创建：`torch.zeros(..., device='cuda:0')`。
- 固定内存 + 异步拷贝：`DataLoader(pin_memory=True)` 配合 `.to(..., non_blocking=True)`。
- 监测：`torch.cuda.memory_allocated()`、`max_memory_allocated()`；必要时 `torch.cuda.synchronize()` 计时。

### Storage/stride
PyTorch 的 Tensor 是指向存储的指针 + 访问元数据
什么是 stride？`stride(dim)` 决定相邻步长
- 核心概念：在 PyTorch（和 NumPy）里，stride 表示在内存中沿着某一维度前进一步，需要跨过多少个“元素”的距离。不是字节数，是元素步长。
- 若元素大小为 element_size 字节，那么字节步长为 stride[i] * element_size。

和 shape 的关系（默认连续 C-order）：
对一个形状为 (d0, d1, ..., dn-1) 的张量，如果它是连续的（contiguous），则有：
- stride[n-1] = 1
- stride[n-2] = d[n-1]
- stride[n-3] = d[n-2] * d[n-1]
- 以此类推（前一维的步长是后面所有维度大小的乘积）

### tensor slicing
许多操作仅仅是对 tensor 创造了不同的`view`，而不是创建新的 tensor。因为不涉及到任何 copy，所以一个 tensor 的变化可能会影响其他 tensor。

```python
import torch

def same_storage(x: torch.Tensor, y: torch.Tensor):
    return x.untyped_storage().data_ptr() == y.untyped_storage().data_ptr()

x = torch.tensor([[1, 2, 3],[4, 5, 6]], dtype=torch.float32)

# 切片得到 view（共享底层存储）
y = x[0]
assert torch.equal(y, torch.tensor([1, 2, 3], dtype=torch.float32))
assert same_storage(x, y)

# 通过 view 改变形状（仍为视图，满足连续性）
y = x.view(3, 2)
assert torch.equal(y, torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float32))
assert same_storage(x, y)

# 修改源张量会反映到视图上（共享存储）
x[0][0] = 100
assert y[0][0] == 100
```


- 视图与拷贝：切片/转置多为视图，共享底层存储；`view` 需 contiguous；用 `.contiguous()` 先整理。
- 常用：逐元素（`pow/sqrt/rsqrt/+/*`）、上三角 `triu()`（如构造自回归 mask）、矩阵乘 `@` 支持批维广播。



- 形状可读性：
  - `einops.einsum/reduce/rearrange` 用命名维度减少 `-1/-2` 混乱。
  - `jaxtyping` 仅做文档标注：`x: Float[torch.Tensor, "batch seq hidden"]`。

### 命名维度：einops 与 jaxtyping 实战

**为什么需要命名维度？** 传统按位置（如 `-1/-2`）索引易出错且难读。`einops` 通过“爱因斯坦求和记号”的语法为维度命名，使重排、归约与广义矩阵乘法显式、可读；`jaxtyping` 则把形状写进类型注解，作为“形状文档”。

#### 动机：别再数 -1/-2

```python
import torch

# 传统写法：用位置维度（容易出错）
x = torch.ones(2, 2, 3)  # batch=2, seq=2, hidden=3
y = torch.ones(2, 2, 3)  # batch=2, seq=2, hidden=3
z = x @ y.transpose(-2, -1)  # (batch, seq, seq)：这里的 -2 是 seq，-1 是 hidden

assert z.shape == (2, 2, 2)
```

```python
from einops import einsum

# 命名维度：显式写出含义，避免数索引
z2 = einsum(x, y, "batch seq hidden, batch seq hidden -> batch seq seq")
assert torch.equal(z, z2)
```

#### 形状标注：jaxtyping 基础

```python
from jaxtyping import Float

# 仅作“文档化”的形状注释（不强制检查，但极大提升可读性）
x_typed: Float[torch.Tensor, "batch seq heads hidden"] = torch.ones(2, 2, 1, 3)
assert x_typed.shape == (2, 2, 1, 3)
```

#### einsum：广义矩阵乘与良好记账

```python
from einops import einsum
from jaxtyping import Float

# 两个序列长度不同的张量，沿 hidden 聚合，得到 (batch, seq1, seq2)
x: Float[torch.Tensor, "batch seq1 hidden"] = torch.ones(2, 3, 4)
y: Float[torch.Tensor, "batch seq2 hidden"] = torch.ones(2, 5, 4)

# 旧写法（需要转置并数轴）
z_old = x @ y.transpose(-2, -1)  # (2, 3, 5)

# 新写法（命名维度，未出现在右侧的维度会被求和消去）
z_new = einsum(x, y, "batch seq1 hidden, batch seq2 hidden -> batch seq1 seq2")
assert torch.equal(z_old, z_new)

# 使用省略号 ... 表示对任意前缀批维进行广播
z_broadcast = einsum(x, y, "... seq1 hidden, ... seq2 hidden -> ... seq1 seq2")
assert torch.equal(z_new, z_broadcast)
```

规则小结：
- 未在输出式中出现的命名维度会被自动求和（如上 `hidden`）。
- `...` 代表任意数量的批维，可简洁表达广播。

#### reduce：显式归约

```python
from einops import reduce
from jaxtyping import Float

xr: Float[torch.Tensor, "batch seq hidden"] = torch.ones(2, 3, 4)

# 旧写法：按位置指定维度
y_old = xr.mean(dim=-1)  # (2, 3)

# 新写法：沿命名维度归约；"... hidden -> ..." 表示把 hidden 维聚合掉
y_new = reduce(xr, "... hidden -> ...", "mean")  # 支持 "sum"/"max"/"min" 等
assert torch.equal(y_old, y_new)
```

#### rearrange：拆分/合并维度以便变换

```python
from einops import rearrange, einsum
from jaxtyping import Float

# 假设 total_hidden = heads * hidden1
xh: Float[torch.Tensor, "batch seq total_hidden"] = torch.ones(2, 3, 8)
w: Float[torch.Tensor, "hidden1 hidden2"] = torch.ones(4, 4)

# 1) 将扁平维度拆成命名子维（heads 与 hidden1）
xh = rearrange(xh, "... (heads hidden1) -> ... heads hidden1", heads=2)  # (2, 3, 2, 4)

# 2) 仅对 hidden1 维做线性变换（其余维度保持广播）
xh = einsum(xh, w, "... hidden1, hidden1 hidden2 -> ... hidden2")  # (2, 3, 2, 4)

# 3) 将 heads 与 hidden2 合并回单一维度，恢复扁平表示
xh = rearrange(xh, "... heads hidden2 -> ... (heads hidden2)")  # (2, 3, 8)
```

以上模式把“按位置的轴操作”转化为“按语义的维度操作”。配合 `jaxtyping` 的形状注释，代码含义与数据流形状一一对应，显著降低了出错与维护成本。

## FLOPs 与 MFU（Model FLOPs Utilization）

- 概念：
  - FLOPs：完成的浮点运算总数；FLOP/s（或 FLOPS）：每秒能力。
  - 大模型训练由矩阵乘主导：`2*m*n*p`。
  - 近似：前向 2·B·#Params，反向 4·B·#Params，总计 6·B·#Params。
- 硬件峰值（示意）：A100 ≈ 312 TFLOP/s（bf16/fp16），H100 ≈ 1979/2 TFLOP/s（dense bf16/fp16）。
- MFU 定义：实际 FLOP/s ÷ 承诺（峰值）FLOP/s；矩阵乘占比高且 dtype 充分利用（bf16 比 float32 高）时 MFU 才高。

## Autograd 与梯度 FLOPs

- 基本用法：`requires_grad=True`；前向得 `loss` 后 `loss.backward()` 生成 `*.grad`。
- 示例：线性 `pred = x @ w`，`loss = 0.5*(pred-5)^2`，得到 `w.grad = x`（如 `[1,2,3]`）。
- 计算量：
  - 前向：见上；
  - 反向：以两层线性为例，对每个参数张量的梯度与中间激活梯度都包含一次乘加，量级与前向相当或两倍；
  - 总结：前 2、反 4、合计 6（相对 #参数 与 #样本）。

## 模型与初始化

- 参数类型：`nn.Parameter`，仍是 `Tensor`；可通过 `state_dict()`/`parameters()` 访问。
- 稳定初始化：`w ~ N(0, 1/√fan_in)`，或 Xavier/He 变体；可用截断正态避免离群值：`nn.init.trunc_normal_`。
- 小线性堆叠模型（Cruncher）可用于端到端演示；记得 `model.to(device)`。

## 数据加载与批采样

- LM 数据常为 int32 序列；建议以 numpy `.npy` 或 `memmap` 存放，避免一次性加载 TB 级数据。
- 采样批：随机起点切片拼装 `[B, L]`；GPU 异步：`pin_memory()` + `.to(device, non_blocking=True)`。

## 优化器与内存/计算

- 关系图谱：
  - Momentum = SGD + 指数滑动平均(grad)
  - AdaGrad = SGD + 累积 grad^2
  - RMSProp = AdaGrad + 指数平均 grad^2
  - Adam = RMSProp + Momentum
- 计数（示例，float32）：
  - 参数数 `#P = Σ layer_params`；激活数 `#A ≈ B×D×层数`；梯度与优化器状态各≈ `#P`。
  - 总显存 ≈ 4 × (#P + #A + #P + #P) bytes（不含碎片）。
  - 单步计算 ≈ 6 × B × #P FLOPs。

## 训练循环（最小可用）

1) 获取 batch；2) 前向 `loss = F.mse_loss(...)`；3) 反向 `loss.backward()`；4) `optimizer.step()`；5) `optimizer.zero_grad(set_to_none=True)`。

```python
# 只示意关键步骤
model.train()
for step, (xb, yb) in enumerate(loader):
    xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
    pred = model(xb)
    loss = F.mse_loss(pred, yb)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
```

## 可重复性与确定性

- 设种子：`torch.manual_seed(s)` / `np.random.seed(s)` / `random.seed(s)`。
- 需要严格可复现可用：`torch.use_deterministic_algorithms(True)`（可能降吞吐）。

## Checkpointing

- 定期保存/恢复：`torch.save({'model': model.state_dict(), 'optimizer': optim.state_dict()}, 'ckpt.pt')`；`torch.load` 加载。

## 混合精度与 FP8

- 取舍：精度↑ → 稳定/准确↑ 但 显存/算力↑；精度↓ 反之。
- 策略：
  - 前向激活用 bf16/fp16；参数与部分归一化/累加保持 float32 或 bf16 视稳定性。
  - PyTorch AMP：`with torch.autocast('cuda', dtype=torch.bfloat16): ...`；fp16 需 `GradScaler`。
  - FP8（H100）：可用 NVIDIA Transformer Engine 在线性/注意力模块应用，需端到端数值验证。

## 参考与延伸

- CS336 Spring 2025 课程资料与资源核算思路（PyTorch 实战导向）。


## 资源核算（Memory & Compute）

- 公式心智模型（线性/Transformer 一阶近似）：
  - 前向 FLOPs ≈ 2 × (#数据点/序列标记) × (#参数)
  - 反向 FLOPs ≈ 4 × (#数据点/序列标记) × (#参数)
  - 单步总 FLOPs ≈ 6 × (#数据点/序列标记) × (#参数)
- AdamW（朴素，FP32）参数侧显存：
  - 参数 4B + 梯度 4B + 一阶矩 4B + 二阶矩 4B ≈ 16B/参数。
  - 常见混合精度做法：参数/梯度用 bf16(2B+2B)，但保留 FP32 master copy(4B)；显存不降太多但更快（见 Zero / Rajbhandari 2019）。
- 激活显存：≈ batch_size × seq_len × hidden_dim × bytes × 常数系数（与层数、残差、注意力缓存相关）；可用激活检查点（重计算）换算力降显存。

### 粗略估算示例

- 训练 70B 参数模型、15T token、1024×H100、MFU≈0.5：
  - total_flops = 6 × 70e9 × 15e12；
  - 每日 FLOPs = (H100 峰值 FLOP/s × MFU × 1024) × 86400；
  - 天数 = total_flops / flops_per_day。
- 8×H100 可训练的最大参数量（AdamW 朴素内存）：
  - 每卡 80GB，总 640GB；按 16B/参数，num_parameters ≈ 640e9 / 16 ≈ 40e9（未计激活与碎片）。
  - 备注：若用 bf16+FP32 master，不明显省显存；激活依赖 batch/seq，需单独核算。
