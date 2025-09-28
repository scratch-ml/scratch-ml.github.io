---
title: architecture
date: 2025-09-23 01:03:26
tags:
hide: true
---


## LayerNorm 到 RMSNorm 的演进
原始版本的 LayerNorm 和 RMSNorm 的公式如下：
$$
\text{LayerNorm}(x) = \frac{x - \mu}{\sigma + \epsilon} \cdot \gamma + \beta
$$ 

RMSNorm(Root Mean Square Layer Normalization) 的公式如下：
$$
\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{N} \sum_{i=1}^{N} x_i^2 + \epsilon}} \cdot \gamma
$$
其中，$\mu$ 和 $\sigma$ 分别是输入 $x$ 的均值和标准差，$\epsilon$ 是一个很小的常数，$\gamma$ 和 $\beta$ 是可学习的参数。

原始 transformer 是基于 LayerNorm 的，但是现代的新模型均已迁移到 RMSNorm 了。一个较为科学的解释是
RMSNorm 优势的 modern explanation:
- Fewer operations:不需要计算均值
- Fewer parameters:不需要存储 bias 参数（$\beta$）

RMSNorm 很重要是因为 data movement 的占比很高。normalization 仅占大概 0.17% 的 flop（另外 99.8% 为矩阵操作 和 element-wise 操作），但是运行时长却占据了 25.5%，所以需要优化。更细致地来说：
- 矩阵乘法是 compute-bound（高算术强度），GPU 能把算力吃满；
- RMSNorm 是 memory-bound（低算术强度），大部分时间在全量读/写激活与参数。
- RMSNorm 往往需要对同一块数据做多次全局内存访问（读激活→归一化→再写回），而“每次遍历”的计算很少，kernel 启动/同步开销也更显著。
- 用 roofline model 看，就是 AI = FLOPs/Bytes 很低(Arithmetic Intensity，算术强度)，性能被内存带宽上限卡住，因而单位 FLOP 的耗时远大于 GEMM。

因此 normalization 的优化方向是“少搬、多合并”。
横轴是算术强度（FLOPs/Byte），纵轴是性能（FLOP/s）；左侧受内存带宽限制（斜线区），右侧受算力上限限制（平顶区）。
- 横轴（算术强度 AI = FLOPs/Byte）: 每搬运1字节数据，能做多少次计算。越大表示“算得多、搬得少”。
- 纵轴（性能 FLOP/s）: 实际每秒能做多少次计算。


## More generally: dropping bias

大多数现代 Transformer 模型都放弃了 bias 项
原始的 Transform：
$$
FFN(x) = \text{max}(0, xW_1 + b_1)W_2 + b_2
$$
这里实际上是 ReLU 作为激活函数后再和一个线性层计算的结果，因为 ReLU 的公式是：
$$
\text{ReLU}(x) = \text{max}(0, x)
$$

现在的 Transformer 模型的大多数实现（如果没有被 gated）：
$$
FFN(x) = \sigma(xW_1)W_2
$$
注：这里 σ 通常是 ReLU、GELU 等激活函数。
原因： memory(similar tp RMSNorm) 和 optimization stability


## Activation

ReLU, GeLU, Swish,

ReLU 的公式是：
$$
\text{FFN}(x) = \text{max}(0, xW_1)W_2
$$


GeLU 的公式是：
$$
\text{FFN}(x) = GELU(xW_1)W_2
$$
$$
GELU(x) = x \Phi(x)
$$


### Gated Activation
原始的 FF layer 如下：
$$
\text{FFN}(x) = \text{max}(0, xW_1)W_2
$$

instead of a Linear + ReLU: augment the above with an (entrywise) linear term:

Gated Activation：

$$
\text{max}(0, xW_1) \otimes (xV)
$$

This gives the gated variant (ReGLU) — note that we have an extra parameter \(V\):

$$
\text{FF}_{\text{ReGLU}}(x) = (\text{max}(0, xW_1) \otimes xV) W_2
$$


## 串行 vs 并行层
正常的 transformer block 是串行的，先计算 Attention，然后 MLP


## Position Embeddings
RoPE

## 超参数
FFN 的 hidden size 通常是 embedding size（d_model） 的 4 倍，但是也有一些模型使用 8 倍。（4 倍是经验值，8 倍是理论值）
#### Exception 1  GLU  variant

GLU 核心思想：把 MLP 第一层的输出拆成两条分支，一个是“值”分支，一个是“门”分支，做按元素相乘，让网络学会“哪些通道要放大/抑制”。本质是可学习的特征选择与条件计算。

标准 FFN（对照）
$$
\text{FFN}(x)=W_2\,\sigma(xW_1)
$$
GLU（原版，sigmoid 门）
$$
\text{GLU}(x)= (xW_g)\ \odot\ \sigma(xW_v),\quad
\text{FF}{\text{GLU}}(x)=W_2\,\text{GLU}(x)
$$
ReGLU（ReLU 门在“值”分支）
$$
\text{ReGLU}(x)= \text{ReLU}(xW_g)\ \odot\ (xW_v),\quad
\text{FF}{\text{ReGLU}}(x)=W_2\,\text{ReGLU}(x)
$$
GeGLU（GELU 门在“值”分支）
$$
text{GeGLU}(x)= \text{GELU}(xW_g)\ \odot\ (xW_v)
$$
SwiGLU（SiLU/Swish 门在“值”分支，LLM 中最常见）
$$
\text{SwiGLU}(x)= \text{SiLU}(xW_g)\ \odot\ (xW_v)
$$

为保持与 GELU-FFN 相同 FLOPs，常把门控 FFN 的中间维度设为约 2/3 原来。(例如 GELU 用 $d_{ff} = 4d_{model}$ 时，SwiGLU 用 $d_{ff} = 8d_{model}/3$)






## 附录
### 什么是"gated"？

**Gated** 指的是在神经网络中使用门控机制，比如：
- **Gated Linear Unit (GLU)**: 使用门控来控制信息流
- **SwiGLU**: 结合了 Swish 激活函数和门控机制
- **Gated Recurrent Unit (GRU)**: 在循环神经网络中的门控


在 Transformer FFN 中的区别
**没有 gated 的 FFN**（传统实现）：
```
FFN(x) = σ(xW₁)W₂
```
这里 σ 通常是 ReLU、GELU 等激活函数。
**有 gated 的 FFN**（现代实现）：
```
FFN(x) = SwiGLU(xW₁, xW₂)W₃
```
其中：
- `⊙` 表示逐元素相乘（Hadamard product）
- 门控机制通过两个分支：一个分支经过激活函数，另一个分支不经过，然后相乘

为什么使用门控？

门控机制的优势：
1. **更好的梯度流**：避免梯度消失问题
2. **更强的表达能力**：可以学习更复杂的非线性变换
3. **训练稳定性**：通常比传统激活函数更稳定

所以"如果没有被 gated"意思是：**如果这个 Transformer 模型没有使用门控机制，那么它的 FFN 层就是传统的简单形式**。
