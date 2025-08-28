---
title: 深度学习基础
date: 2025-04-15 00:41:01
tags:
draft: true
hide: true
---
## Attention
注意力机制是一种让模型能够关注输入序列中最重要部分的机制。就像我们看一张图时，会重点关注图中的某些突出部分，而不是全部，注意力机制就是这样把有限的注意力集中在重点信息上。注意力机制主要由三个关键组件构成：
1. Query（查询）：表示当前需要关注的内容
2. Key（键）：用于与Query进行匹配，计算相关性
3. Value（值）：包含实际的信息内容

在注意力机制（尤其是 Transformer 的自注意力机制）中，key、value、query 的维度如下：
- 假设输入特征维度为 $d_{model}$，则：
    - Query（Q）维度：$d_{model} \times d_k$
    - Key（K）维度：$d_{model} \times d_k$
    - Value（V）维度：$d_{model} \times d_v$
- 一般 $d_k = d_v = d_{model} / \text{head数}$（多头注意力时）


> 注意：
> - 输入特征维度（$d_{model}$）：这是每个词被表示成向量后的维度，例如768维的向量，意味着每个词可以用768个数字来表示。这些数字共同编码了词的语义、语法等各种特征。
> - 词表大小（Vocabulary Size）：这是模型能够识别的不同词的数量，例如BERT的词表大小是30,000个词，GPT-3的词表大小是50,000个词。
> - 举个例子：假设我们有一个词"猫"，在词表中，它可能被分配一个编号，比如1234，但这个编号会被转换成一个768维的向量，比如[0.1, -0.3, 0.5, ...]（共768个数字），这个向量包含了"猫"这个词的各种语义信息。


输入 $X$ 通过三个不同的线性变换分别得到 Q、K、V：
$$
Q = XW^Q, \quad K = XW^K, \quad V = XW^V
$$

- $W^Q$ 形状为 $d_{model} \times d_k$
- $W^K$ 形状为 $d_{model} \times d_k$
- $W^V$ 形状为 $d_{model} \times d_v$
###   $d_{model}$
- $d_{model}$ 是输入特征的维度，指输入特征的每个 token 的特征维度。
- 在 Transformer 中，$d_{model}$ 通常设置为 768。在 LLM 中，$d_{model}$ 通常设置为 4096，例如llama3.1-70b, llama3.1-405b, d_{model}=4096。


### $d_k$（Key/Query 的维度）
- $d_k$ 是 Key（K）和 Query（Q）向量的维度。
- 在计算注意力分数（即 Q 和 K 的点积）时，Q 和 K 必须有相同的维度 $d_k$。
- $d_k$ 的大小影响注意力分数的分布，通常设置为 $d_{model}/\text{head数}$（多头注意力时，每个头分配一部分维度）。

### $d_v$（Value 的维度）
- $d_v$ 是 Value（V）向量的维度。
- 在加权求和得到注意力输出时，输出的每个向量的维度就是 $d_v$。
- $d_v$ 通常也设置为 $d_{model}/\text{head数}$，但理论上可以和 $d_k$ 不同。

**总结：**
- $d_k$：决定了 Q、K 的维度，影响注意力分数的计算。
- $d_v$：决定了 V 的维度，影响最终注意力输出的特征维度。

在大语言模型（LLM，如 GPT、BERT 等）中，输入 $X$ 的维度通常为 $(\text{batch\_size}, \text{seq\_len}, d_{model})$，其中：
- $\text{batch\_size}$：批大小，一次输入的样本数量
- $\text{seq\_len}$：每个样本的序列长度（如 token 数）
- $d_{model}$：每个 token 的特征维度（如 embedding size）

例如，若 $\text{batch\_size}=2$，$\text{seq\_len}=5$，$d_{model}=768$，则 $X$ 的形状为 $(2, 5, 768)$。


### 注意力机制的计算过程

注意力机制的核心计算过程可以分为以下几个步骤：

1. **计算注意力分数（Attention Scores）**
   - 通过 Query 和 Key 的点积计算相关性分数：
   $$
   \text{Attention Scores} = QK^T
   $$
   - 得到的分数矩阵维度为 $(\text{batch\_size}, \text{seq\_len}, \text{seq\_len})$
   - 每个 batch 中的分数矩阵维度为 $(\text{seq\_len}, \text{seq\_len})$
   - 每个分数表示 Query 和 Key 之间的相关性

2. **缩放（Scaling）**
   - 为了防止点积结果过大导致 softmax 梯度消失，需要对分数进行缩放：
   $$
   \text{Scaled Scores} = \frac{QK^T}{\sqrt{d_k}}
   $$
   - 除以 $\sqrt{d_k}$ 是为了使方差保持在合理范围内

3. **Softmax 归一化**
   - 对缩放后的分数应用 softmax 函数，将分数转换为概率分布：
   $$
   \text{Attention Weights} = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})
   $$
   - 确保所有权重和为 1，且都是非负数

4. **加权求和**
   - 使用注意力权重对 Value 进行加权求和：
   $$
   \text{Output} = \text{Attention Weights} \cdot V
   $$
   - 最终输出维度为 $(\text{seq\_len}, d_v)$

### 多头注意力机制（Multi-Head Attention）

多头注意力机制允许模型同时关注不同位置的不同特征：

1. **并行计算**
   - 将输入分成多个头（heads）
   - 每个头独立计算注意力
   - 最后将所有头的结果拼接

2. **计算过程**
   - 假设有 $h$ 个头，每个头的维度为 $d_k = d_v = d_{model}/h$
   - 对每个头 $i$：
     $$
     \text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
     $$
   - 将所有头的结果拼接：
     $$
     \text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O
     $$
   - 其中 $W^O$ 是输出投影矩阵

### 注意力机制的优势

1. **并行计算**
   - 不同于 RNN 需要顺序计算，注意力机制可以并行处理所有位置
   - 大大提高了计算效率

2. **长距离依赖**
   - 可以直接建立任意两个位置之间的关联
   - 解决了 RNN 难以处理长距离依赖的问题

3. **可解释性**
   - 注意力权重可以直观地显示模型关注的重点
   - 有助于理解模型的决策过程

4. **灵活性**
   - 可以处理变长序列
   - 支持不同类型的输入（文本、图像等）

### 实际应用示例

以机器翻译为例，当翻译"我喜欢猫"到英文时：

1. Query 代表当前要翻译的词
2. Key 和 Value 代表源语言中的每个词
3. 注意力机制会计算当前词与源语言中每个词的相关性
4. 根据相关性权重，从源语言中提取最相关的信息
5. 最终生成对应的英文翻译

这种机制使得模型能够：
- 准确捕捉词与词之间的对应关系
- 处理词序不同的语言对
- 保持翻译的准确性和流畅性

## Transformer
Transformer 是一种基于注意力机制的神经网络架构，它通过自注意力机制来处理序列数据。







