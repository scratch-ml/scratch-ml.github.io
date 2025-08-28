---
title: notebook
date: 2025-06-05 19:47:16
tags:
hide: true
draft: true
---
## Overview
### 1. Basics
Tokenization & Tokenizer：在字符串和整数序列之间进行转换的工具，建立一个映射关系。
- BPE：Byte Pair Encoding，一种常用的分词方法，将字符串分解为子字符串，然后合并最常见的子字符串，直到达到预定的词汇表大小。

Architecture: Transformer 及其更新
- 激活函数：ReLU、SwiGLU  
- 位置编码（Positional Encoding）：RoPE、ALiBi
- 归一化（Normalization）：Layer Normalization、RMSNorm
- 归一化的位置：pre-norm、post-norm
- MLP：Dense、MoE
- 注意力机制（Attention Mechanism）：full、sliding window、linear
- Lower-dimension attention： group-query attention(GQA)、multi-head latent attention(MLA)

Training:
- Optimizer： AdamW
- Learning Rate Scheduler： Cosine、Cosine with Warmup、Linear、Linear with Warmup
- Loss Function： Cross-Entropy、Label Smoothing、KL-Divergence
- Regularization： Dropout、Weight Decay
 
### 2. System
Kernels:

Parallelism:
- Tensor Parallelism
- Data Parallelism
- Model Parallelism
- Pipeline Parallelism


Inference:
- 被 RL、evaluation 所需要
- 推理的计算量最后会超过训练
- Prefill 和 Decode
  - Prefill: Compute-bound
  - Decode: Memory-bound
- 加速
  - 使用 cheaper model（蒸馏、量化）
  - 投机解码（Greed Decoding）
  - kv cache、batching


### 3. Scaling Laws



### 4. Data

Evaluation:
Data Curation: 
Data processing


### 5. Alignment
base moodel 具有原始潜力，擅长补齐 next token。Alignment 使得模型真正有用。

Alignmen 的目标
- 让 LLM 做到指令跟随
- 指定风格
 -安全性：拒绝回答危险问题

SFT 


### 6. Summary

效率驱动设计决策
- 计算资源受限的现状
  - 当前处于计算资源受限的时代
  - 设计决策需要最大化利用现有硬件资源

- 数据处理优化
  - 避免在低质量/无关数据上浪费计算资源
  - 确保数据质量，提高计算效率

- 分词策略
  - 使用原始字节虽然优雅，但在当前模型架构下计算效率低
  - 需要采用更高效的分词方法

- 模型架构改进
  - 许多改进都以减少内存使用或计算量（FLOPs）为目标
  - 例如：共享KV缓存、滑动窗口注意力机制等

- 训练策略
  - 单轮训练（single epoch）已经足够
  - 不需要过多轮次的训练

- 扩展法则应用
  - 在较小的模型上使用较少的计算资源进行超参数调优
  - 更高效地利用计算资源

- 模型对齐
  - 如果模型能更好地针对特定用例进行调优
  - 可以使用更小的基础模型



## Tokenization

### 1. 概述

分词器(Tokenizer)是自然语言处理中的基础组件，用于在字符串和整数序列之间进行转换。其主要功能包括：

- 将文本字符串编码(encode)为整数序列(tokens)
- 将整数序列解码(decode)回原始文本
- 建立词汇表(vocabulary)，定义所有可能的token

### 2. 常见分词方法

#### 2.1 字符级分词(Character-based Tokenization)

**特点：**
- 将文本分解为单个Unicode字符
- 每个字符映射到一个整数(Unicode码点)
- 词汇表大小约为150K

**优缺点：**
- 优点：实现简单，可以处理任何文本
- 缺点：
  - 词汇表过大
  - 许多字符使用频率低，效率不高
  - 序列长度过长

#### 2.2 字节级分词(Byte-based Tokenization)

**特点：**
- 将文本转换为UTF-8字节序列
- 每个字节映射到0-255之间的整数
- 词汇表大小固定为256

**优缺点：**
- 优点：词汇表小，实现简单
- 缺点：
  - 压缩率低(compression ratio = 1)
  - 序列长度过长
  - 不适合Transformer等模型(因为注意力机制的计算复杂度与序列长度平方相关)

#### 2.3 词级分词(Word-based Tokenization)

**特点：**
- 将文本按词分割
- 使用正则表达式识别词边界
- 词汇表大小取决于训练数据中的唯一词数

**优缺点：**
- 优点：符合人类直觉，语义单位清晰
- 缺点：
  - 词汇表可能非常大
  - 罕见词处理困难
  - 需要处理未知词(UNK token)

#### 2.4 字节对编码(BPE, Byte Pair Encoding)

**特点：**
- 结合了字节级和词级分词的优点
- 通过训练自动确定词汇表
- 常用字符序列用单个token表示，罕见序列用多个token表示

**工作原理：**
1. 从字节级token开始
2. 统计相邻token对的出现频率
3. 合并最常见的token对
4. 重复步骤2-3直到达到目标词汇表大小

**优点：**
- 词汇表大小可控
- 压缩效果好
- 可以处理未知词
- 被GPT-2等主流模型采用

### 3. 实现示例

#### 3.1 基础接口

```python
class Tokenizer(ABC):
    """分词器抽象接口"""
    def encode(self, string: str) -> list[int]:
        """将字符串编码为整数序列"""
        raise NotImplementedError

    def decode(self, indices: list[int]) -> str:
        """将整数序列解码为字符串"""
        raise NotImplementedError
```

#### 3.2 BPE分词器参数

```python
@dataclass(frozen=True)
class BPETokenizerParams:
    """BPE分词器参数"""
    vocab: dict[int, bytes]     # 索引到字节的映射
    merges: dict[tuple[int, int], int]  # token对到新token的映射
```

### 4. 实际应用

#### 4.1 GPT-2分词器

- 使用BPE算法
- 采用预分词(pre-tokenization)处理
- 支持特殊token(如`<|endoftext|>`)
- 可通过tiktoken库使用

#### 4.2 性能优化方向

1. 优化merge操作，只处理相关的token对
2. 实现预分词
3. 支持特殊token
4. 提高实现效率

### 5. 总结

- 分词是NLP中的必要步骤，但可能不是最优解
- 不同分词方法各有优劣，需要根据具体应用场景选择
- BPE是目前最主流的分词方法，在效率和效果上取得了很好的平衡
- 未来可能直接使用字节级处理，但目前分词仍然是必要的


## PyTorch, resource accounting