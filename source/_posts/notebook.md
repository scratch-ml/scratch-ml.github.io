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


