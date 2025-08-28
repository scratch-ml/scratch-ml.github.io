---
title: vLLM 技术架构与实现深度解析
date: 2025-04-01 20:17:23
tags: [LLM, 性能优化, 推理加速]
hide: true
---

# vLLM 技术架构与实现深度解析

## 1. 简介
随着大语言模型（LLM）在各个领域的广泛应用，模型服务部署中的性能瓶颈日益凸显。主要挑战包括推理延迟高、显存利用率低、并发处理能力受限等问题。vLLM 应运而生，作为高性能 LLM 推理引擎，它为这些关键问题提供了创新的解决方案。vLLM 在推理速度提升 2-3 倍的同时，显著降低了显存占用。根据 [vLLM 论文](https://arxiv.org/abs/2309.06180) 所声明的，这种性能提升主要源于三个技术创新：
- PagedAttention 技术：把操作系统的内存分页概念应用于 LLM 推理，实现了高效的显存管理机制
- Continuous Batching 处理系统：采用动态请求聚合和智能调度策略，优化了并发请求的处理效率
- 分布式计算架构：支持灵活的模型部署方案，适应不同规模的硬件环境和计算需求

本文将深入探讨 vLLM 的技术架构、核心组件实现以及最佳实践，并将详细分析其性能优化原理，以帮助读者更好地理解 vLLM 的架构概览。

<!-- ## 2. 

- **PagedAttention**: 创新的注意力机制计算方案
- **连续批处理**: 动态请求合并与调度
- **高效内存管理**: 基于分页的 KV Cache 管理
- **分布式推理**: 支持模型并行和张量并行
- **API 兼容性**: 支持 OpenAI 风格的 API -->

## 2. 核心特性
### 2.1 PagedAttention 机制

<!-- #### 2.1.1 传统 KV Cache 的问题 -->
PagedAttention 是 vLLM 的核心创新，它通过将操作系统的内存分页思想引入到 LLM 推理过程中，改变了 KV Cache 的管理方式。

标准的 Transformer 解码过程中，模型需要为每个正在生成的 Sequence 分配一块**连续的显存空间**，用于存储该 Sequence 历史所有 token 的 Key 和 Value（即 KV Cache）。随着推理过程中 Sequence 长度的不断增长，KV Cache 也会线性扩展。如果有多个用户并发请求，每个请求的 Sequence 长度和生命周期都不同，显存中会出现大量"空洞"——即部分已分配但暂未使用或已释放的空间无法被及时复用，造成显存碎片化。下面我将举例说明碎片的产生及其种类：

- **内部碎片**：在LLM 服务系统中，为了满足未来可能增长的需求，通常为请求预留一个大块连续显存（例如最大 2048 tokens）。但实际上请求可能只用了一部分，例如只用了 300 tokens，这就导致剩下的那部分未被使用的显存浪费了，这种已被分配但未被使用的显存称为内部碎片。

- **外部碎片**：外部碎片是指未被分配出去的小块显存空间零散地分布在整体显存中，无法有效利用，即使总空闲显存量看起来很多，但因为不连续，无法满足新请求对一大块连续内存的需求。例如，请求 1 释放了 500 个 token 的空间，请求 2 释放了 300 个 tokens 的空间，但来了一个需要 1000 个 token 空间的新请求时，尽管总有 800 空闲，但由于它们不连续，仍然无法满足新请求。
<center class ='img'>
<figure>
    <img title="碎片化显存示意图" src="frag.png" width=500 height=300>
    <figcaption class="image-caption">不同LLM serving system 中碎片化显存示意图</figcaption>
  </figure>
</center>

为了解决上述显存碎片化严重的问题，提高显存利用率或者说相同显存下支持更多的并发请求，vLLM 引入了 PagedAttention 机制。 PagedAttention 通过引入操作系统的分页机制，将 KV Cache 划分为固定大小的块（页），每个块可独立分配和释放，并通过 Block Table 记录每个 Sequence 的块分配，实现逻辑连续但物理离散的高效显存管理。同时，系统维护 Sequence 到 blocks 的映射关系，利用高效索引结构支持数据的快速定位和动态扩展，从而显著提升了多用户并发场景下的显存利用率和访问效率。另外，在这里需要澄清一个概念，**PagedAttention 并不是一个全新的注意力机制，而是对传统 KV Cache 管理方式的改进**。

<center class ='img'>
<figure>
    <img title="PagedAttention 显存管理示意图" src="blocktable.png" width=650 height=300>
    <figcaption class="image-caption">PagedAttention 显存管理示意图</figcaption>
  </figure>
</center>


此外，vLLM 还支持 KV Cache 共享（KV Cache Sharing），即在多样本生成、beam search 等场景下，多个输出序列可以共享公共部分的 KV Cache，从而显著节省内存，提高并发处理能力。以 beam search 场景为例，给定输入 prompt，模型会首先生成第一个 token 的概率分布。选出概率最高的前 k 个 token（k 即 beam size），每个 token 作为一个“束”（beam）的起点，形成 k 个候选序列，对于每个候选序列，模型再次预测下一个 token 的概率分布,同样选出累计概率最高的前 k 个。重复上述扩展和筛选过程，直到所有 beam 都生成了终止符（如 <eos>），或达到最大长度，最终可以输出 top-k 个序列，供下游任务选择。而这些候选序列的 KV Cache 可以共享公共部分，从而显著节省内存，提高并发处理能力。



通过这种创新的内存管理方式，PagedAttention 不仅解决了传统 KV Cache 管理的痛点，还带来了显著的性能提升。它的成功实现使得 vLLM 能够在有限的硬件资源下支持更多的并发请求，成为大规模 LLM 服务部署的重要技术基础。

### 2.2 调度系统

vLLM 的调度系统负责管理和优化推理请求：

- **请求队列管理**: 动态优先级调度
- **批处理优化**: 自适应批大小调整
- **资源分配**: GPU 内存和计算资源的动态分配

### 2.3 分布式架构

支持多种并行策略：

- **模型并行**: 处理大型模型的跨设备部署
- **张量并行**: 提高计算效率
- **流水线并行**: 优化延迟和吞吐量

## 4. 使用方式

### 4.1 Python API

```python
from vllm import LLM, SamplingParams

# 初始化模型
llm = LLM(model="llama-2-7b")

# 设置生成参数
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=100
)

# 执行推理
outputs = llm.generate("Tell me a story", sampling_params)
```

### 4.2 REST API

vLLM 提供了兼容 OpenAI API 的服务接口：

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-2-7b",
    "prompt": "Tell me a story",
    "max_tokens": 100
  }'
```

## 5. 核心模块功能

### 5.1 Worker 模块

- 负责实际的模型推理执行
- 管理计算资源和内存分配
- 实现批处理逻辑

### 5.2 Scheduler 模块

- 请求调度和优先级管理
- 动态批处理策略
- 负载均衡

### 5.3 Cache Manager 模块

- KV Cache 的分配和回收
- 内存碎片整理
- 缓存命中优化

## 6. 性能优势

与传统推理框架相比，vLLM 具有显著优势：

- **更高吞吐量**: 通过批处理优化提升 2-3 倍
- **更低延迟**: PagedAttention 减少 40% 响应时间
- **更好的内存利用**: 提升 50% 以上的内存效率

## 7. 最佳实践

### 7.1 部署建议

- 根据负载选择合适的并行策略
- 优化批处理参数
- 监控内存使用情况

### 7.2 性能调优

- 调整缓存大小
- 优化请求队列配置
- 合理设置批处理阈值

## 8. 总结

vLLM 通过创新的 PagedAttention 机制和高效的调度系统，显著提升了 LLM 推理性能。其核心优势在于：

1. 高效的内存管理
2. 灵活的批处理策略
3. 优秀的系统扩展性
4. 便捷的接口支持

这些特性使其成为大规模 LLM 部署的理想选择。随着持续优化和社区贡献，vLLM 的性能和功能还将进一步提升。

## 参考资料

1. vLLM 官方文档
2. PagedAttention 论文
3. vLLM GitHub 仓库