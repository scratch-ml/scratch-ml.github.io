---
title: gpus
date: 2025-08-28 16:33:34
hide: true
tags: 硬件
---
## Part 1: GPUs in depth — how they work and important parts
在大模型时代，算力的增长（compute scaling）带来了模型性能的巨大提升。高效的 Deep Learning 算法固然很重要，但是更快的硬件、更高的硬件利用率、以及不断改进的并行化策略，都可以显著提升模型性能，而且往往起到决定性因素。因此，深入理解硬件，并结合算法与并行化策略，是提升模型性能的关键。

首先，我们来想一下如何获取这种 compute scaling。根据摩尔定律，单位面积上的晶体管数量


Grid 是CUDA编程模型中​​最高级别的线程组织单位​​。你可以把它理解为一次内核（Kernel）调用所启动的​​所有线程块（Block）的集合​​
。它定义了整个并行计算任务的规模和空间。
你可以这样想象：​​线程​​是干活的小工，每个都做同样类型的工作但处理不同的材料。​​线程块​​是一个工作组，小工们在组内可以方便地共享工具（共享内存）和协调步骤（同步）。而工头（SM）管理多个工作组。​​Warp​​是工头派活的最小单位，一次派32个小工去干​​完全相同​​的活，这样管理起来最高效。


## Part 2: Understanding GPU performance

## Part 3: Putting it together — unpacking FlashAttention
