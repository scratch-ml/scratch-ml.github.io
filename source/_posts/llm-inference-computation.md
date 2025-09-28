---
title: LLM 推理计算详解
date: 2025-06-17 00:28:00
tags: [LLM, Inference, Computation, Deep Learning]
draft: true
hide: true
---


# cs336 talk2
### 在 1024 张 H100 上训练一个 70B 参数的模型，总共灌进去 15 T tokens，需要多久？

total_flops = 6 * 70e9 * 15e12  #这里的 6 来自于哪里？
assert h100_flop_per_sec == 1979e12/2
mfu = 0.5
flops_per_day = h100_flop_per_sec * mfu * 1024 * 24 * 3600
days = total_flops / flops_per_day

143.9 days


### 基于 AdamW 在 8*H100 上最大可以训练多大的模型？
h100_bytes = 80e9
bytes_per_parameter = 4 + 4 + (4 + 4) # parameter + gradient + optimizer state
num_parameters = 8 * 80e9 / bytes_per_parameter

40B parameters（注意：这里为考虑 activation 的显存占用，因为和 batch size 和 seq length 有关）






