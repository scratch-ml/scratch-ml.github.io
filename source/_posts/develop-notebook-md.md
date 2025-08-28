---
title: develop-notebook.md
date: 2025-06-26 15:15:04
draft: true
hide: true
tags:
---
## 一. vllm 优雅退出
1. k8s pod 滚动更新时的机制：kill -TERM <PID>
2. 业务进程需要监听信号，并处理
3. 业务进程需要 exec 才能成为主进程，否则接收不到主进程的信号

## 二. async context manager 使用不规范，导致请求耗时额外增加 40 ms
__aexit__ 方法中，xxx


## 三.【严重】异常处理机制不完善，导致 async task 退出
asyncio.exceptions.CancelledError 并不是 Exception，而是 BaseException，所以不会被 except 捕获

## 四. TBD
