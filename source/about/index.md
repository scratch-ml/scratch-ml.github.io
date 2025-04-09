---
title: About
layout: about
date: 2025-03-25 20:44:17
---
#### 关于博主
目前，我在大模型初创公司**阶跃星辰**担任研发总监，从事大模型推理服务、推理框架开发和优化工作，并将专注于这一领域，持续提升自己在 LLM Inference 方向的技术能力。在此之前，我曾以全职员工的身份在小米、阿里巴巴分别工作过一段时间，从事操作系统研发工作；也在字节跳动以研究实习生的身份工作近两年，从事 DRL 模型推理服务开发、推理 SDK 开发、DRL 训练框架开发与优化等工作。
#### Education
- **南京大学** *2014/09~2018/06*
  - **学位**：本科，计算机科学与技术系
- **清华大学** *2019/08~2022/06*
  - **学位**：硕士，软件学院
  - **导师**：Prof.[Zhenhua Li](https://www.thucloud.com/zhenhua/index.html)
#### Work Experience
- **小米通讯技术有限公司** *2018/07~2019/08*
  - **岗位**：Linux 内核开发工程师，手机部-性能优化组
  - **职责**：负责小米手机系统内核的性能优化，包括调度器、文件系统等。
- **字节跳动科技有限公司** *2020/01~2021/11*
  - **岗位**：研究实习生(Supervised by Dr.[Yibo Zhu](http://yibozhu.com/))，AML-Machine Learning Systems 组
  - **职责**：进行 DRL 模型推理服务开发、推理 SDK 开发、DRL训练框架开发与优化等工作。
- **阿里巴巴-云智能集团** *2022/06~2023/08*
  - **岗位**：高级开发工程师，基础产品事业部-操作系统组
  - **职责**：进行云原生场景下操作系统层级的性能优化工作，包括启动优化、资源分配优化等。
- **阶跃星辰智能科技有限公司** *2023/08~至今*
  - **职责**：LLM 推理服务开发、推理框架开发等。

#### Projects
##### Model Inference
- **DRL 模型推理服务开发**
- **LLM 服务开发与优化**
##### 操作系统
- **[Android 系统异步 discard 优化](https://www.thucloud.com/zhenhua/papers/[MobiCom'20]%20Android%20Not%20Responding.pdf)**：参与 ANR(Android Not Responding)/Watchdog 异常监控及自动化分析工具的研发，定位应用、系统卡顿的根源。并针对 ext4 文件系统在闪存实时 discard 机制的缺陷，设计了可中断、可恢复的异步 discard 策略，使得 ANR 发生率下降 32%、Watchdog 异常发生率降低 46%，相关成果被 MobiCom'20 收录，本人为第一作者。
- **[k8s 集群千节点启动优化](https://mp.weixin.qq.com/s/C-Cx7wGmwFIiW8ugnBcbRg)**：分析 k8s 集群高并发启动节点下速度较慢的瓶颈，通过调整限流、预置关键镜像等手段优化链路瓶颈，把千节点并发启动 P90 耗时从 506s 降低至 54s。
- **云原生资源管理组件开发**：独立开发 k8s 节点上的资源管理组件，劫持 kubelet 与 Contained 之间的交互请求并进行资源管理与分配，将内核特性透出(如 CPU burst 能力)，提高单机的资源隔离与超卖能力。

<!-- - **Android 系统异步 discard 优化**：参与ANR/Watchdog问题监控及自动化分析工具研发，定位系统卡顿根源。针对ext4文件系统在闪存实时discard机制的缺陷，设计了可中断、可恢复的异步discard策略，实现ANR发生率下降32%、Watchdog发生率降低46%。成果以第一作者身份发表于计算机领域顶会 MobiCom'20（CCF-A）。
- **千节点启动优化**：分析 k8s 集群节点高并发启动场景下速度较慢的根因，进行节点级启动优化。通过瓶颈限流调整、探测频率调整、预置关键镜像等手段，将 1000 节点并发启动 P90 耗时从 506s 降低至 54s，上线至超 2w 核 CPU 的业务使用。
- **云原生资源管理组件开发**：独立开发了 k8s 节点上工作负载的资源管理组件，劫持 kubelet 与 Contained 之间的交互请求，并在关键的 hook 点调用节点资源控制器进行资源管理与分配，将底层内核特性进行透出(如 CPU burst)，提高单机的资源隔离与超卖能力。 -->
#### Publications
- ***[Aging or Glitching? Why Does Android Stop Responding and What Can We Do About It?](https://www.thucloud.com/zhenhua/papers/[MobiCom'20]%20Android%20Not%20Responding.pdf)*** ***Mingliang Li***, *Hao Lin, Cai Liu, Zhenhua Li, Feng Qian3, Yunhao Liu, etc.* （**MobiCom**, 2020）
- ***[ParliRobo: Participant Lightweight AI Robots for Massively Multiplayer Online Games (MMOGs)](https://www.thucloud.com/zhenhua/papers/[MM'23]%20ParliRobo.pdf)*** *Jianwei Zheng, Changnan Xiao, ***Mingliang Li***, Zhenhua Li, Feng Qian, Wei Liu, etc.* （**MM**, 2023）
- ***[Aging or Glitching? What Leads to Poor Android Responsiveness and What Can We Do About It?](https://www.thucloud.com/zhenhua/papers/TMC'24%20Poor%20Android%20Responsiveness.pdf)*** *Hao Lin, Cai Liu, Zhenhua Li*, Feng Qian, ***Mingliang Li***, Ping Xiong, and Yunhao Liu*. (**TMC**, 2024)
- ***[Step-Video-T2V Technical Report: The Practice, Challenges, and Future of Video Foundation Model](https://arxiv.org/abs/2502.10248)***  Step-Video-T2V Team.(Arxiv, 2025)
- ***[Step-Audio: Unified Understanding and Generation in Intelligent Speech Interaction]()*** Step-Audio Team.(Arxiv, 2025)
#### Awards
- 2021 年，清华大学综合一等奖学金
- 2022年，清华大学优秀毕业论文
- 2022年，清华大学软件学院优秀毕业生


<!-- - 2021 年，清华大学综合一等奖学金-广联达奖学金（第 2/17）
- 2022年，清华大学优秀毕业论文（前 6/85）
- 2022年，清华大学软件学院优秀毕业生 （前 12/85） -->