---
title: cuda-graph
date: 2025-09-19 02:29:34
tags:
hide: true
---




<!-- 因为cuda graph要求静态shape， prefill做不到，因为prompt的长度不确定！但是decode阶段的长度固定为1，只是batch size不确定。因此，vllm为decode捕获了多个batch size版本的graph，实例运行时可以padding到最近的batch size版本，实现推理。

作者：yangxianpku
链接：https://www.zhihu.com/question/7987565201/answer/1887063079948379889
来源：知乎
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。 -->