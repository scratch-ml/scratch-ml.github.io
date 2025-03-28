---
title: hello-world
date: 2025-03-25 17:33:41
tags: [Hexo, Fluid]
index_img: /img/cover.jpg
banner_img: /img/cover.jpg

---
Stepcast Router 的流量复制功能允许用户在模型服务级别和推理实例级别上复制流量。Router 会去配置中心读取流量复制规则，根据 source 模型服务（或实例）、target 模型服务（或实例）以及流量比例等参数来完成请求的复制、改写适配与分发。这里提供一个 CLI 工具  traffic_copy.py 脚本，通过它来对配置中心的流量规则进行读写，支持创建、更新、获取和删除流量复制规则。


<center class ='img'>
<figure>
    <img title="DOGE" src="a.jpg" width=300 height=200>
    <figcaption>肖像</figcaption>
  </figure>
</center>



备注：配置中心的地址是xxx，也可以直接点进这里配置、更新、查看、删除流量规则。


https://github.com/xlite-dev/CUDA-Learn-Notes

<!-- more -->

## Quick Start

### Create a new post

``` bash
$ hexo new "My New Post"
```

More info: [Writing](https://hexo.io/docs/writing.html)

### Run server

``` bash
$ hexo server
```

More info: [Server](https://hexo.io/docs/server.html)

### Generate static files

``` bash
$ hexo generate
```

More info: [Generating](https://hexo.io/docs/generating.html)

### Deploy to remote sites

``` bash
$ hexo deploy
```

More info: [Deployment](https://hexo.io/docs/one-command-deployment.html)