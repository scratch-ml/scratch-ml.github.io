---
title: tokenization
date: 2025-09-28 23:16:48
tags:
---


### 1. 概述

分词器(Tokenizer)是自然语言处理中的基础组件，用于在字符串和整数序列之间进行转换。其主要功能包括：
- 将文本字符串编码(encode)为整数序列(tokens)
- 将整数序列解码(decode)回原始文本
- 建立词汇表(vocabulary)，定义所有可能的token

<center class ='img'>
<figure>
    <img title="分词示例" src="https://cdn.jsdelivr.net/gh/scratch-ml/scratch-ml.github.io@main/source/_posts/tokenization/tokenized-example.png" width=600>
    <figcaption>分词示例</figcaption>
  </figure>
</center>

### 2. 常见分词方法

#### 2.1 字符级分词(Character-based Tokenization)

字符级分词是最直观的分词方式，它将文本按照单个Unicode字符进行切分。Unicode是一个将字符映射到整数码点(code point)的文本编码标准，截至2024年9月发布的 Unicode 16.0版本，该标准定义了 154,998个字符，涵盖168种文字系统。每个字符都对应一个唯一的整数ID，比如字符"s"的码点是115（通常记作U+0073），而汉字"牛"的码点是29275。在 Python 中，我们可以使用`ord()`函数将字符转换为整数，用`chr()`函数将整数转换回字符：

```python
>>> ord('s')
115
>>> ord('牛')
29275
>>> chr(115)
's'
>>> chr(29275)
'牛'
```


从实现角度来看，字符级分词几乎不需要任何预处理。你只需要遍历文本中的每个字符，然后查找它对应的Unicode码点即可。这种简单性使得它成为很多研究项目的首选，特别是在处理多语言文本或包含大量特殊符号的场景时。

然而，字符级分词也有明显的局限性。最大的问题是词汇表过于庞大，导致模型的嵌入层参数量激增。更糟糕的是，大部分Unicode字符在实际应用中很少出现，这意味着模型需要为大量低频字符分配参数空间，造成了严重的资源浪费。

另一个问题是序列长度。由于每个字符都是一个token，原本一个单词可能需要拆分成十几个字符token，这大大增加了序列的长度。对于基于注意力机制的模型来说，这意味着计算复杂度的平方级增长，严重影响了训练和推理效率。

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


BPE被广泛采用，主要是因为它巧妙地解决了自然语言处理中的几个关键问题：
- 彻底解决未登录词问题：在传统的单词级分词中，模型遇到训练时没见过的词（如专业术语、新词）时，只能将其标记为<UNK>（未知符号），导致信息丢失。BPE将这类罕见词或新词拆解为已知的子词单元（例如将 "unseen" 拆为 "un" 和 "seen"），从而让模型能够处理
- 平衡词汇表大小与语义粒度：单词级分词词表可能异常庞大（英文可达数十万词），导致模型参数爆炸。字符级分词虽词表极小，但序列过长，模型难以学习语义。BPE在两者间取得平衡，通过一个大小可控（通常为3万至10万）的词表，既保留了常见词的完整性，又将低频词拆分为有意义的子词
- 强大的多语言适应性：BPE不依赖任何语言的预先定义规则或词典，完全基于数据驱动。这使得它能够无缝处理多种语言混合的文本，并对德语、土耳其语等形态丰富的语言有很好的效果


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

### 可运行实现（含 `merge` 与 `BPETokenizerParams`）
```python
from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, Tuple, List

@dataclass(frozen=True)
class BPETokenizerParams:
    vocab: Dict[int, bytes]
    merges: Dict[Tuple[int, int], int]

def merge(indices: List[int], pair: Tuple[int, int], new_index: int) -> List[int]:
    i = 0
    out: List[int] = []
    a, b = pair
    n = len(indices)
    while i < n:
        if i < n - 1 and indices[i] == a and indices[i + 1] == b:
            out.append(new_index)  # @inspect out
            i += 2
        else:
            out.append(indices[i])  # @inspect out
            i += 1
    return out

def train_bpe(string: str, num_merges: int) -> BPETokenizerParams:  # @inspect string, @inspect num_merges
    # Start with the list of bytes of string.
    indices = list(map(int, string.encode("utf-8")))  # @inspect indices
    merges: Dict[Tuple[int, int], int] = {}  # index1, index2 => merged index  # @inspect merges
    vocab: Dict[int, bytes] = {x: bytes([x]) for x in range(256)}  # index -> bytes  # @inspect vocab
    next_index = 256
    for i in range(num_merges):
        if len(indices) < 2:
            break
        # Count the number of occurrences of each pair of tokens
        counts: Dict[Tuple[int, int], int] = defaultdict(int)
        for index1, index2 in zip(indices, indices[1:]):  # For each adjacent pair
            counts[(index1, index2)] += 1  # @inspect counts
        if not counts:
            break
        # Find the most common pair.
        pair = max(counts, key=counts.get)  # @inspect pair
        index1, index2 = pair
        # Merge that pair.
        new_index = next_index  # @inspect new_index
        next_index += 1
        merges[pair] = new_index  # @inspect merges
        vocab[new_index] = vocab[index1] + vocab[index2]  # @inspect vocab
        indices = merge(indices, pair, new_index)  # @inspect indices
    return BPETokenizerParams(vocab=vocab, merges=merges)

if __name__ == "__main__":
    s = "banana bandana"
    params = train_bpe(s, num_merges=10)
    print("Final merges:", params.merges)
    print("Final vocab size:", len(params.vocab))
```

### 如何快速调试
- 直接在你使用的 IDE/扩展里运行该脚本，`# @inspect` 会在相应位置显示中间变量值（如 `indices`、`counts`、`pair`、`new_index`、`merges`、`vocab`）。
- 若需要断点细看每一轮，可在循环内部加一行：
```python
import ipdb; ipdb.set_trace()
```
- 想缩小输出规模调试，可先用更短字符串（如 `"banana"`）与较小的 `num_merges`（如 3）。

- 若你已有某段现成实现但行为异常，贴出那段代码或异常现象/输入输出，我可以对比定位具体问题。

- 本次我：全局搜索后确认仓库内没有该函数的真实实现，于是提供了一个最小可运行版本并在关键变量处保留 `# @inspect` 注释，方便你逐步观察每次合并的状态。