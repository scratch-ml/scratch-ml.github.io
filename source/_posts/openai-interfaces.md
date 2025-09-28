---
title: OpenAI 接口演进
date: 2025-09-24 12:14:03
tags:
---

## OpenAI 接口演进：Completion、Chat Completions 与 Responses 全面对比

最近，OpenAI 在开发者博客发布《Why we built the Responses API》，系统阐述了为何在 Completions 与 Chat Completions 之后推出全新的 Responses 接口：它以“推理—行动—汇报”的统一循环为中心，原生支持多模态、托管工具与持久化推理状态，面向智能体与复杂工作流而来（见：[Why we built the Responses API](https://developers.openai.com/blog/responses-api/)）。基于这篇官方说明与最新文档，本文将梳理 OpenAI 接口从 Completions → Chat Completions → Responses 的演进脉络，并给出何时选用各接口的建议与最小可跑示例。

本文主要覆盖的接口：
- Completions API（`/v1/completions`）
- Chat Completions API（`/v1/chat/completions`）
- Responses API（`/v1/responses`）

### 核心对比表

| 维度 | Completion API (`/v1/completions`) | Chat Completions API (`/v1/chat/completions`) | Responses API (`/v1/responses`) |
|---|---|---|---|
| 核心定位 | 传统“给定 prompt → 续写文本” | 多轮对话（messages 角色制） | 面向智能体的“推理-行动-汇报”统一接口 |
| 交互/输入 | 单一 `prompt` | `messages` 数组（`system/user/assistant`） | 统一 `input` 项，可混合文本/图像等 |
| 状态管理 | 无状态 | 无状态（客户端拼接历史） | 有状态：可用 `previous_response_id` 延续上下文与推理状态 |
| 工具能力 | 不支持 | 函数调用（客户端执行并回填结果） | 原生托管工具（如 code interpreter、文件检索；Web 搜索在 OpenAI 侧可见，Azure Responses 当前不支持） |
| 多模态 | 仅文本 | 以文本为主，部分模型扩展视觉 | 原生多模态（文本/图像等输入为一等公民） |
| 返回结构 | `choices[].text` | `choices[].message` | 多种“输出条目 Items”（消息、工具调用、推理摘要等），SDK 提供 `output_text` 快捷取值 |
| 流式输出 | SSE token 流 | SSE token 流（`stream=true`） | 语义化事件流（如 `response.output_text.delta`、工具事件等） |
| 典型场景 | 摘要、改写、翻译、单轮代码补全 | 聊天机器人、客服问答、函数调用协作 | 智能体/自动化工作流、多工具编排、长会话与推理密集任务 |

### 选择建议

- **只需单轮文本生成**：选 Completion。
- **传统多轮聊天，自己维护上下文**：选 Chat Completions。
- **需要多模态、工具编排、服务端保存会话/推理状态、事件级流式**：选 Responses（官方主推未来方向；注意 Azure Responses 目前不支持 Web 搜索工具）。

### 最小可跑示例（Python）

#### Completion：单轮文本续写
```python
from openai import OpenAI
client = OpenAI()
resp = client.completions.create(
    model="gpt-3.5-turbo-instruct",
    prompt="Write a one-sentence bedtime story about a unicorn."
)
print(resp.choices[0].text.strip())
```

#### Chat Completions：多轮对话与函数调用
```python
from openai import OpenAI
client = OpenAI()

# 基本对话
chat = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Explain what an API is in one sentence."}]
)
print(chat.choices[0].message.content)

# 函数调用（模型决定是否调用；客户端负责执行后再回传）
chat_fc = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    functions=[{
        "name": "get_weather",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"]
        }
    }],
    tool_choice="auto"
)
print(chat_fc.choices[0].message)  # 检查是否需调用函数
```

#### Responses：服务端会话与事件流、工具一体化
```python
from openai import OpenAI
client = OpenAI()

# 1) 基本调用
r1 = client.responses.create(model="gpt-4.1", input="Explain what an API is.")
print(r1.output_text)

# 2) 继续对话（仅传上一次响应ID，无需携带完整历史）
r2 = client.responses.create(
    model="gpt-4.1",
    previous_response_id=r1.id,
    input="Summarize in one sentence."
)
print(r2.output_text)

# 3) 使用托管工具（示例：代码解释器；Web 搜索在 Azure Responses 暂不支持）
r3 = client.responses.create(
    model="gpt-4.1",
    tools=[{"type": "code_interpreter", "container": {"type": "auto"}}],
    instructions="Solve 3x + 11 = 14 using Python.",
    input="Please compute and show the steps."
)
print(r3.output_text)

# 4) 事件流（语义增量）
stream = client.responses.create(model="gpt-4.1", input="Tell me a joke.", stream=True)
for event in stream:
    if event.type == "response.output_text.delta":
        print(event.delta, end="")
```

### 注意要点

- **状态延续**：Responses 通过 `previous_response_id` 延续上下文，亦可检索既往 response，降低长对话令牌与客户端拼接成本。
- **事件级流式**：不仅有文本增量，还包含工具调用等事件，便于构建可观测的智能体 UI。
- **工具矩阵**：OpenAI Responses 已整合 code interpreter、文件检索等；Web 搜索在 OpenAI 示例与 Cookbook 可见，Azure 官方文档明确“当前不支持”。

### 参考链接

- 微软文档：Azure Responses API（含状态、`previous_response_id`、事件流、代码解释器与已知不支持项）— 参见
  - [Responses API 指南（含生成/检索/删除/链式/流式）](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/completions)
  - [REST 参考（含 Completions 返回结构）](https://learn.microsoft.com/en-us/azure/cognitive-services/openai/reference)
- OpenAI 开发者博客：为什么构建 Responses（状态化、多模态、托管工具、语义流等）— [Why we built the Responses API](https://developers.openai.com/blog/responses-api/)
- OpenAI Cookbook：Responses 的 Web 搜索与状态管理示例—[Web Search and States with Responses API](https://cookbook.openai.com/examples/responses_api/responses_example)
- OpenAI Agents SDK：Responses 事件流语义—[Streaming（Agents SDK）](https://openai.github.io/openai-agents-python/streaming/)
- Chat Completions 形态参考（第三方兼容文档）—[OpenRouter Chat completion](https://openrouter.ai/docs/api-reference/chat-completion)
- 函数调用实践与要点（Chat Completions）—[Function Calling Tutorial](https://www.hackwithgpt.com/blog/function-calling/)

