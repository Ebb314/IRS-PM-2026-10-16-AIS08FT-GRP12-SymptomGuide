# Ollama 本地模型接入（LLM_BACKEND=ollama）

## 做了什么

原来的 local profile 通过 transformers 直接加载仓库里的 Qwen3-4B，需要 PyTorch、bitsandbytes 和 CUDA。现在新增了 Ollama 后端，可以使用本机 Ollama 已部署的任意模型，例如 `qwen3:8b`。

- 对话、意图理解、证据提议、知识问答的逻辑都没有改。它们只调用模型的 `generate()`、`chat()` 和 `.name`，新增的 `OllamaGenerator`（`backend/app/services/ollama_runtime.py`）实现了同样的接口。
- 不需要安装 PyTorch / transformers，原来的 `requirements.txt` 就够用。
- 模型调用出错或输出格式不合格时，沿用原有的可见回退逻辑，例如改用规则抽取，或提示“未能生成可靠解释”，不会编造回答。

## 使用步骤

1. 确认 Ollama 正在运行，并查看模型的准确名称：

   ```powershell
   ollama list
   ```

   默认使用 `qwen3:8b`。如果你的模型名称不同，在启动前设置：

   ```powershell
   $env:OLLAMA_MODEL='你的模型名'
   ```

2. 启动：

   ```powershell
   conda activate gc1
   powershell -ExecutionPolicy Bypass -File .\scripts\run.ps1 -Profile ollama
   ```

3. 检查是否生效：打开 http://127.0.0.1:8000/api/health ，应该看到 `"ai_profile":"local"` 和 `"generator":"ollama:qwen3:8b"`。如果 Ollama 没启动或模型名写错，服务会在启动时直接报错，并列出已安装的模型。

## 可选：语义检索（推荐）

原来的 local profile 用 MiniLM 嵌入做语义检索，把 "feel very tired" 这类口语对应到 Fatigue 等概念。Ollama 后端默认不做嵌入，检索退回 BM25 关键词匹配，口语表达的识别会弱一些。想开启语义检索：

```powershell
ollama pull nomic-embed-text
pip install numpy
$env:OLLAMA_EMBED_MODEL='nomic-embed-text'
powershell -ExecutionPolicy Bypass -File .\scripts\run.ps1 -Profile ollama
```

第一次启动时需要为 15,135 个症状名计算嵌入，大约需要一到几分钟。结果缓存在 `backend/runtime/embeddings/`，之后启动会直接读取缓存。

## 环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `LLM_BACKEND` | `transformers` | `-Profile ollama` 会自动设为 `ollama` |
| `OLLAMA_MODEL` | `qwen3:8b` | Ollama 模型名 |
| `OLLAMA_URL` | `http://127.0.0.1:11434` | Ollama 地址 |
| `OLLAMA_TIMEOUT_SECONDS` | `180` | 单次请求超时时间 |
| `OLLAMA_THINK` | `false` | Qwen3 思考模式。项目需要直接输出 JSON 或文本，建议保持关闭；即使开启，`<think>` 内容也会被去掉 |
| `OLLAMA_EMBED_MODEL` | 空 | 嵌入模型；为空时使用 BM25 检索 |
| `OLLAMA_NUM_CTX` | `8192` | 上下文长度。所有请求必须使用同一个值，否则 Ollama 每次都会重新加载模型 |
| `OLLAMA_KEEP_ALIVE` | `30m` | 两轮对话之间模型保持加载的时长 |

## 注意事项

- **追问按钮**：kg 模式下，Informed Search 选出的问题会附在 LLM 回复之后，并显示 Yes / No / Not sure 按钮（LLM 被告知不要自己提问）。点按钮回答不调用 LLM，几乎即时返回。random 模式保持队友原来的设计：不显示按钮。
- **响应速度**：一轮对话通常会调用模型两次（理解一次、回复一次）。8B 模型在显卡上一般每轮几秒；只用 CPU 时会慢很多，前端会显示等待计时。
- **思考模式**：请求里会带 `"think": false`。如果你的 Ollama 版本较旧，不认识这个参数，模型输出的 `<think>…</think>` 也会在代码里被去掉。
- **测试**：`backend/tests/test_ollama_runtime.py` 用模拟的 Ollama 接口测试了请求格式、出错处理、启动检查、嵌入缓存，以及“Abnormality of the immune system”这类知识问答由模型回答。真实回答质量需要用你本机的模型实际体验。
- **原来的 Qwen3-4B 方式依然可用**：`run.ps1 -Profile local`。
