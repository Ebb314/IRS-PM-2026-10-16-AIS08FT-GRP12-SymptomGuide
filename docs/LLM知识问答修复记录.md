# Ask a knowledge question：改为 LLM 解释（2026-09-15）

本记录替代上一版本“没有文档就返回 catalog-only 目录说明”的行为。

## 当前逻辑

1. 保持最新症状主题解析，知识提问不改动证据、轮数和待回答问题。
2. 有对应知识文档：继续走来源约束的 LLM 回答并显示实际引用。
3. 没有对应文档（如 Drowsiness）：调用本地 Qwen 生成简短、通俗的术语解释，不再把 CSV 元数据当作答案。
4. 新回答模式为 `local-llm-general`，`answer_basis=model-general-knowledge`，citations 为空。界面说明“来自模型知识，无检索来源”，不会把这类回答包装为已验证的 RAG。
5. 模型未启用时明确提示启用 local；模型生成或校验失败时提示重试。不会改回“is listed as a symptom node”文字，也不会假装调用成功。

模型提示要求围绕当前术语作 2–4 句一般性解释，不根据用户输入推断诊断，不提供处方/剂量，也不编造链接或文献。解释允许纯文本输出，也兼容合法 JSON；主题、长度、引用和部分越界表述有校验。这些校验不是完整医学事实核验，通用知识回答仍可能出错。

## 实现与验证

- backend/app/services/dialogue_ai.py：新增 general_knowledge，替换无文档时的目录返回。
- frontend/src/components/AITrace.vue：区分模型通用知识与检索来源。
- backend/tests/test_full_kg.py：覆盖模型调用分支、纯文本/JSON、虚构引用、主题错误和并发初始化；当前后端 93 项测试通过。
- backend/app/dependencies.py：在缓存外加初始化锁，防止页面词表请求与首条聊天并发加载模型，导致导入异常或重复占用显存。
- frontend/browser-knowledge.cjs：真实页面点击按钮，验证 Drowsiness 和 Headache 的 LLM 回答、主题切换与状态不变。
- evaluation/llm-knowledge-browser.json：保存真实模型回答及 token/耗时记录。

最终页面实测通过：Drowsiness（P986）与随后报告的 Headache（P885）均返回 `local-llm-general`，模型为 Qwen2.5-1.5B-Instruct、设备 CUDA。暂存工程测试生成分别为 38/50 个 token，耗时约 1.71/2.25 秒（仅生成耗时，不包含首次加载）。测试同时验证无虚构引用、最新主题切换、证据/轮数/待回答问题不被知识提问改变。该结果只代表这两个真实模型用例，不代表全量症状的医学解释已核验。页面截图见 `screenshots/llm-drowsiness.png`。

当前运行服务必须使用 `AI_PROFILE=local`。已有模型、Conda 环境和启动方式无需重新安装。修改 Python 后需重启服务；前端已重新构建。

本次真实测试先发现模型生成了纯文本解释，但被 JSON 格式要求拒绝，因此将术语解释与结构化 NLU 的输出协议分开；没有放宽 NLU 的结构化抽取协议。

手动启动后端（Anaconda PowerShell Prompt）：

```powershell
conda activate gc1
cd E:\Study\StudyAtNUS\GC1_Project
$env:AI_PROFILE='local'
$env:REASONING_MODE='random'
$env:NLP_ENGINE='rules'
$env:NLP_USE_TRANSFORMERS='false'
$env:HF_HUB_OFFLINE='1'
python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
```

若 8000 端口已有本项目服务，先在其终端按 Ctrl+C，再运行启动命令。无需再次下载模型。

浏览器复跑：

```powershell
cd E:\Study\StudyAtNUS\GC1_Project\frontend
$env:TEST_BASE_URL='http://127.0.0.1:8000'
node browser-knowledge.cjs
```

输入 `I have Drowsiness.`，点击 Ask a knowledge question，页面应出现模型生成的解释和 Qwen 调用信息。此前保存的聊天记录不会被改写；需要再次提问才能看到新回答。
