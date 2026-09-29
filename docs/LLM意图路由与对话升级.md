# LLM 意图路由与上下文对话升级

## 项目定位与调用链

此前 `route()` 主要通过正则判断输入，未命中的输入默认成为症状报告；问候、产品问题和无关请求容易进入症状抽取。现在 local 模式下，每条自由输入先调用 Qwen 判断意图，再进入对应处理器。按钮产生的结构化 Yes/No/Not sure 和证据纠正操作已有明确意图，继续走确定性接口。

调用链：`ChatbotService._execute → DialogueAI.classify → IntentService.classify → 意图处理器 → 会话提交`。

| 意图 | 行为 |
|---|---|
| report | 进入现有上下文 NLP、证据合并及队友模块适配器 |
| knowledge | 解析明确或最近症状，优先现有来源约束生成；无对应文档时用模型知识解释；无词表主题的健康问题进入一般教育回答 |
| summary | LLM 根据结构化证据和状态生成摘要 |
| explain | LLM 根据实际模式、待回答问题和项目事实解释追问/分数 |
| help | LLM 回答操作、能力、流程等问题，只提供已传入的项目事实 |
| social | LLM 回应问候、感谢并引导下一步 |
| out_of_scope | LLM 简短回应边界并提供项目内替代选项 |
| clarify | LLM 询问用户究竟要报告症状、问知识还是了解项目操作 |

## 意图决策与状态保护

分类模型输出受枚举限制的单个标签，同时兼容 JSON。输入包括当前消息、最新症状主题、待回答症状，以及用 BM25 + MiniLM + RRF 检索出的最多四条相似标注示例。示例库有 26 条句子，覆盖八类意图；这是对话路由示例，不是症状词表，症状支持范围仍由完整 CSV 决定。

若 LLM 标签与高度相似的标注示例冲突（cosine≥0.80，跨类别差值≥0.15），再次调用 LLM 判断两句是否表达同一意图。确认后使用示例标签，否则澄清；这个复核过程记录在 consistency_check。格式错误或推理异常也进入澄清，不猜测后修改证据。模型不输出 confidence 时保存 null，不虚构可靠性分数；兼容 JSON 中的自评 confidence，低于 0.70 时澄清，但该值不是经过校准的概率。模型不能保证每次正确识别意图，多意图输入仍有误分风险。

非 report 分支不调用召回与搜索，也不修改 evidence、round、pending question 和 knowledge focus；仍会增加消息与会话版本，以保留聊天历史和并发控制。report 分支继续由已有 NLP 校验与适配器约束执行，分类标签本身不能直接添加症状或执行操作。

对话生成使用当前结构化证据、实际 reasoning mode 和项目事实。random 模式必须描述为随机候选/追问测试，不能冒称真实临床推理或把分数解释为概率。模型输入区分用户问题和结构化项目事实。输出有长度、部分诊断/剂量和伪造引用检查，这些检查不是完整语义核验或医学事实验证；摘要仍可能产生措辞误差，结构化证据面板是记录依据。

范围外和澄清回答使用已选择的回应目标约束生成，避免继续执行无关请求。范围外回答若缺少边界声明，则添加固定边界前缀并保留模型正文，记录 scope_notice_added；澄清回答必须包含三个方向。校验失败时提供明确的安全回退，并记录 dialogue-failed，不能把回退当成成功生成。随机追问解释不向生成模型传入数值分数，减少臆造患病概率的机会。

## 验证记录与局限

后端回归测试 115 项通过，前端 TypeScript 检查与生产构建通过。真实 Qwen 主回归集 10/10、改写表达集 6/6 路由标签通过，并验证非报告对话不修改证据、轮数和待回答问题。报告包含实际生成文本、模型耗时、检索示例、初次标签及二次确认结果。`llm-intents-initial.json` 保留早期失败结果。

这些是开发回归样本，其中部分与检索示例相同，改写集也用于发现并修复问题；文件名 holdout 沿用测试脚本命名，不应当作严格未见测试集。16/16 不代表真实用户总体准确率，也不证明全部回应事实正确。小模型仍可能用错界面标签或输出不够自然；应结合实际结构化面板确认操作。后续评估需要独立收集多意图、中文、长上下文和恶意指令样本。

## 实现文件

- `backend/app/services/intent_service.py`：分类契约、上下文、置信度门槛、分类/生成失败处理、各意图提示词。
- `backend/app/services/dialogue_ai.py`：连接新路由及通用对话，保留症状知识生成。
- `backend/app/services/chatbot_service.py`：自由输入在抽取前做意图判断，保存 routing trace。
- `frontend/src/components/AITrace.vue`：显示路由模式、自评置信度和实际分类模型调用 token/耗时。
- `backend/tests/test_intent_service.py`：八类标签、无效/低置信度结果与非报告状态保护。
- `frontend/test-intents.cjs`：真实模型连续会话测试，保存 `evaluation/llm-intents.json`。

## 运行和测试

无需新装依赖或下载模型。停止旧服务后，在 Anaconda PowerShell Prompt 执行：

```powershell
conda activate gc1
cd E:\Study\StudyAtNUS\GC1_Project
$env:AI_PROFILE='local'
$env:REASONING_MODE='random'
$env:NLP_ENGINE='rules'
$env:NLP_USE_TRANSFORMERS='false'
$env:HF_HUB_OFFLINE='1'
python -m uvicorn app.main:app --app-dir backend --port 8000
```

打开 `http://127.0.0.1:8000/`。另一终端执行真实模型测试：

```powershell
cd E:\Study\StudyAtNUS\GC1_Project\frontend
$env:TEST_BASE_URL='http://127.0.0.1:8000'
node test-intents.cjs
node browser-knowledge.cjs
```

后端隔离回归（使用测试替身，不占用模型）：

```powershell
cd E:\Study\StudyAtNUS\GC1_Project
$env:AI_PROFILE='offline'
python -m pytest backend/tests -q
```

前端源码修改后在 frontend 执行 `npm run build`，再刷新浏览器。offline 模式仍使用旧规则降级，不应作为本次 LLM 能力的演示配置。新增分类调用会增加每轮延迟；本项目目前使用单个本地小模型，未实现流式输出或生产级并发调度。

## 求职展示边界

可展示的工程亮点是上下文感知 LLM 路由、结构化决策契约、低置信度澄清、状态隔离、模块适配器和可审计模型调用。测试通过率应引用报告中的具体样本数量，不能声称识别任意输入或验证了全部症状的回答质量。


