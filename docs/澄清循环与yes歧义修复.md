# 澄清循环与 yes 歧义修复（2026-09-16）

## 根因

先前系统保存了待回答的症状，却没有保存自己最近要求的对话输入类型。“yes”既可能回答症状问题，也可能回应帮助菜单，单靠逐条意图分类会丢失这一差别。加上 `waht` 和 `don't get what you said` 未被识别为理解困难，就出现重复澄清，甚至有把 yes 误记成症状肯定的风险。

## 当前行为

- 将 waht 归一化为 what；识别 don't get / don't understand / can't follow 等理解困难表达。
- 新增持久化字段 `dialogue_prompt`（none / description / menu / term）与 `clarification_attempts`，与待回答症状 `last_question` 分开。
- 对话仍等待描述、选项或术语时，裸 yes/no/ok/not sure 不会进入症状肯定/否定。
- 第一次含糊回答：说明单个 yes 不足以确定含义，给出具体输入方式。
- 再次含糊回答：停止重复问选择题，说明需要更具体的输入后才能继续；不修改症状证据。
- 菜单支持 1/2/3，分别进入描述引导、术语提问、使用帮助。恢复明确输入后重置澄清计数。
- 只有没有挂起的对话澄清、且确有待回答症状时，自由输入 yes/no 才可直接回答该症状。结构化症状回答接口仍按明确的症状 ID 处理。

LLM 继续负责分类和文字生成；上述状态约束决定哪些输入足以推进流程。生成回复重复、缺少下一步操作或重新提出 yes/no 问题时，使用明确回退，trace 保留 `clarification-fallback`，不会伪装为生成成功。

## 文件与兼容性

- `backend/app/models/chat_models.py`：持久化对话状态；旧会话缺失字段时采用默认值。
- `conversation_context.py`：拼写与理解困难识别、短回复绑定、旧菜单文本兼容识别。
- `chatbot_service.py`：提交回复时保存提示类型和澄清次数。
- `dialogue_ai.py` / `intent_service.py`：新的澄清动作、LLM 生成及防重复校验。
- `backend/tests/test_clarification_state.py`：拼写、旧会话、序列化、已有待答症状时的证据保护和明确输入恢复。

会话数据以 JSON 保存，新增默认字段无需破坏性数据库迁移。原工程数据库和用户已有聊天保留。

## 验证与复测

后端 131 项测试通过。真实模型回归脚本在两种条件下执行用户三轮原句：空会话，以及已经记录 Drowsiness 并存在待回答症状的会话。检查三轮动作、回复不重复、澄清计数及症状证据/轮数/待答症状保持不变。运行记录见 `evaluation/clarification-loop.json`，包括实际回复与生成/回退来源。

服务启动方式不变，需使用 `AI_PROFILE=local` 并重启后端；本次不需要重装依赖或重建前端。复测命令：

```powershell
conda activate gc1
cd E:\Study\StudyAtNUS\GC1_Project\frontend
$env:TEST_BASE_URL='http://127.0.0.1:8000'
node test-clarification.cjs
```

用户手动输入：`I dont get waht you said` → `yes` → `yes`。之后输入明确描述或问题，系统应能恢复正常流程。原来的聊天消息不会被重写。

这次解决的是输入绑定与重复澄清的状态问题，不代表小模型已能理解任意拼写错误或全部多轮语义。测试覆盖的自由文本 yes 与症状按钮操作有所区别，应按接口语义分别评估。

