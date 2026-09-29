# AI 升级验收记录

日期：2026-09-15。最终交付：`E:\Study\StudyAtNUS\GC1_Project`。

## 已验证

| 层次 | 结果 | 证据 |
| --- | --- | --- |
| Python 后端单元 / 契约 / 故障回归 | 78 passed | backend/tests，使用现有 gc1 Python 3.11 |
| 旧规则 NLP 回归 | 54 条通过 | evaluation/report.json、REPORT.md |
| 前端类型检查与生产构建 | 通过 | npm run build |
| 原有真实 Chrome 流程 | 11 项通过 | evaluation/browser-report.json |
| 真实 Qwen + MiniLM Chrome 流程 | 8 项通过 | evaluation/browser-ai-report.json |
| 真实模型 NLU | 24 条，混合 F1 0.9091；规则基线 0.6286 | evaluation/ai_report.json |
| 检索消融 | 9 条，BM25 / dense 各 7/9；融合 8/9 Recall@3 | 同上 |
| 槽位 | 7 个标注槽位精确匹配 | 同上；开发中使用过该样例集 |
| 对话 | 4 条合法来源生成、1 条原文回退、1 条无支持拒答 | 同上 |

模型：Qwen2.5-1.5B-Instruct。硬件：RTX 3060 Laptop GPU。推理使用 float16 / CUDA，greedy decoding。实际 Python 包版本见 evaluation/local_environment.json。运行模型时不调用云 LLM API；固定版本下载完成后仅本地加载。

## 真实 AI 浏览器覆盖

症状输入、模型调用与槽位显示；带引用的模型回答；知识问题不改变证据 / round / pending；从权威状态总结；省略回答绑定待回答症状；指令式消息不改变证据；刷新恢复；移动端无横向溢出与无 JavaScript 错误。测试使用虚构输入，并在 finally 中删除测试会话。

截图：docs/screenshots/ai-nlu.png、ai-rag.png。模型推理是实际调用，疾病候选与问题算法是 mock。两者不能混淆。

## 发现与修复

1. 0.5B 初步探针频繁复制 JSON 占位符，改用具体 few-shot 并试用 1.5B。提示和模型同时变化，不把这当作受控模型大小消融。
2. 初次槽位 2/7，加入单实体保守原文投影后 7/7；收益属于混合系统，非纯 LLM。
3. 检索融合不应让零分文档投票；修正后重新运行单路与 RRF 评测。
4. 真实浏览器发现省略回答输出空集或误判；缩减上下文仍不足，因此拆成标签分类与程序绑定，并用显式断言保护，记录分歧。
5. 解释问题曾生成“进一步诊断”的不当用途，加入校验并退回真实项目说明。

初次模型报告 ai_report_initial.json 与上下文失败响应 context_failure.json 保留为历史证据；最终报告是 ai_report.json。后续修复使用过评测样例，数据属于开发回归集，不能称为独立泛化或临床测试。

## 限制与未完成的外部事项

没有真实队友算法联调、DDXPlus/PrimeKG 对齐、临床验证、微调、外部独立测试集、用户量或线上部署证据。剩余 NLP 失败逐条保留在最终报告。引用 / span 合法不保证语义正确，范围路由不能覆盖所有攻击。GPU 并发吞吐与 CPU 性能未测。

非失败提醒：Starlette/httpx 和 anyio 两项弃用提示；Transformers 的 torch_dtype / generation flags 提醒；Vite 主 JS 包约 1.09 MB 的体积提醒。没有为了隐藏提醒而提高阈值或声称已优化。

## 复跑

按 docs/AI升级运行手册.md 逐条执行命令，无需任何环境配置脚本。真实模型评测要先关闭另一份 GPU 服务，浏览器测试则需要启动服务。质量评测的退出码只表示脚本执行完成，正确性必须看报告指标和逐条结果。
