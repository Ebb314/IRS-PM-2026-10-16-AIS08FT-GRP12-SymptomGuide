# Symptom log 显示与快捷功能移除

日期：2026-09-17。

本次检查发现：之前组件标题是 My symptoms；窗口宽度不超过 760px 时，整个侧栏会排在聊天区下面；没有症状记录时，不会生成某条症状的状态下拉框。这些情况会让侧栏及编辑入口不容易被找到。没有证据证明用户遇到的情况一定是缓存导致，因此同时检查构建部署并改进 HTML 缓存策略。

本次修改：

- 组件标题、无障碍标签、后端操作说明统一为 **Symptom log**。
- 顶栏始终提供 Symptom log（记录数）定位链接。
- 宽窗口保持右侧日志；窄窗口将日志放在聊天区前，限制高度并允许内部滚动。
- 始终显示 Show symptoms 状态筛选下拉框；每条已记录症状提供 Change status 下拉框，可设 Present、Absent、Not sure。
- 空记录明确解释如何产生记录，不伪造症状或显示可修改的虚假数据。仅询问症状定义不会自动记录该症状。
- 删除 Explain question、Summarize evidence、Ask a knowledge question 按钮、各自自动发送预设消息的事件处理、Knowledge topic 显示、对应 CSS 和前端 knowledge_focus 字段。
- 正常文本聊天仍使用统一的消息接口；未删除一般自然语言问答能力。后端旧会话中的 knowledge_focus 兼容字段不再用于这组前端快捷功能。
- HTML 入口返回 Cache-Control: no-store，构建使用新的内容哈希资源文件；部署时把不再引用的旧资源移至备份，避免混用旧版。

验证：前端 TypeScript / Vite 构建通过，后端 152 项测试通过。构建仍有已有的大 bundle 警告。交付时使用真实浏览器检查日志、状态控件与快捷功能移除，并用真实 API 检查记录修改。

使用方式：刷新 http://127.0.0.1:8000/，应直接看到 Symptom log。记录为空时显示空状态与筛选下拉框；输入个人症状描述并保存后，每条症状才有可编辑的 Change status。顶部日志链接可以直接定位该区域。

实测记录：内置浏览器约 624px 宽度下已确认日志位于聊天区前方；虚构测试 Drowsiness 生成记录后，使用状态下拉框改为 Not sure，刷新后仍为版本 2、Not sure。页面没有三个快捷按钮或 Knowledge topic。最终布局进一步压缩窄窗口头部，并在全部状态视图中隐藏没有匹配记录的空分组；可通过状态筛选查看空状态。HTTP 实测入口返回 200 和 Cache-Control: no-store。

## 后续复核：服务启动入口

复核时 8000 端口连接被拒绝；当时后台服务没有运行。界面文件与已部署修改仍在，未发现快捷按钮重新出现。旧 scripts/run.ps1 强制要求 .venv，也没有选择本地 LLM 配置。现已修复为支持已激活的 Conda 环境、显式 Python 路径和 local / offline 配置。该脚本只启动服务，不安装依赖或创建虚拟环境。

以后在 Anaconda PowerShell 中启动：

```powershell
conda activate gc1
cd E:\Study\StudyAtNUS\GC1_Project
.\scripts\run.ps1
```

也可不激活环境，显式指定已有解释器：

```powershell
cd E:\Study\StudyAtNUS\GC1_Project
.\scripts\run.ps1 -Python 'D:\Anaconda\envs\gc1\python.exe'
```

默认 local 配置使用现有 Qwen3 与 MiniLM 模型、agent 对话引擎及 random 适配器。保留手动指定的模型路径和适配器设置；不下载新模型。前台运行时保持终端打开，Ctrl+C 停止服务。`-Profile offline` 只用于不加载 LLM 的离线模式。

本次实际通过修改后的脚本启动并打开页面，确认 Symptom log、状态筛选及空记录提示存在，三个快捷按钮和 Knowledge topic 未出现。之前的 152 项后端测试和真实状态编辑检查仍对应同一套未改变的应用代码；本次额外检查的是启动脚本语法、启动行为与服务配置。
