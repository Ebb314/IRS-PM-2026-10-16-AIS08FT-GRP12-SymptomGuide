> **当前版本为全量词表版。** 默认加载 kg_nodes.csv 的全部 15,135 个节点，模拟模式为 random，当前完整步骤与两项交互修复见 [全量症状与交互修复记录](全量症状与交互修复记录.md)。旧 24 条 AI 报告与 10 节点示例仍保留为历史开发回归。
# AI 升级版：手动配置、运行与测试

适用目录：`E:\Study\StudyAtNUS\GC1_Project`。本说明优先于旧手册中的环境脚本说明。你可以完全不运行 setup.ps1、run.ps1、verify.ps1。以下命令在 **Anaconda PowerShell Prompt** 中逐行执行。

## 1. 先使用已有环境

```powershell
conda env list
conda activate gc1
cd E:\Study\StudyAtNUS\GC1_Project
python --version
node --version
python -c "import torch, transformers, sentence_transformers; print(torch.__version__, transformers.__version__, sentence_transformers.__version__); print(torch.cuda.is_available())"
```

本机实测使用 `D:\Anaconda\envs\gc1\python.exe`：Python 3.11.16、PyTorch 2.5.1+cu121、Transformers 4.57.6、Sentence Transformers 3.4.1，RTX 3060 Laptop GPU。**本次没有改装你的 Conda 环境。** Node 使用已有的 22。优先使用这个已验证环境；Python 3.12 可用于离线服务，完整 GPU 路径本次只验证了 3.11。

如果 `conda activate` 在普通 PowerShell 中不可用，改用 Anaconda PowerShell Prompt；也可显式调用 `& 'D:\Anaconda\envs\gc1\python.exe' ...`，无需全局改 PATH。

## 2. 仅在缺少依赖时安装

已有环境上面检查成功，可以跳过安装。若希望另建环境：

```powershell
conda create -n gc1-ai python=3.11 -y
conda activate gc1-ai
cd E:\Study\StudyAtNUS\GC1_Project
python -m pip install -r backend/requirements.txt
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r backend/requirements-local.txt
```

无 NVIDIA GPU 时，将 PyTorch 安装命令的索引换成 `https://download.pytorch.org/whl/cpu`。代码会自动选择 CPU，但 CPU 延迟尚未实测。已经安装了可用的 CUDA PyTorch 就不要重复安装。requirements-local.txt 固定了实测的 Transformers / Sentence Transformers 版本，基础 Web 依赖使用兼容范围；完整实测版本见 evaluation/local_environment.json。不要把旧 requirements-lock.txt 当作 GPU 依赖锁文件。

前端只需安装一次，改 package-lock.json 后才需要重新安装：

```powershell
cd E:\Study\StudyAtNUS\GC1_Project\frontend
npm.cmd ci
npm.cmd run build
cd ..
```

## 3. 模型下载：已交付的本机目录可直接使用

`models/Qwen2.5-1.5B-Instruct` 与 `models/all-MiniLM-L6-v2` 已下载并实测。它们不进入 Git。首次搬到另一台机器时，可以不用下载脚本，分别执行：

```powershell
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-1.5B-Instruct', revision='989aa7980e4cf806f80c7fef2b1adb7bc71aa306', local_dir='models/Qwen2.5-1.5B-Instruct', allow_patterns=['*.json','*.safetensors','*.txt','*.md','LICENSE'])"
python -c "from huggingface_hub import snapshot_download; snapshot_download('sentence-transformers/all-MiniLM-L6-v2', revision='1110a243fdf4706b3f48f1d95db1a4f5529b4d41', local_dir='models/all-MiniLM-L6-v2', allow_patterns=['*.json','*.safetensors','*.txt','*.md','1_Pooling/*'])"
```

第一条约 3 GB，第二条约 90 MB 模型权重，另外有配置文件。代码只从本地加载，`trust_remote_code=False`；缺失模型不会在服务启动时偷偷下载。下载源与许可证见 models 的 README 和 docs/AI技术详解与求职材料.md。想用一条命令时才运行 `python scripts/download_models.py`。

## 4. 启动完整 AI 版本

在项目根目录：

```powershell
conda activate gc1
$env:AI_PROFILE='local'
$env:NLP_ENGINE='rules'
$env:NLP_USE_TRANSFORMERS='false'
$env:REASONING_MODE='random'
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000 --no-access-log
```

打开 `http://127.0.0.1:8000`。首次访问会加载模型，可能需要几十秒。`http://127.0.0.1:8000/api/health` 应返回 `ai_profile: local`、`generator: Qwen2.5-1.5B-Instruct`。NLP_ENGINE=rules 在这里代表显式断言基线，AI_PROFILE=local 会在其上接入 MiniLM + Qwen，不代表模型被禁用。

不要使用 `--reload` 或多个 workers 做这次本机演示：每个进程可能再加载一份 GPU 模型。当前面向单用户演示，数据库事务和模型锁串行化请求。启动第二份 GPU 服务前先按 Ctrl+C 关闭前一份。

默认模型路径随项目自动计算。移动模型时可设置：

```powershell
$env:LOCAL_MODEL_PATH='E:/your-models/Qwen2.5-1.5B-Instruct'
$env:EMBEDDING_MODEL_PATH='E:/your-models/all-MiniLM-L6-v2'
```

如需完全无模型的离线回归，停止服务后设置 `$env:AI_PROFILE='offline'` 再启动。此时 NLP 规则、BM25、来源摘录和全部会话功能仍可运行，不会显示真实 LLM 调用。

## 5. 前后端分开开发

终端 A 启动上面的后端。终端 B：

```powershell
cd E:\Study\StudyAtNUS\GC1_Project\frontend
npm.cmd run dev -- --host 127.0.0.1
```

打开 Vite 输出的 5173 地址。代理把 `/api` 转发到 8000。修改 Vue 文件会热更新；修改 Python 文件后手动重启后端。完成修改再运行 `npm.cmd run build`，让后端 8000 使用新页面。

## 6. 一层一层测试

先运行不需要模型或队友的测试：

```powershell
cd E:\Study\StudyAtNUS\GC1_Project
python -m pytest backend/tests -q
python scripts/evaluate.py --check
```

模型质量评测（先关闭其他 GPU 服务，避免两份模型占用显存）：

```powershell
python scripts/evaluate_ai.py
```

它加载真实 Qwen 和 MiniLM，输出 evaluation/ai_report.json：24 条 NLU、9 条检索、6 条对话。**质量评测不会把未达满分当作程序崩溃**；必须阅读报告里的错误、回退和指标。它会覆盖当前报告，历史初版保存在 ai_report_initial.json。不要只把脚本退出码 0 当成模型全部答对。

启动后端后在另一个终端测试浏览器：

```powershell
cd E:\Study\StudyAtNUS\GC1_Project\frontend
$env:TEST_BASE_URL='http://127.0.0.1:8000'
node browser-fullkg.cjs
# 下项要求后端 AI_PROFILE=local
# 旧 browser-ai.cjs 仅适用于显式 DEMO / mock 配置
```

使用本机 Chrome 的 headless 模式，无需另下浏览器。Edge 用户可设置 `$env:BROWSER_CHANNEL='msedge'`。测试用虚构文本，最后删除自己的会话。报告和截图分别进入 evaluation 与 docs/screenshots。

## 7. 五分钟展示顺序

1. 输入 `I have a mild cough for three days.`，展示实体、断言、程度、持续时间，以及真实模型名 / GPU / token / 耗时。
2. 点击 Ask a knowledge question，展示 Qwen 回答、NHS 来源与检索排名；指出不会把这次提问误认为发热证据。
3. 点击 Explain question，解释算法选择问题与 LLM 组织语言的职责边界；若触发来源摘录回退，说明为什么拒绝不可靠生成。
4. 点击 Summarize evidence，再输入当前问题的省略回答，展示会话状态延续。
5. 修改证据状态、刷新恢复、导出 JSON。打开 evaluation/ai_report.json，介绍基线对照和失败样例。

## 8. 常见问题

- 没有 AI execution 模型信息：确认 health 中 profile 是 local；旧 run.ps1 用 .venv，不一定有 GPU 依赖，推荐以上 Conda 命令。
- 模型启动失败：确认两个 models 目录存在；不要把模型路径指向仓库名字符串；看终端错误而不是反复点击发送。
- CUDA out of memory：关闭先前启动的模型服务 / 评测进程，再启动一个进程。不要结束不属于本项目的进程。
- 回答退回规则或来源原文：属于明确展示的校验结果；并非请求没有运行。查看 trace 的 validation / fallback_reason。
- 前端等待：真实模型需要秒级时间；HTTP 超时为 180 秒，重试保留相同 request_id。超时并不证明服务端没有提交，幂等记录负责避免重复变更。
- 端口占用：给 uvicorn 改端口；浏览器测试对应修改 TEST_BASE_URL。Vite 代理默认仍指向 8000。
- 删除会话：页面 Delete & start over 删除该会话及其幂等响应记录。SQLite 文件并非加密存储；仅使用虚构输入。

旧手册中的 Docker、Ollama、队友 HTTP 接口可作扩展参考；这次真实验证路径为 Transformers 本地加载，不依赖 Ollama 或云 API。

