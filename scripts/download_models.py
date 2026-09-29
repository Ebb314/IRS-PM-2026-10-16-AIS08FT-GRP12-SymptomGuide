"""Optional one-time download. Equivalent individual commands are in docs/AI升级运行手册.md."""
from huggingface_hub import snapshot_download
from pathlib import Path
root=Path(__file__).resolve().parents[1]
snapshot_download('Qwen/Qwen2.5-1.5B-Instruct',revision='989aa7980e4cf806f80c7fef2b1adb7bc71aa306',local_dir=str(root/'models/Qwen2.5-1.5B-Instruct'),allow_patterns=['*.json','*.safetensors','*.txt','*.md','LICENSE'])
snapshot_download('sentence-transformers/all-MiniLM-L6-v2',revision='1110a243fdf4706b3f48f1d95db1a4f5529b4d41',local_dir=str(root/'models/all-MiniLM-L6-v2'),allow_patterns=['*.json','*.safetensors','*.txt','*.md','1_Pooling/*'])
