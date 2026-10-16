"""Download a pinned official model snapshot; no remote Python code is executed."""
from pathlib import Path
import argparse,json
from huggingface_hub import snapshot_download

parser=argparse.ArgumentParser()
parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
args=parser.parse_args()
revision='cdbee75f17c01a7cc42f958dc650907174af0554'
target=args.root/'models/Qwen3-4B-Instruct-2507'
snapshot_download('Qwen/Qwen3-4B-Instruct-2507',revision=revision,local_dir=str(target),
    allow_patterns=['*.json','*.safetensors','*.txt','*.md','LICENSE'],max_workers=3)
(target/'download-receipt.json').write_text(json.dumps({'repository':'Qwen/Qwen3-4B-Instruct-2507','revision':revision},indent=2))
print('Completed:',target)
