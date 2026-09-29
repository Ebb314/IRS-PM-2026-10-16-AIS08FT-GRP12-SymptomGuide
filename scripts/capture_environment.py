import json,platform,importlib.metadata
from pathlib import Path
names=['torch','transformers','sentence-transformers','huggingface-hub','numpy','fastapi','pydantic','uvicorn','pytest','httpx']
result={'python':platform.python_version(),'packages':{name:importlib.metadata.version(name) for name in names}}
Path('evaluation/local_environment.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
