"""Reusable local model. Never execute remote custom model code."""
import json,os
import threading
from time import perf_counter
from pathlib import Path

class ModelFailure(RuntimeError): pass

class LocalGenerator:
    def __init__(self,path,device="auto"):
        import torch
        from transformers import AutoTokenizer,AutoModelForCausalLM
        self.torch=torch
        self.device="cuda" if device=="auto" and torch.cuda.is_available() else ("cpu" if device=="auto" else device)
        self.tokenizer=AutoTokenizer.from_pretrained(str(path),local_files_only=True,trust_remote_code=False)
        self.quantization='none'
        use_4bit=os.getenv('LOCAL_MODEL_4BIT','true' if 'Qwen3-4B' in str(path) else 'false').lower()=='true'
        kwargs={'local_files_only':True,'trust_remote_code':False,
                'torch_dtype':torch.float16 if self.device=='cuda' else torch.float32}
        if use_4bit:
            if self.device!='cuda': raise ModelFailure('4-bit local profile requires a CUDA GPU; set LOCAL_MODEL_4BIT=false for CPU')
            from transformers import BitsAndBytesConfig
            kwargs.update(quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',
                bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16),device_map={'':0})
            self.quantization='bnb-nf4-double'
        self.model=AutoModelForCausalLM.from_pretrained(str(path),**kwargs)
        if not use_4bit: self.model=self.model.to(self.device)
        self.model.eval()
        self.lock=threading.Lock(); self.name=Path(path).name

    def generate(self,system,user,max_tokens=220):
        messages=[{"role":"system","content":system},{"role":"user","content":user}]
        return self.chat(messages,max_tokens,context_tokens=2048,sampling=False)

    def chat(self,messages,max_tokens=260,context_tokens=4096,sampling=True):
        messages=[dict(m) for m in messages]
        dropped=0
        with self.lock,self.torch.inference_mode():
            while True:
                text=self.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
                inputs=self.tokenizer(text,return_tensors="pt").to(self.device)
                if inputs['input_ids'].shape[-1]<=context_tokens: break
                if len(messages)<=2: raise ModelFailure('Current message exceeds context budget')
                del messages[1]; dropped+=1
                if len(messages)>2 and messages[1]['role']=='assistant':
                    del messages[1]; dropped+=1
            started=perf_counter()
            decoding={'do_sample':sampling}
            seed=int(os.getenv('LOCAL_GENERATION_SEED','20260916'))
            if sampling:
                self.torch.manual_seed(seed)
                decoding.update(temperature=0.7,top_p=0.8,top_k=20)
            tokens=self.model.generate(**inputs,max_new_tokens=max_tokens,**decoding,pad_token_id=self.tokenizer.eos_token_id)
            added=tokens[0,inputs["input_ids"].shape[-1]:]
            return self.tokenizer.decode(added,skip_special_tokens=True),{
                "model":self.name,"device":self.device,"quantization":self.quantization,"input_tokens":int(inputs["input_ids"].shape[-1]),
                "output_tokens":len(added),"latency_ms":round((perf_counter()-started)*1000,2),
                "context_messages":len(messages)-1,"dropped_messages":dropped,
                "sampling":sampling,"seed":seed if sampling else None}

def parse_json(text):
    text=text.strip()
    if text.startswith(chr(96)*3): text=text.split("\n",1)[-1].rsplit(chr(96)*3,1)[0].strip()
    start=text.find("{")
    if start<0: raise ModelFailure("No JSON object")
    try:
        result,end=json.JSONDecoder().raw_decode(text[start:])
        if text[start+end:].strip(): raise ModelFailure("Trailing model output")
        return result
    except (ValueError,TypeError) as exc: raise ModelFailure("Invalid JSON") from exc

