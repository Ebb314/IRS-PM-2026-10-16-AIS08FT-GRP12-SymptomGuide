"""Ollama-served model behind the same interface as model_runtime.LocalGenerator.

Selected with AI_PROFILE=local and LLM_BACKEND=ollama. The dialogue code only uses
generate(), chat() and .name, so no dialogue logic changes. Requires no PyTorch.

OllamaEncoder is optional (OLLAMA_EMBED_MODEL). Without it, retrieval falls back
to BM25 keyword search; with it, the 15k concept embeddings are cached on disk.
"""
import hashlib
import os
import re
from pathlib import Path
from time import perf_counter

import httpx

from app.services.model_runtime import ModelFailure

THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.S)


class OllamaGenerator:
    def __init__(self, model, base_url="http://127.0.0.1:11434", timeout=180.0, think=False, transport=None,
                 num_ctx=None, keep_alive=None):
        if not model:
            raise ValueError("Set OLLAMA_MODEL to an installed Ollama model, e.g. qwen3:8b (see `ollama list`).")
        self.model, self.think, self.name = model, think, f"ollama:{model}"
        self.num_ctx = int(num_ctx or os.getenv("OLLAMA_NUM_CTX", "8192"))
        self.keep_alive = keep_alive or os.getenv("OLLAMA_KEEP_ALIVE", "30m")  # stay loaded between turns
        self.client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout, transport=transport, trust_env=False)

    def check(self):
        """Fail at startup with a clear message instead of on the first chat turn."""
        try:
            tags = self.client.get("/api/tags").json().get("models", [])
        except (httpx.HTTPError, ValueError) as exc:
            raise ModelFailure(f"Ollama is not reachable at {self.client.base_url}. Start Ollama first.") from exc
        names = {m.get("name") for m in tags} | {m.get("model") for m in tags}
        if self.model not in names and f"{self.model}:latest" not in names:
            raise ModelFailure(f"Ollama model '{self.model}' is not installed. Installed: {sorted(n for n in names if n)}")
        return self

    def generate(self, system, user, max_tokens=220):
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return self.chat(messages, max_tokens, context_tokens=2048, sampling=False)

    def chat(self, messages, max_tokens=260, context_tokens=4096, sampling=True):
        messages = [dict(m) for m in messages]
        dropped = 0
        # Same history policy as LocalGenerator: keep the system prompt and the latest
        # turn, drop the oldest turns first. Tokens are estimated (~4 chars each), so a
        # single long prompt is never rejected here; the fixed num_ctx (8192) leaves
        # ample headroom so Ollama does not silently truncate the prompt.
        while len(messages) > 2 and sum(len(m["content"]) for m in messages) / 4 > context_tokens:
            del messages[1]; dropped += 1
            if len(messages) > 2 and messages[1]["role"] == "assistant":
                del messages[1]; dropped += 1
        seed = int(os.getenv("LOCAL_GENERATION_SEED", "20260916"))
        # num_ctx must be identical on every call: Ollama reloads the model whenever it changes.
        options = {"num_predict": max_tokens, "num_ctx": self.num_ctx,
                   "seed": seed, "temperature": 0.7 if sampling else 0.0}
        if sampling:
            options.update(top_p=0.8, top_k=20)
        payload = {"model": self.model, "messages": messages, "stream": False, "options": options,
                   "keep_alive": self.keep_alive}
        if not self.think:
            payload["think"] = False  # Qwen3: answer directly; the project parses plain JSON/text
        started = perf_counter()
        try:
            response = self.client.post("/api/chat", json=payload)
            response.raise_for_status()
            body = response.json()
            text = body["message"]["content"]
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            raise ModelFailure(f"Ollama request failed: {exc}") from exc
        text = THINK_BLOCK.sub("", text).strip()
        return text, {
            "model": self.name, "device": "ollama", "quantization": "ollama-managed",
            "input_tokens": int(body.get("prompt_eval_count", 0)), "output_tokens": int(body.get("eval_count", 0)),
            "latency_ms": round((perf_counter() - started) * 1000, 2),
            "context_messages": len(messages) - 1, "dropped_messages": dropped,
            "sampling": sampling, "seed": seed if sampling else None,
        }


class OllamaEncoder:
    """Duck-types SentenceTransformer.encode(texts, normalize_embeddings=True) for HybridIndex."""

    def __init__(self, model, base_url="http://127.0.0.1:11434", timeout=180.0, cache_dir=None,
                 batch_size=256, transport=None):
        import numpy  # noqa: F401  (fail early: dense retrieval needs numpy)
        self.model, self.batch_size = model, batch_size
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout, transport=transport, trust_env=False)

    def _embed(self, texts):
        import numpy as np
        out = []
        for i in range(0, len(texts), self.batch_size):
            try:
                response = self.client.post("/api/embed", json={"model": self.model, "input": texts[i:i + self.batch_size]})
                response.raise_for_status()
                out.extend(response.json()["embeddings"])
            except (httpx.HTTPError, ValueError, KeyError) as exc:
                raise ModelFailure(f"Ollama embedding failed ({self.model}): {exc}") from exc
        return np.asarray(out, dtype="float32")

    def encode(self, texts, normalize_embeddings=True):
        import numpy as np
        texts = list(texts)
        cache = None
        if self.cache_dir and len(texts) > 1:  # cache corpus builds, not single queries
            key = hashlib.sha256((self.model + "\x00" + "\x00".join(texts)).encode()).hexdigest()[:16]
            cache = self.cache_dir / f"{re.sub(r'[^A-Za-z0-9_.-]', '_', self.model)}-{key}.npy"
            if cache.exists():
                return np.load(cache)
        vectors = self._embed(texts)
        if normalize_embeddings:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / np.where(norms == 0, 1, norms)
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache, vectors)
        return vectors
