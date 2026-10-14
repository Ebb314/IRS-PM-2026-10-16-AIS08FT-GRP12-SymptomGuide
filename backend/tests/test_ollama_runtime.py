"""LLM_BACKEND=ollama: OllamaGenerator / OllamaEncoder speak the Ollama HTTP API (mocked here)."""
import json
from pathlib import Path

import httpx
import pytest

from app.models.chat_models import SessionState
from app.services.dialogue_ai import DialogueAI
from app.services.model_runtime import ModelFailure
from app.services.nlp_service import NLPService
from app.services.ollama_runtime import OllamaEncoder, OllamaGenerator


def chat_reply(content, sent_log=None, prompt_tokens=42, output_tokens=17):
    def handler(request):
        body = json.loads(request.content)
        if sent_log is not None:
            sent_log.append(body)
        assert request.url.path == "/api/chat"
        return httpx.Response(200, json={"message": {"role": "assistant", "content": content},
                                         "prompt_eval_count": prompt_tokens, "eval_count": output_tokens})
    return httpx.MockTransport(handler)


def test_generate_sends_deterministic_request_and_strips_thinking():
    sent = []
    gen = OllamaGenerator("qwen3:8b", transport=chat_reply("<think>hmm</think>\nPlain answer.", sent))
    text, meta = gen.generate("SYS", "USER", max_tokens=99)
    assert text == "Plain answer."
    body = sent[0]
    assert body["model"] == "qwen3:8b" and body["stream"] is False and body["think"] is False
    assert body["messages"] == [{"role": "system", "content": "SYS"}, {"role": "user", "content": "USER"}]
    assert body["options"]["num_predict"] == 99 and body["options"]["temperature"] == 0.0
    assert body["options"]["num_ctx"] >= 8192
    assert meta["model"] == "ollama:qwen3:8b" and meta["input_tokens"] == 42 and meta["output_tokens"] == 17
    assert gen.name == "ollama:qwen3:8b"


def test_chat_sampling_and_history_trimming():
    sent = []
    gen = OllamaGenerator("qwen3:8b", transport=chat_reply("ok", sent))
    history = [{"role": "system", "content": "S"}]
    for i in range(6):
        history += [{"role": "user", "content": f"old {i} " + "x" * 3000},
                    {"role": "assistant", "content": "y" * 3000}]
    history.append({"role": "user", "content": "latest"})
    _, meta = gen.chat(history, max_tokens=50, context_tokens=4096, sampling=True)
    msgs = sent[0]["messages"]
    assert msgs[0]["content"] == "S" and msgs[-1]["content"] == "latest"
    assert meta["dropped_messages"] > 0 and len(msgs) < len(history)
    assert sent[0]["options"]["temperature"] == 0.7 and sent[0]["options"]["top_k"] == 20


@pytest.mark.parametrize("handler", [
    lambda r: httpx.Response(500, text="boom"),
    lambda r: httpx.Response(200, json={"unexpected": True}),
    lambda r: (_ for _ in ()).throw(httpx.ConnectError("refused")),
])
def test_failures_raise_model_failure(handler):
    gen = OllamaGenerator("qwen3:8b", transport=httpx.MockTransport(handler))
    with pytest.raises(ModelFailure):
        gen.generate("s", "u")


def test_startup_check_reports_missing_model_and_unreachable_server():
    tags = httpx.MockTransport(lambda r: httpx.Response(200, json={"models": [{"name": "qwen3:8b", "model": "qwen3:8b"}]}))
    assert OllamaGenerator("qwen3:8b", transport=tags).check()
    with pytest.raises(ModelFailure, match="not installed"):
        OllamaGenerator("llama3:70b", transport=tags).check()
    down = httpx.MockTransport(lambda r: (_ for _ in ()).throw(httpx.ConnectError("refused")))
    with pytest.raises(ModelFailure, match="not reachable"):
        OllamaGenerator("qwen3:8b", transport=down).check()
    with pytest.raises(ValueError):
        OllamaGenerator("")


def test_encoder_normalizes_batches_and_caches(tmp_path):
    np = pytest.importorskip("numpy")
    calls = []

    def handler(request):
        body = json.loads(request.content)
        calls.append(len(body["input"]))
        return httpx.Response(200, json={"embeddings": [[3.0, 4.0] for _ in body["input"]]})

    enc = OllamaEncoder("nomic-embed-text", cache_dir=tmp_path, batch_size=2, transport=httpx.MockTransport(handler))
    vectors = enc.encode(["a", "b", "c"], normalize_embeddings=True)
    assert calls == [2, 1] and vectors.shape == (3, 2)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1)
    enc.encode(["a", "b", "c"])
    assert calls == [2, 1]  # corpus served from disk cache
    enc.encode(["query"])
    assert calls == [2, 1, 1]  # single queries are never cached


def test_knowledge_question_is_answered_by_ollama_model():
    """The screenshot case: no curated passage, so the answer must come from the model."""
    nlp = NLPService(Path(__file__).parents[1] / "app/data/kg_nodes.csv", use_models=False)
    answer = ("Abnormality of the immune system means the body's defence system does not work as "
              "expected, for example reacting too weakly or too strongly.")
    gen = OllamaGenerator("qwen3:8b", transport=chat_reply(answer))
    dialogue = DialogueAI(gen, vocabulary=nlp.nodes, mention_finder=nlp._mentions)
    text, trace = dialogue.general_knowledge(
        "What is Abnormality of the immune system", nlp.nodes[next(
            k for k, n in nlp.nodes.items() if n.kg_name == "Abnormality of the immune system")], {})
    assert text == answer
    assert trace["response_mode"] == "local-llm-general"
    assert trace["model_call"]["model"] == "ollama:qwen3:8b"
    assert "not enabled" not in text


def test_context_size_is_identical_across_calls_so_ollama_never_reloads():
    sent = []
    gen = OllamaGenerator("qwen3:8b", transport=chat_reply("ok", sent))
    gen.generate("s", "u", max_tokens=220)
    gen.chat([{"role": "system", "content": "s"}, {"role": "user", "content": "u"}], max_tokens=260, context_tokens=4096)
    gen.chat([{"role": "system", "content": "s"}, {"role": "user", "content": "u"}], max_tokens=180)
    assert len({body["options"]["num_ctx"] for body in sent}) == 1
    assert all(body["keep_alive"] == "30m" for body in sent)
