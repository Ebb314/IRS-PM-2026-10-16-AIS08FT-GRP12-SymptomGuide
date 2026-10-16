from uuid import uuid4
import pytest

def test_health_and_flow(client):
    assert client.get("/health").json()["mode"] == "mock"
    assert len(client.get("/api/vocabulary").json()) == 10
    a = client.post("/api/chat/start",json={"message":"fever"}).json()
    assert a["next_symptom"]["kg_id"] == "DEMO:breathlessness"
    b = client.post("/api/chat/message",json={"session_id":a["session_id"],"expected_version":a["version"],"request_id":uuid4().hex,"message":"cough"})
    assert b.status_code == 200
    assert client.get("/api/chat/session/"+a["session_id"]).json()["version"] == 2
    assert client.delete("/api/chat/session/"+a["session_id"]).status_code == 204
    assert client.get("/api/chat/session/"+a["session_id"]).status_code == 404

@pytest.mark.parametrize("message", ["", "   ", "x"*2001])
def test_invalid_text(client, message):
    assert client.post("/api/chat/start",json={"message":message}).status_code == 422

def test_unknown_session_and_invalid_answer(client):
    assert client.get("/api/chat/session/missing").status_code == 404
    assert client.post("/api/chat/answer",json={"answer":"maybe"}).status_code == 422

def test_sessions_isolated(client):
    a = client.post("/api/chat/start",json={"message":"fever"}).json()
    b = client.post("/api/chat/start",json={"message":"headache"}).json()
    assert a["session_id"] != b["session_id"]
    assert client.get("/api/chat/session/"+a["session_id"]).json()["positive_symptoms"] == a["positive_symptoms"]

def test_delete_removes_start_replay(client):
    payload = {"message":"fever","request_id":uuid4().hex}
    a = client.post("/api/chat/start",json=payload).json()
    client.delete("/api/chat/session/"+a["session_id"])
    b = client.post("/api/chat/start",json=payload).json()
    assert a["session_id"] != b["session_id"]

