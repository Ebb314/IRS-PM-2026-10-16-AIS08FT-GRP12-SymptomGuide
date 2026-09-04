from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_and_chat_api_flow():
    assert client.get("/health").json() == {"status": "ok"}

    started = client.post("/api/chat/start", json={"message": "I have fever and cough."})
    assert started.status_code == 200
    body = started.json()
    assert body["status"] == "RUNNING"
    assert body["next_symptom"]["kg_id"] == "P006"

    inspected = client.get("/api/chat/session/{}".format(body["session_id"]))
    assert inspected.status_code == 200
    assert len(inspected.json()["positive_symptoms"]) == 2


def test_unknown_session_returns_404():
    response = client.get("/api/chat/session/does-not-exist")
    assert response.status_code == 404
