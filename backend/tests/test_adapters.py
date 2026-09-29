import httpx
import pytest
from app.services.remote_adapters import RemoteGateway, RemoteRecallService, RemoteSearchService, AdapterError
from app.services.session_service import SQLiteSessionService
from app.models.chat_models import Symptom

def test_remote_contract_and_payload():
    def handle(req):
        assert req.url.path == "/recall"
        assert b"DEMO:fever" in req.content
        return httpx.Response(200,json=[{"id":"d","name":"external","score":.5,"source":"teammate-v1","evidence_ids":["DEMO:fever"]}])
    remote = RemoteRecallService(RemoteGateway("http://test",transport=httpx.MockTransport(handle)))
    assert remote.recall([Symptom(kg_id="DEMO:fever",kg_name="fever")],[])[0].source == "teammate-v1"

@pytest.mark.parametrize("scenario", ["timeout","server_error","bad_schema","bad_json"])
def test_adapter_failure(scenario):
    def handle(req):
        if scenario == "timeout": raise httpx.ReadTimeout("timeout")
        if scenario == "server_error": return httpx.Response(500)
        if scenario == "bad_json": return httpx.Response(200,text="oops")
        return httpx.Response(200,json=[{"id":"d","name":"x","score":3}])
    remote = RemoteRecallService(RemoteGateway("http://test",transport=httpx.MockTransport(handle)))
    with pytest.raises(AdapterError):
        remote.recall([],[])

def test_sqlite_restart_persists_and_returns_copies(tmp_path):
    path = str(tmp_path/"sessions.db")
    a = SQLiteSessionService(path)
    with a.transaction():
        state = a.create(); state.version = 3; a.save(state)
    read = a.get(state.session_id); read.version = 99
    assert a.get(state.session_id).version == 3
    a.close()
    b = SQLiteSessionService(path)
    assert b.get(state.session_id).version == 3
    b.close()

