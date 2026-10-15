from fastapi import APIRouter, Depends, HTTPException, Response
from app.dependencies import get_chatbot_service
from app.models.chat_models import StartChatRequest, AnswerChatRequest, MessageRequest, CorrectionRequest, ChatResponse
from app.services.chatbot_service import InvalidAnswerError
from app.services.session_service import SessionNotFoundError
from app.services.remote_adapters import AdapterError
from app.logging_setup import get_logger

_log = get_logger("flow")
router = APIRouter(prefix="/chat", tags=["chat"])

def invoke(method, request):
    try:
        return method(request)
    except SessionNotFoundError as exc:
        _log.warning("404 session not found")
        raise HTTPException(404, "Session not found or deleted") from exc
    except InvalidAnswerError as exc:
        _log.warning("409 %s", exc)
        raise HTTPException(409, str(exc)) from exc
    except AdapterError as exc:
        _log.error("503 adapter error: %s", exc)
        raise HTTPException(503, str(exc)) from exc
    except Exception:
        _log.exception("500 unexpected error in %s", getattr(method, "__name__", method))
        raise

@router.post("/start", response_model=ChatResponse)
def start(request: StartChatRequest, service=Depends(get_chatbot_service)):
    return invoke(service.start, request)

@router.post("/message", response_model=ChatResponse)
def message(request: MessageRequest, service=Depends(get_chatbot_service)):
    return invoke(service.message, request)

@router.post("/answer", response_model=ChatResponse)
def answer(request: AnswerChatRequest, service=Depends(get_chatbot_service)):
    return invoke(service.answer, request)

@router.post("/correct", response_model=ChatResponse)
def correct(request: CorrectionRequest, service=Depends(get_chatbot_service)):
    return invoke(service.correct, request)

@router.get("/session/{session_id}", response_model=ChatResponse)
def inspect(session_id: str, service=Depends(get_chatbot_service)):
    return invoke(service.inspect, session_id)

@router.delete("/session/{session_id}", status_code=204)
def delete(session_id: str, service=Depends(get_chatbot_service)):
    invoke(service.sessions.delete, session_id)
    return Response(status_code=204)

