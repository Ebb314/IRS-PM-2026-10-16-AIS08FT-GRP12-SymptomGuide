from fastapi import APIRouter, Depends, HTTPException

from app.dependencies import get_chatbot_service
from app.models.chat_models import AnswerChatRequest, ChatResponse, SessionState, StartChatRequest
from app.services.chatbot_service import ChatbotService, InvalidAnswerError
from app.services.session_service import SessionNotFoundError


router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/start", response_model=ChatResponse)
def start_chat(
    request: StartChatRequest,
    service: ChatbotService = Depends(get_chatbot_service),
) -> ChatResponse:
    return service.start(request)


@router.post("/answer", response_model=ChatResponse)
def answer_follow_up(
    request: AnswerChatRequest,
    service: ChatbotService = Depends(get_chatbot_service),
) -> ChatResponse:
    try:
        return service.answer(request)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Session not found") from exc
    except InvalidAnswerError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/session/{session_id}", response_model=SessionState)
def get_session(
    session_id: str,
    service: ChatbotService = Depends(get_chatbot_service),
) -> SessionState:
    try:
        return service.sessions.get(session_id)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Session not found") from exc
