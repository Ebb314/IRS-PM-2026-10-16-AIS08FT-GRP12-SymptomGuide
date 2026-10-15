from pathlib import Path
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.api.chat import router
from app.config import settings
from app.dependencies import get_chatbot_service
from app.logging_setup import setup_logging

setup_logging()

app = FastAPI(title=settings.app_name, version="2.1.0", description="Academic dialogue prototype. Mock mode is for integration testing, not clinical use.")
app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins), allow_methods=["GET","POST","DELETE"], allow_headers=["Content-Type"])
app.include_router(router, prefix=settings.api_prefix)

@app.middleware("http")
async def fresh_frontend_entry(request, call_next):
    response = await call_next(request)
    # Always refresh the HTML entry after a deployment; hashed assets can cache.
    if response.headers.get('content-type', '').startswith('text/html'):
        response.headers['Cache-Control'] = 'no-store'
    return response

@app.get("/health")
@app.get("/api/health")
def health(service=Depends(get_chatbot_service)):
    return {"status":"ok", "mode":service.mode, "nlp_engine":"conversation-agent-v1" if hasattr(service.dialogue,'understand') else service.nlp.engine, "vocabulary_size":len(service.nlp.nodes),
        "ai_profile":settings.ai_profile,"dialogue_engine":settings.dialogue_engine,
        "generator":service.dialogue.generator.name if service.dialogue.generator else None}

@app.get("/api/vocabulary")
def vocabulary(service=Depends(get_chatbot_service)):
    return [s.model_dump() for s in service.nlp.nodes.values()]

# A build can be served by the same local backend without a second process.
dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if dist.is_dir():
    app.mount("/", StaticFiles(directory=dist, html=True), name="frontend")



