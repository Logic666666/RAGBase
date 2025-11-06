from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi import Depends
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional

from .settings import get_settings, Settings
from .services.kb import KnowledgeBaseService
from .services.rag import RagService


app = FastAPI(title="AI RAG Knowledge", version="0.1.0")

# Static minimal UI
app.mount("/", StaticFiles(directory="static", html=True), name="static")


class CreateKbBody(BaseModel):
    name: str


class GitIngestBody(BaseModel):
    repo_url: str
    branch: Optional[str] = None
    username: Optional[str] = None
    token: Optional[str] = None


class ChatBody(BaseModel):
    kb: str
    question: str
    top_k: int = 4


def get_kb_service(settings: Settings = Depends(get_settings)):
    return KnowledgeBaseService(settings)


def get_rag_service(settings: Settings = Depends(get_settings)):
    return RagService(settings)


@app.get("/health")
def health(settings: Settings = Depends(get_settings)):
    return {
        "ok": True,
        "model": settings.deepseek_model,
        "ollama": settings.ollama_base_url,
    }


@app.post("/kb")
def create_kb(body: CreateKbBody, kb: KnowledgeBaseService = Depends(get_kb_service)):
    kb.create_kb(body.name)
    return {"created": body.name}


@app.get("/kb")
def list_kb(kb: KnowledgeBaseService = Depends(get_kb_service)):
    return {"items": kb.list_kb()}


@app.delete("/kb/{name}")
def delete_kb(name: str, kb: KnowledgeBaseService = Depends(get_kb_service)):
    kb.delete_kb(name)
    return {"deleted": name}


@app.post("/kb/{name}/upload")
async def upload_files(
    name: str,
    files: List[UploadFile] = File(...),
    kb: KnowledgeBaseService = Depends(get_kb_service),
):
    saved = await kb.save_and_ingest_files(name, files)
    return {"ingested": saved}


@app.post("/kb/{name}/git")
def ingest_git(name: str, body: GitIngestBody, kb: KnowledgeBaseService = Depends(get_kb_service)):
    count = kb.ingest_git_repo(name=name, repo_url=body.repo_url, branch=body.branch, username=body.username, token=body.token)
    return {"ingested_docs": count}


@app.post("/chat")
def chat(body: ChatBody, rag: RagService = Depends(get_rag_service)):
    answer, sources = rag.answer_question(body.kb, body.question, body.top_k)
    return {"answer": answer, "sources": sources}


