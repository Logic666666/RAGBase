from fastapi import FastAPI, UploadFile, File
from fastapi import Depends
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional

from .core.config import get_settings, Settings
from .services.kb import KnowledgeBaseService
from .services.rag import RagService
from .agent.orchestrator import Agent
from .tools.registry import ToolRegistry
from .tools.builtin.knowledge_base import KnowledgeBaseTool
from .infrastructure.llm_client import OllamaChatClient
from .observability.tracer import Tracer


app = FastAPI(title="AI RAG Knowledge", version="0.1.0")

# Static minimal UI
app.mount("/static", StaticFiles(directory="static", html=True), name="static")


@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
    <html>
        <head>
            <title>AI RAG Knowledge</title>
            <meta http-equiv="refresh" content="0; url=/static/index.html">
        </head>
        <body>
            <p>Redirecting to <a href="/static/index.html">AI RAG Knowledge Interface</a></p>
        </body>
    </html>
    """


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


class AgentRunBody(BaseModel):
    kb: str
    task: str
    max_steps: int = 5


def get_kb_service(settings: Settings = Depends(get_settings)):
    return KnowledgeBaseService(settings)


def get_rag_service(settings: Settings = Depends(get_settings)):
    return RagService(settings)


@app.get("/health")
def health(settings: Settings = Depends(get_settings)):
    return {
        "ok": True,
        "model": settings.chat_model,
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
async def ingest_git(name: str, body: GitIngestBody, kb: KnowledgeBaseService = Depends(get_kb_service)):
    count = await kb.ingest_git_repo(name=name, repo_url=body.repo_url, branch=body.branch, username=body.username, token=body.token)
    return {"ingested_docs": count}


@app.post("/chat")
async def chat(body: ChatBody, rag: RagService = Depends(get_rag_service)):
    answer, sources = await rag.answer_question(body.kb, body.question, body.top_k)
    return {"answer": answer, "sources": sources}


def build_agent(settings: Settings, kb: str) -> Agent:
    """
    构建 Agent 实例。

    组装逻辑：
      1. RagService（共享检索）
      2. KnowledgeBaseTool（把检索包装为工具，固定知识库）
      3. ToolRegistry（注册工具）
      4. LLMClient（推理引擎）
      5. Tracer（可观测）
    """
    rag = RagService(settings)
    registry = ToolRegistry()
    registry.register(KnowledgeBaseTool(rag, kb))

    llm = OllamaChatClient(
        base_url=settings.ollama_base_url,
        model=settings.chat_model,
        temperature=0.2,
        think=settings.llm_think,
        max_tokens=settings.llm_max_tokens,
    )
    return Agent(llm=llm, tools=registry, tracer=Tracer())


@app.post("/agent/run")
async def agent_run(body: AgentRunBody, settings: Settings = Depends(get_settings)):
    """
    提交任务给 Agent。

    与 /chat 的区别：
      - /chat: 单轮检索 + 单轮生成（快速直接）
      - /agent/run: 多步推理循环，Agent 自主决定检索次数和策略
        返回完整执行 Trace（可复盘）
    """
    agent = build_agent(settings, body.kb)
    result = await agent.run(body.task)

    return {
        "answer": result.answer,
        "completed": result.completed,
        "steps": result.steps,
        "trace": agent.tracer.summary() if agent.tracer else None,
    }


