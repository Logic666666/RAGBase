import asyncio
import json
import os
from functools import lru_cache

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi import Depends
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional

from .core.config import get_settings, Settings
from .services.kb import KnowledgeBaseService
from .services.rag import RagService
from .agent.orchestrator import Agent
from .tools.registry import ToolRegistry
from .tools.builtin.knowledge_base import KnowledgeBaseTool
from .tools.builtin.codebase import GrepCodeTool, ReadFileTool, ListFilesTool
from .tools.builtin.note_take import NoteTakeTool, ReadNoteTool, ListNotesTool
from .tools.builtin.read_pdf import ReadPdfTool
from .infrastructure.llm_client import OllamaChatClient
from .observability.tracer import Tracer
from .sessions import JsonSessionStore, SessionManager
from .workspace import FileWorkspace
from .events import EventBus, InMemoryEventBus


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
    ok = kb.delete_kb(name)
    if not ok:
        raise HTTPException(
            status_code=500,
            detail=f"删除知识库 {name} 失败（可能存在文件占用或权限问题）",
        )
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


@lru_cache
def get_event_bus() -> EventBus:
    """
    事件总线单例（进程级）。
    Agent 步骤事件 → EventBus → SSE 推送给前端。
    """
    return InMemoryEventBus()


def build_agent(settings: Settings, kb: str, session_id: str, max_steps: int = 5) -> Agent:
    """
    构建 Agent 实例。

    组装逻辑：
      1. 文档检索工具：KnowledgeBaseTool（search_kb）
      2. 代码库工具：grep_code / read_file / list_files
         （对齐 Claude Code 的 Grep/Read/Glob——代码分析用精确搜索而非向量检索）
      3. 工作区笔记：note_take / read_note / list_notes
         （对齐 Claude Code 的 filesystem as memory——中间产物外置）
      4. LLMClient（推理引擎）+ Tracer（可观测）

    代码库 = 知识库的 source 目录（用户上传/导入的代码所在地）
    工作区 = data/sessions/{session_id}/（按会话隔离，每个任务自己的笔记）
    """
    rag = RagService(settings)
    codebase_dir = os.path.join(settings.data_dir, "kb", kb, "source")
    workspace = FileWorkspace(os.path.join(settings.data_dir, "sessions", session_id))

    registry = ToolRegistry()
    registry.register(KnowledgeBaseTool(rag, kb))
    registry.register(GrepCodeTool(codebase_dir))
    registry.register(ReadFileTool(codebase_dir))
    registry.register(ReadPdfTool(codebase_dir))
    registry.register(ListFilesTool(codebase_dir))
    registry.register(NoteTakeTool(workspace))
    registry.register(ReadNoteTool(workspace))
    registry.register(ListNotesTool(workspace))

    llm = OllamaChatClient(
        base_url=settings.ollama_base_url,
        model=settings.chat_model,
        temperature=0.2,
        think=settings.llm_think,
        max_tokens=settings.llm_max_tokens,
        num_ctx=settings.llm_num_ctx,
    )
    return Agent(
        llm=llm,
        tools=registry,
        tracer=Tracer(),
        max_steps=max_steps,
        event_bus=get_event_bus(),
        session_id=session_id,
    )


@lru_cache
def get_session_manager() -> SessionManager:
    """
    会话管理器依赖注入（进程级单例）。

    研究报告任务是长任务（10-30 分钟），
    采用异步执行：提交即返回 session_id，前端轮询进度。

    与 get_settings 同模式：@lru_cache + 无参，
    保证整个进程只有一个 SessionManager 实例。

    单例理由：
      SessionManager._sessions 持有后台任务的 asyncio.Task 句柄，
      （取消任务、终止运行中会话等能力依赖它）。
      若每个请求新建实例，_sessions 每次都是空的，
      这些内存态能力会全部失效。

    内部自行获取 settings（lru_cache 缓存），不通过 Depends 注入参数。
    """
    settings = get_settings()
    store = JsonSessionStore(os.path.join(settings.data_dir, "sessions"))
    return SessionManager(store=store, agent_builder=build_agent)


@app.post("/agent/run")
async def agent_run(
    body: AgentRunBody,
    manager: SessionManager = Depends(get_session_manager),
    settings: Settings = Depends(get_settings),
):
    """
    提交任务给 Agent（异步会话）。

    与 /chat 的区别：
      - /chat: 单轮检索 + 单轮生成（快速直接，同步返回）
      - /agent/run: 多步推理循环（长任务，异步执行）

    立即返回 session_id，任务后台执行：
      轮询 GET /agent/status/{session_id} 查进度
      完成后 GET /agent/result/{session_id} 取结果 + trace
    """
    session_id = await manager.submit(
        kb=body.kb, task=body.task, max_steps=body.max_steps, settings=settings,
    )
    return {"session_id": session_id, "status": "running"}


@app.get("/agent/sessions")
async def agent_sessions(
    manager: SessionManager = Depends(get_session_manager),
):
    """
    列出所有历史会话（按创建时间倒序）。

    追溯入口：用户可通过 session_id 回看任意一次研究的
    完整 trace（GET /agent/result/{session_id}）。
    """
    records = await manager.list_sessions()
    return {
        "items": [
            {
                "session_id": r.session_id,
                "status": r.status.value,
                "task": r.task[:100],       # 截断，列表轻量
                "kb": r.kb,
                "completed": r.completed,
                "steps": r.steps,
                "created_at": r.created_at,
            }
            for r in records
        ]
    }


@app.delete("/agent/sessions/{session_id}")
async def delete_session(
    session_id: str,
    manager: SessionManager = Depends(get_session_manager),
):
    """删除历史会话（记录 + 工作区数据）"""
    record = await manager.get_status(session_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"会话 {session_id} 不存在")
    await manager.delete_session(session_id)
    return {"deleted": session_id}


@app.get("/agent/events/{session_id}")
async def agent_events(
    session_id: str,
    request: Request,
    bus: EventBus = Depends(get_event_bus),
):
    """
    实时事件流（SSE）：Agent 步骤事件逐步推送。

    前端用 EventSource 订阅，增量渲染 thought/工具调用/结果。
    事件类型：plan / think / llm / tool_call / tool_result / final_answer。

    心跳：15 秒无事件发注释行，防止代理/网关空闲断连。
    """
    queue = bus.subscribe(session_id)

    async def gen():
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15)
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                except asyncio.TimeoutError:
                    # 心跳（SSE 注释行）
                    yield ": heartbeat\n\n"
        finally:
            bus.unsubscribe(session_id, queue)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/agent/status/{session_id}")
async def agent_status(
    session_id: str,
    manager: SessionManager = Depends(get_session_manager),
):
    """查询会话进度（running / done / failed）"""
    record = await manager.get_status(session_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"会话 {session_id} 不存在")

    return {
        "session_id": record.session_id,
        "status": record.status.value,
        "steps": record.steps,
        "error": record.error,
    }


@app.get("/agent/result/{session_id}")
async def agent_result(
    session_id: str,
    manager: SessionManager = Depends(get_session_manager),
):
    """获取会话最终结果（answer + trace + transcript）"""
    record = await manager.get_status(session_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"会话 {session_id} 不存在")
    if record.status.value == "running":
        raise HTTPException(status_code=409, detail="会话仍在执行中")

    return {
        "session_id": record.session_id,
        "status": record.status.value,
        "task": record.task,           # 任务原文（历史会话查看时展示用户问题）
        "answer": record.result,
        "completed": record.completed,
        "reason": record.reason,
        "steps": record.steps,
        "trace": record.trace,
        # transcript：完整消息历史（resume/追溯的数据基础）
        "messages": record.messages,
        "error": record.error,
    }


