# 技术栈分析文档

## 项目概述

这是一个基于 FastAPI + LangChain + Ollama 的 RAG（Retrieval-Augmented Generation）知识库系统，支持文件上传、Git 仓库导入、向量存储和智能问答功能。


## 项目整体结构

说明整体目录与关键模块的职责映射，方便对项目进行快速定位与修改：

- 根目录
  - `README.md`：项目说明、快速启动、部署与配置指南。
  - `requirements.txt`：Python 依赖清单。
  - `Dockerfile`：应用容器化构建脚本。
  - `deploy-configs/`：生产部署与 Nginx 配置。
  - `static/`：极简前端页面（挂载到 `/static`）。

- `app/`（核心后端实现）
  - `app/main.py`：FastAPI 应用入口、路由与依赖注入。
  - `app/settings.py`：配置管理（Pydantic Settings），负责初始化 `./data` 目录结构。
  - `app/services/`
    - `kb.py`：KnowledgeBaseService，负责知识库的创建、删除、文件保存与 Git 摄取逻辑。
    - `vectorstore.py`：VectorStore，Chroma 的封装：向量化、写入、检索器构建。
    - `rag.py`：RagService，检索 + 上下文组装 + 调用 Chat 模型生成回答。
  - `app/utils/`
    - `chunking.py`：文本分块策略（支持中英文、chunk_size/overlap 配置）。

- 运行时数据（由应用创建）
  - `./data/kb/<kb-name>/source/`：保存源文件（上传或 Git 克隆）。
  - `./data/vectorstore/<kb-name>/`：Chroma 持久化向量存储目录。

- 测试/示例数据
  - `data/kb/test/`、`data/vectorstore/test/`：示例或测试用的知识库数据。


## 工作流分析

下面按关键用例描述端到端工作流，包含主要函数/文件、输入输出和数据落盘位置。

1) 上传文件摄取（POST /kb/{name}/upload）

- 发起：用户通过前端或 API 上传一个或多个文件给 `/kb/{name}/upload`。
- 入口：`app/main.py` -> `KnowledgeBaseService.save_and_ingest_files`（`app/services/kb.py`）。
- 处理步骤：
  1. 调用 `create_kb(name)` 确保 `data/kb/<name>/source` 与 `data/vectorstore/<name>` 存在。
  2. 校验文件扩展名（`SUPPORTED_EXTS`），保存到 `data/kb/<name>/source/<filename>`。
  3. 从文件读取文本并调用 `chunk_texts()`（`app/utils/chunking.py`）分块。
  4. 将分块文本与 metadata 传递给 `VectorStore.add_documents(persist_dir, docs)` 写入 Chroma（`app/services/vectorstore.py`），持久化到 `data/vectorstore/<name>/`。
- 输出：返回已写入的文档块数量（ingested count）。


2) Git 仓库摄取（POST /kb/{name}/git）

- 发起：用户提供仓库 URL、可选分支/认证信息到 `/kb/{name}/git`。
- 入口：`KnowledgeBaseService.ingest_git_repo`（`app/services/kb.py`）。
- 处理步骤：
  1. 在 `data/kb/<name>/` 下创建临时目录（`git_tmp_xxx`），通过 GitPython 克隆仓库（支持 token 验证与加速策略）。
  2. 遍历仓库文件，筛选 `SUPPORTED_EXTS`，复制到 `data/kb/<name>/source/`。
  3. 读取文件、分块、并调用 `VectorStore.add_documents` 写入 `data/vectorstore/<name>/`。
- 输出：返回导入的文档块数量。


3) 问答（RAG：POST /chat）

- 发起：客户端调用 `/chat`，传入 `kb`、`question`、可选 `top_k`。
- 入口：`app/main.py` -> `RagService.answer_question`（`app/services/rag.py`）。
- 处理步骤：
  1. 通过 `VectorStore.as_retriever(persist_dir, top_k)` 加载 Chroma collection（基于 `data/vectorstore/<kb>/`）。
  2. 将问题向量化并进行相似度检索，得到 top_k 个 Document（text + metadata）。
  3. 将检索到的文档组织为 `context` 字符串（包含每个文档的 source 路径），构造 SystemMessage + HumanMessage。
  4. 调用 `ChatOllama`（使用 `settings.chat_model`）生成回答，返回回答文本与来源 snippets。
- 输出：返回 JSON 包含 `answer` 与 `sources`（每个 source 含文件路径和文本片段）。


4) 系统启动与配置

- 启动：通过 `uvicorn app.main:app` 启动服务（Dockerfile / docker-compose 中也使用该命令）。
- 配置加载：`get_settings()`（`app/settings.py`）读取 `.env` 并创建 `./data` 目录结构，设置 `ollama_base_url`、`embedding_model`、`chat_model` 等。
- 模型分离：
  - 嵌入模型（`EMBEDDING_MODEL`）用于 `OllamaEmbeddings`（`VectorStore._embeddings`）执行向量化。
  - 问答模型（`CHAT_MODEL`）用于 `ChatOllama`（`RagService._llm`）生成回答。


## 核心技术栈分析

### 1. Python 基础技术栈

**FastAPI (核心 Web 框架)**
- **文件**: [`app/main.py`](app/main.py:1)
- **用途**: 构建 RESTful API 服务
- **主要特性**: 
  - 异步支持 (`async/await`)
  - 自动 API 文档生成
  - 依赖注入系统
  - 类型提示和验证

```python
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi import Depends
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="AI RAG Knowledge", version="0.1.0")
```

**Pydantic (数据验证和序列化)**
- **文件**: [`app/main.py`](app/main.py:5), [`app/settings.py`](app/settings.py:1)
- **用途**: 数据模型定义、配置管理、请求验证
- **核心类**: `BaseModel`, `BaseSettings`

```python
# 请求数据模型定义
class CreateKbBody(BaseModel):
    name: str

class ChatBody(BaseModel):
    kb: str
    question: str
    top_k: int = 4

# 配置管理
class Settings(BaseSettings):
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "deepseek-r1:1.5b")
    chat_model: str = os.getenv("CHAT_MODEL", "deepseek-r1:1.5b")
    data_dir: str = os.getenv("DATA_DIR", "./data")

# 在路由中使用
@app.post("/kb")
def create_kb(body: CreateKbBody, kb: KnowledgeBaseService = Depends(get_kb_service)):
    kb.create_kb(body.name)
    return {"created": body.name}
```

### 2. LangChain 生态系统

**LangChain Core (LLM 应用框架)**
- **文件**: [`app/services/rag.py`](app/services/rag.py:5), [`app/services/vectorstore.py`](app/services/vectorstore.py:5)
- **用途**: 构建 LLM 应用，管理对话流程
- **核心组件**:
  - `ChatOllama`: Ollama 模型集成
  - `OllamaEmbeddings`: 文本向量化
  - `HumanMessage`, `SystemMessage`: 消息管理

```python
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
```

**LangChain Community (社区扩展)**
- **文件**: [`app/services/vectorstore.py`](app/services/vectorstore.py:6)
- **用途**: 向量数据库集成
- **核心组件**: `Chroma` 向量存储

```python
from langchain_community.vectorstores import Chroma
```

**LangChain Text Splitters (文本处理)**
- **文件**: [`app/utils/chunking.py`](app/utils/chunking.py:3)
- **用途**: 文本分块处理，支持中英文混合文本
- **核心功能**: `RecursiveCharacterTextSplitter`

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_texts(text: str) -> list[str]:
    # 针对中英文混合文本优化的分块策略
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,        # 每个文本块的最大字符数
        chunk_overlap=150,     # 相邻块之间的重叠字符数
        separators=[
            "\n\n",             # 段落分隔符
            "\n",               # 行分隔符
            "。",               # 中文句号
            "！",               # 中文感叹号
            "？",               # 中文问号
            ".",                # 英文句号
            "！",               # 英文感叹号
            "？",               # 英文问号
            " ",                # 空格
            ""                  # 字符级别
        ],  # 分割优先级顺序：段落 > 行 > 中文句子 > 英文句子 > 单词 > 字符
    )
    return splitter.split_text(text)
```

### 3. 向量数据库技术

**ChromaDB (向量存储和检索)**
- **文件**: [`app/services/vectorstore.py`](app/services/vectorstore.py:6)
- **用途**: 文档向量化存储和相似性搜索
- **核心功能**:
  - 文本向量化存储
  - 相似性检索
  - 持久化存储
  - 集合名称安全处理

```python
def add_documents(self, persist_dir: str, docs: List[Tuple[str, dict]]):
    # 分离文本内容和元数据
    texts = [t for t, _ in docs]
    metas = [m for _, m in docs]
    
    # 创建安全的集合名称（符合ChromaDB规范）
    safe_collection_name = self._get_safe_collection_name(persist_dir)
    
    # 创建Chroma向量存储实例
    vs = Chroma(
        collection_name=safe_collection_name,
        embedding_function=self._embeddings(),  # Ollama嵌入模型
        persist_directory=persist_dir
    )
    
    # 添加文本到向量存储（自动向量化）
    vs.add_texts(texts=texts, metadatas=metas)
    vs.persist()

def as_retriever(self, persist_dir: str, top_k: int):
    # 加载已存在的向量存储
    safe_collection_name = self._get_safe_collection_name(persist_dir)
    vs = Chroma(
        collection_name=safe_collection_name,
        embedding_function=self._embeddings(),
        persist_directory=persist_dir
    )
    
    # 配置检索器，设置返回结果数量
    return vs.as_retriever(search_kwargs={"k": top_k})
```

### 4. 大模型 API 集成

**Ollama 集成 (本地大模型服务)**
- **文件**: [`app/services/rag.py`](app/services/rag.py:5), [`app/services/vectorstore.py`](app/services/vectorstore.py:7)
- **用途**: 本地部署大语言模型服务
- **支持的模型**:
  - DeepSeek 系列 (`deepseek-r1:1.5b`, `deepseek-r1:7b`)
  - 可独立配置嵌入模型和对话模型

```python
# 嵌入模型配置
model_name = self.settings.embedding_model or self.settings.deepseek_model
return OllamaEmbeddings(
    base_url=self.settings.ollama_base_url,
    model=model_name,
)

# 对话模型配置
model_name = self.settings.chat_model or self.settings.deepseek_model
return ChatOllama(
    base_url=self.settings.ollama_base_url,
    model=model_name,
    temperature=0.2,
)
```

### 5. RAG (Retrieval-Augmented Generation) 实现

**RAG 服务架构**
- **文件**: [`app/services/rag.py`](app/services/rag.py:14)
- **核心流程**:
  1. 向量检索: 将用户问题向量化，搜索相似文档
  2. 上下文构建: 格式化检索到的文档
  3. 问答生成: 结合上下文生成回答

```python
SYS_PROMPT = (
    "You are a helpful assistant. Use the provided context to answer the question. "
    "Cite sources as file paths if relevant. If the answer is not in the context, say you don't know."
)

def answer_question(self, kb: str, question: str, top_k: int = 4) -> tuple[str, List[dict]]:
    """
    执行RAG问答检索的完整流程
    """
    # 步骤1：创建向量检索器，指定知识库和返回数量
    retriever = self.vs.as_retriever(self._kb_vector_dir(kb), top_k)
    
    # 步骤2：执行向量相似度搜索
    # 原理：将问题文本向量化，计算与存储向量的相似度
    docs = retriever.invoke(question)
    
    # 步骤3：构建格式化的上下文
    context = "\n\n".join([
        f"[{i+1}] ({d.metadata.get('source','')})\n{d.page_content}"
        for i, d in enumerate(docs)
    ])

    # 步骤4：构建消息并生成回答
    messages = [
        SystemMessage(content=SYS_PROMPT),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}"),
    ]

    # 调用LLM生成回答
    resp = self._llm().invoke(messages)
    answer = resp.content if hasattr(resp, "content") else str(resp)
    
    # 整理来源信息
    sources = [{"source": d.metadata.get("source", ""), "snippet": d.page_content[:300]} for d in docs]
    return answer, sources
```

### 6. 数据处理技术

**文件处理和 Git 集成**
- **文件**: [`app/services/kb.py`](app/services/kb.py:1)
- **支持的文件类型**: `.txt`, `.md`, `.py`, `.java`, `.sql`, `.json`, `.csv`
- **Git 集成**: 支持 HTTPS 和私有仓库（Token 认证）
- **文本分块**: 智能中英文混合文本分割

```python
SUPPORTED_EXTS = {
    ".txt", ".md", ".py", ".java", ".sql", ".json", ".csv"
}

# Git 仓库摄取功能
async def save_and_ingest_files(self, name: str, files: List[UploadFile]) -> int:
    """处理上传的文件"""
    saved_files = []
    for file in files:
        # 验证文件扩展名
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in SUPPORTED_EXTS:
            continue
            
        # 保存文件
        content = await file.read()
        file_path = os.path.join(self.kb_source_dir(name), file.filename)
        with open(file_path, "wb") as f:
            f.write(content)
        saved_files.append(file_path)
    
    # 文本分块和向量化
    if saved_files:
        docs = self._collect_docs(saved_files)
        self.vs.add_documents(self.kb_vector_dir(name), docs)
    
    return len(saved_files)

def ingest_git_repo(self, name: str, repo_url: str, branch: str | None, username: str | None, token: str | None) -> int:
    """摄取Git仓库"""
    # 克隆仓库
    target_dir = os.path.join(self.kb_source_dir(name), "git_repo")
    
    # 处理认证
    if username and token:
        repo_url = repo_url.replace("https://", f"https://{username}:{token}@")
    
    # 调用GitPython克隆仓库
    # ... Git操作代码
    
    # 收集和处理文档
    docs = self._collect_docs([target_dir])
    self.vs.add_documents(self.kb_vector_dir(name), docs)
    
    return len(docs)
```

### 7. Uvicorn ASGI服务器

**Uvicorn与FastAPI协作**
- **依赖**: [`requirements.txt`](requirements.txt:3) 中的 `uvicorn[standard]>=0.30.0`
- **启动命令**: `uvicorn app.main:app --host 0.0.0.0 --port 8090 --reload`
- **协作关系**:
  - Uvicorn处理底层网络通信和HTTP协议
  - FastAPI处理业务逻辑和路由分发

```python
# 典型的启动配置
# development
uvicorn app.main:app --host 0.0.0.0 --port 8090 --reload

# production
uvicorn app.main:app --host 0.0.0.0 --port 8090 --workers 4

# with custom log config
uvicorn app.main:app --host 0.0.0.0 --port 8090 --log-config uvicorn_config.json
```

**Uvicorn在Docker中的应用**
```dockerfile
# Dockerfile中的配置
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8090"]
```

### 8. 部署和容器化技术

**Docker 容器化**
- **文件**: [`Dockerfile`](Dockerfile:1), [`docker-compose.yml`](deploy-configs/docker-compose.yml:1)
- **用途**: 应用容器化部署
- **特性**:
  - 多阶段构建
  - Ollama 服务集成
  - Nginx 反向代理
  - SSL 证书支持

**Dockerfile关键配置**
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# 复制应用代码
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 使用Uvicorn启动
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8090"]
```

**Nginx 反向代理**
- **文件**: [`deploy-configs/nginx/nginx.conf`](deploy-configs/nginx/nginx.conf:1)
- **用途**: 负载均衡、SSL 终端、静态文件服务
- **关键配置**:
```nginx
server {
    listen 80;
    server_name localhost;
    
    location / {
        proxy_pass http://app:8090;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
    
    location /static {
        alias /app/static;
    }
}
```

### 9. 配置管理

**环境变量配置**
- **文件**: [`app/settings.py`](app/settings.py:1)
- **特性**:
  - Pydantic Settings 管理
  - 环境变量自动加载
  - 类型安全和验证
  - 默认值和后备配置

```python
class Settings(BaseSettings):
    # Ollama服务配置
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    # 嵌入模型配置（用于文本向量化）
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "deepseek-r1:1.5b")
    
    # 问答模型配置（用于对话生成）
    chat_model: str = os.getenv("CHAT_MODEL", "deepseek-r1:1.5b")
    
    # 数据存储目录
    data_dir: str = os.getenv("DATA_DIR", "./data")
    
    # Git操作超时配置（单位：秒）
    git_timeout: int = int(os.getenv("GIT_TIMEOUT", "300"))  # 5分钟
    git_connect_timeout: int = int(os.getenv("GIT_CONNECT_TIMEOUT", "30"))  # 30秒
    
    # Git加速器配置
    git_accelerator_enabled: bool = os.getenv("GIT_ACCELERATOR_ENABLED", "true").lower() == "true"
    git_accelerator_priority: str = os.getenv("GIT_ACCELERATOR_PRIORITY", "ghproxy,fastgit,original")

    class Config:
        env_file = ".env"  # 支持.env文件

# 单例模式，确保配置只加载一次
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    # 自动创建必要的目录
    try:
        os.makedirs(settings.data_dir, exist_ok=True)
        os.makedirs(os.path.join(settings.data_dir, "vectorstore"), exist_ok=True)
        os.makedirs(os.path.join(settings.data_dir, "kb"), exist_ok=True)
    except PermissionError:
        import logging
        logging.warning(f"Permission denied when creating directories.")
    return settings
```

### 10. API接口设计

**RESTful API设计**
- **文件**: [`app/main.py`](app/main.py:20)
- **核心端点**:

```python
# 健康检查
@app.get("/health")
def health(settings: Settings = Depends(get_settings)):
    return {
        "ok": True,
        "model": settings.chat_model,
        "ollama": settings.ollama_base_url,
    }

# 知识库管理
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

# 文件上传
@app.post("/kb/{name}/upload")
async def upload_files(
    name: str,
    files: List[UploadFile] = File(...),
    kb: KnowledgeBaseService = Depends(get_kb_service),
):
    saved = await kb.save_and_ingest_files(name, files)
    return {"ingested": saved}

# Git仓库导入
@app.post("/kb/{name}/git")
def ingest_git(name: str, body: GitIngestBody, kb: KnowledgeBaseService = Depends(get_kb_service)):
    count = kb.ingest_git_repo(
        name=name,
        repo_url=body.repo_url,
        branch=body.branch,
        username=body.username,
        token=body.token
    )
    return {"ingested_docs": count}

# RAG问答
@app.post("/chat")
def chat(body: ChatBody, rag: RagService = Depends(get_rag_service)):
    answer, sources = rag.answer_question(body.kb, body.question, body.top_k)
    return {"answer": answer, "sources": sources}
```

### 数据流架构

```
用户请求 → FastAPI路由 → 服务层(LangChain) → 向量存储(Chroma) → Ollama模型
    ↑           ↑              ↑                ↑              ↑
  REST API   依赖注入      RAG逻辑处理      相似性搜索      文本生成
```

## 技术栈总结

### 满足要求的技术栈

✅ **Python 数据处理**: 
- 基本数据处理通过文件解析和文本处理实现
- 支持多种文件格式（CSV, JSON, TXT, MD 等）

✅ **大模型 API 调用**:
- 集成 Ollama 服务，支持 DeepSeek 模型
- 可配置嵌入模型和对话模型

✅ **RAG/向量检索**:
- 完整的 RAG 实现，包含向量存储、相似性搜索、上下文构建
- 使用 ChromaDB 作为向量数据库

✅ **LangChain 生态**:
- LangChain Core: 对话管理和消息处理
- LangChain Community: 向量数据库集成
- LangChain Text Splitters: 智能文本分块

✅ **FastAPI**:
- 完整的 RESTful API 实现
- 异步支持、依赖注入、自动文档生成

### 技术架构优势

1. **模块化设计**: 服务分层清晰，易于维护和扩展
2. **模型分离**: 嵌入模型和对话模型可独立配置
3. **本地部署**: 基于 Ollama，支持完全本地化部署
4. **生产就绪**: 包含完整的部署方案、SSL 支持、监控等
5. **多语言支持**: 优化的中英文混合文本处理
6. **扩展性强**: 支持多种文件格式和 Git 仓库导入

### 9. Uvicorn ASGI服务器

**Uvicorn与FastAPI协作**
- **依赖**: [`requirements.txt`](requirements.txt:3) 中的 `uvicorn[standard]>=0.30.0`
- **启动命令**: `uvicorn app.main:app --host 0.0.0.0 --port 8090 --reload`
- **协作关系**:
  - Uvicorn处理底层网络通信和HTTP协议
  - FastAPI处理业务逻辑和路由分发

```python
# 典型的启动配置
# development
uvicorn app.main:app --host 0.0.0.0 --port 8090 --reload

# production
uvicorn app.main:app --host 0.0.0.0 --port 8090 --workers 4

# with custom log config
uvicorn app.main:app --host 0.0.0.0 --port 8090 --log-config uvicorn_config.json
```

**Uvicorn在Docker中的应用**
```dockerfile
# Dockerfile中的配置
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8090"]
```

### 部署和运行方案

**本地开发运行**
```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动Uvicorn服务器
uvicorn app.main:app --host 0.0.0.0 --port 8090 --reload

# 3. 访问API文档
curl http://localhost:8090/docs
```

**Docker容器化部署**
```bash
# 构建镜像
docker build -t ai-rag-app .

# 运行容器
docker run -d -p 8090:8090 ai-rag-app

# 或使用docker-compose
docker compose up -d
```

**生产环境部署**
```bash
# 使用gunicorn+uvicorn workers
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8090

# 或使用systemd服务
sudo systemctl start ai-rag-service
```

该技术栈实现了一个功能完整、生产就绪的 RAG 知识库系统，满足了用户的所有技术要求。