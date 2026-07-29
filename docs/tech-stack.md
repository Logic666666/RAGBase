# 技术栈分析文档

## 项目概述

基于 FastAPI + ChromaDB + Ollama 的 RAG（Retrieval-Augmented Generation）知识库系统。
所有基础设施组件（LLM 客户端、Embedding 客户端、向量存储、文本分块）均为手搓实现，
不依赖 LangChain 等第三方 agent/LLM 框架。

---

## 架构设计

### 分层架构

```
┌───────────────────────────────────────────────┐
│              API 层（app/api/）                  │
│   FastAPI 路由 · Pydantic Schema · 依赖注入     │
├───────────────────────────────────────────────┤
│           业务服务层（app/services/）             │
│   知识库管理（kb.py）· RAG 管线（rag.py）         │
├───────────────────┬───────────────────────────┤
│  Agent 核心       │  基础设施层                  │
│  （agent/）       │  （infrastructure/）         │
│                   │                             │
│  · Agent Loop     │  · llm_client.py （手搓）    │
│  · 记忆系统       │  · embeddings.py （手搓）     │
│  · 规划模块       │  · vector_store.py          │
│                   │  · text_splitter.py         │
│  工具系统(tools/) │                             │
│   · 工具注册中心  │                             │
│   · 内置工具集    │                             │
└───────────────────┴───────────────────────────┘
```

### 关键设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| LLM 框架 | 自实现，不依赖 LangChain | 完全控制调用流程，理解底层协议，方便后续扩展 tool calling |
| Embedding 调用 | 客户端主动发送 HTTP 请求 | 非黑盒，每一环可观测、可缓存、可测试 |
| 异步支持 | 全 async（FastAPI + httpx） | LLM 调用为长 I/O 等待，异步不阻塞；为后续 streaming 打基础 |
| 向量数据库 | ChromaDB 原生 SDK | 避免通过 langchain-community 封装，直接操作集合 |

---

## 基础设施层详解（app/infrastructure/）

基础设施层是项目与外部系统交互的桥梁。四个组件全部自实现，不依赖 LLM 框架。

### 1. LLM 客户端（llm_client.py）

手搓的 Ollama Chat API 客户端，直接通过 HTTTP 调用 `/api/chat` 接口。

**核心接口：**

```python
class OllamaChatClient:
    async def chat(self, messages: list[Message]) -> str:
        """发送消息列表，获取 LLM 回复文本"""
        
    async def chat_with_response(self, messages: list[Message]) -> ChatResponse:
        """获取完整响应对象（为后续 tool calling 预留）"""
```

**数据模型：**

```python
@dataclass
class Message:
    role: str      # "system" | "user" | "assistant" | "tool"
    content: str

@dataclass
class ChatResponse:
    content: str   # LLM 回复文本
    done: bool     # 是否完成
```

**Ollama API 交互格式：**

```
POST /api/chat
{
    "model": "deepseek-r1:1.5b",
    "messages": [{"role": "user", "content": "你好"}],
    "stream": false,
    "options": {"temperature": 0.2}
}

响应：
{
    "message": {"role": "assistant", "content": "你好！"},
    "done": true
}
```

**技术要点：**
- 使用 `httpx.AsyncClient` 实现非阻塞 HTTP 请求
- 构造时一次性传入 `base_url` 和 `model`，后续调用只需传消息
- `chat_with_response` 为后续 tool calling（Phase 2）预留扩展点

---

### 2. Embedding 客户端（embeddings.py）

手搓的 Ollama Embedding API 客户端，将文本转换为向量表示。

**核心接口：**

```python
class OllamaEmbeddingClient:
    async def embed_query(self, text: str) -> list[float]:
        """单段文本 → 向量"""
    
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量文本 → 向量列表"""
```

**Ollama API 交互格式：**

```
POST /api/embeddings
{
    "model": "bge-m3",
    "prompt": "要向量化的文本"
}

响应：
{
    "embedding": [0.012, 0.345, ..., 0.789],
    "prompt": "要向量化的文本"
}
```

**设计原则（客户端主动调用而非委托给向量库）：**
- Embedding 过程透明可控——每一步都可观测、可异常处理
- 可添加缓存层（相同文本避免重复计算）
- 可替换 embedding 提供方（Ollama → 其他 API）不影响向量存储层

---

### 3. 向量存储（vector_store.py）

基于 ChromaDB 原生 SDK 的向量存储与相似性检索服务。

**核心接口：**

```python
class VectorStore:
    async def add_documents(
        self, persist_dir: str, docs: list[tuple[str, dict]]
    ) -> None:
        """添加文档：文本 → embedding → ChromaDB 存储"""
    
    async def similarity_search(
        self, persist_dir: str, query: str, top_k: int = 4
    ) -> list[tuple[str, dict, float]]:
        """相似度搜索：返回 (text, metadata, distance) 三元组"""
    
    def delete_collection(self, persist_dir: str) -> None:
        """删除向量集合（对应删除知识库的清理）"""
```

**内部流程（add_documents）：**

```
文本 + 元数据
    → OllamaEmbeddingClient.embed_documents()  → 向量
    → ChromaDB persistent client  → 持久化到磁盘
```

**内部流程（similarity_search）：**

```
查询文本
    → OllamaEmbeddingClient.embed_query()  → 查询向量
    → ChromaDB collection.query()  → 相似文档
    → 返回 (原文, 元数据, 距离) 三元组
```

**技术要点：**
- Embedding 不在 ChromaDB 侧自动调用，而是客户端算好再传入——每个环节可见可控
- 使用 ChromaDB 的 `PersistentClient`，数据持久化到磁盘 `data/vectorstore/<kb_name>/`
- 集合名称通过哈希函数确保符合 ChromaDB 命名规范（非 ASCII 字符处理）
- 使用余弦距离（cosine）作为相似度度量

---

### 4. 文本分块（text_splitter.py）

递归字符文本分割器的手搓实现，将长文档切分为适合向量检索的短块。

**核心接口：**

```python
def split_text(
    text: str,
    separators: Optional[list[str]] = None,
    chunk_size: int = 1200,
    chunk_overlap: int = 150,
) -> list[str]:
    """将文本分割成块"""
```

**分块策略（优先级从高到低）：**

```
段落 (\n\n) → 行 (\n) → 中文句号(。！？) → 英文句号(.!?) → 空格 → 字符级
```

**两阶段处理：**

1. **递归切分**（`_split_by_separators`）：从高优先级分隔符开始尝试切分，切出来的块如果仍然过大则降级到下一级分隔符递归处理，直到所有块 ≤ `chunk_size`
2. **合并 + 重叠**（`_merge_and_overlap`）：将过小的相邻块合并回接近 `chunk_size`，然后在相邻块之间添加 `chunk_overlap` 字符的重叠文本。overlap 确保一个语义单元在块边界被切断时，其前半部分仍作为上下文出现在下一块的开头

**技术要点：**
- 纯函数实现，无外部依赖（不同于 LangChain 的 `RecursiveCharacterTextSplitter`）
- 分隔符列表可自定义（非中文文本可调整）
- 中英文混合文本优化：中文标点和英文标点都在分隔符列表中

---

## 业务服务层（app/services/）

### RAG 管线（rag.py）

将基础设施组件编排为完整的检索增强生成管线。

**核心流程：**

```
用户提问
    → VectorStore.similarity_search()  → 检索相关文档
    → 构建格式化上下文（含来源引用）
    → OllamaChatClient.chat()  → 生成回答
    → 返回 (answer, sources)
```

**核心代码：**

```python
async def answer_question(
    self, kb: str, question: str, top_k: int = 4
) -> tuple[str, list[dict]]:
    results = await self.vs.similarity_search(vs_dir, question, top_k)
    context = build_context(results)
    answer = await self.llm.chat([
        Message(role="system", content=SYS_PROMPT),
        Message(role="user", content=f"Context:\n{context}\n\nQuestion: {question}"),
    ])
    sources = [{"source": meta["source"], "snippet": text[:300]} for text, meta, _ in results]
    return answer, sources
```

---

### 知识库服务（kb.py）

管理知识库的创建、文件上传、Git 仓库导入和向量索引构建。

**核心流程（文件上传）：**

```
上传文件
    → 校验文件类型（SUPPORTED_EXTS）
    → 保存到 data/kb/<name>/source/
    → 读取内容 → split_text() 分块
    → VectorStore.add_documents()  → 向量化并存储
```

**核心流程（Git 导入）：**

```
Git 仓库 URL
    → 支持 Token 认证和 GitHub 加速代理
    → 使用 GitPython 克隆到临时目录
    → 筛选支持的文件类型
    → 复制到 data/kb/<name>/source/
    → 分块 → 向量化并存储
    → 清理临时目录
```

---

## 测试体系

采用分层测试策略，共 26 项自动化测试：

| 层级 | 测试对象 | 依赖 | 数量 |
|------|---------|------|------|
| 单元测试 | text_splitter（纯算法） | 无 | 9 |
| 单元测试 | embeddings（响应解析） | mock | 3 |
| 单元测试 | llm_client（消息解析） | mock | 3 |
| 集成测试 | vector_store（ChromaDB 操作） | real ChromaDB + mock embedding | 4 |
| 集成测试 | rag（管线编排） | mock embedding + mock LLM | 2 |
| 冒烟测试 | 项目状态、模块导入 | 无 | 4 |

**运行方式：**

```bash
pytest tests/ -v            # 全部 26 项
pytest tests/unit/ -v       # 单元测试（毫秒级，无外部依赖）
pytest tests/integration/ -v # 集成测试
```

---

## RAG 实现详解

### 向量检索流程

```
用户问题 "如何实现 RAG？"
    ↓
1. 客户端算向量
    OllamaEmbeddingClient.embed_query("如何实现 RAG？")
    → [0.123, 0.456, ..., 0.789]
    ↓
2. ChromaDB 搜索
    collection.query(query_embeddings=[向量], n_results=4)
    ↓
3. 返回结果
    [("RAG 的基本架构...", {"source": "doc1.txt"}, 0.12),
     ("嵌入模型选择...",  {"source": "doc2.txt"}, 0.23),
     ...]
    ↓
4. 构建上下文
    [1] (doc1.txt)
    RAG 的基本架构包括检索器和生成器两个组件。
    
    [2] (doc2.txt)  
    嵌入模型的选择直接影响检索质量。
    ↓
5. LLM 生成回答
    基于上下文 + 问题 → 生成带引用的回答
    ↓
6. 返回
    {"answer": "RAG 系统由...", "sources": [{"source": "doc1.txt", ...}, ...]}
```

### 与 LangChain 版本的关键差异

| 维度 | LangChain 版本 | 手搓版本 |
|------|---------------|---------|
| Embedding | `OllamaEmbeddings` + Chroma 自动调用 | 客户端显式调用 `OllamaEmbeddingClient` |
| 向量检索 | `retriever.invoke()` 返回 Document 对象 | `similarity_search()` 返回 tuple |
| LLM 调用 | `ChatOllama.invoke()` + 消息类 | `OllamaChatClient.chat()` + 自定 Message |
| 文本分块 | `RecursiveCharacterTextSplitter` | 自实现递归分割 |
| 异步 | 混合（部分 sync） | 全 async |
| 测试 | 无 | 26 项自动化测试 |

---

## 技术栈总览

| 类别 | 技术 | 用途 |
|------|------|------|
| 框架 | FastAPI | Web 服务 + 依赖注入 |
| 运行时 | Uvicorn | ASGI 服务器 |
| 语言 | Python 3.11+ | 主力开发语言 |
| LLM 服务 | Ollama | 本地大模型部署 |
| 向量数据库 | ChromaDB | 向量索引与相似度搜索 |
| HTTP 客户端 | httpx | 异步 HTTP 请求（替代 requests） |
| Git 操作 | GitPython | 代码仓库克隆 |
| 配置管理 | Pydantic Settings | 环境变量 + 类型校验 |
| 测试 | pytest + pytest-asyncio | 自动化测试框架 |
| 容器化 | Docker + Docker Compose | 生产部署 |
| 反向代理 | Nginx | SSL + 负载均衡 |
