# 技术栈分析文档

## 项目概述

基于 FastAPI + ChromaDB + Ollama 的知识管理与研究系统，融合 RAG 问答与 ReAct Agent 两种模式。
所有基础设施组件（LLM 客户端、Embedding 客户端、向量存储、文本分块）及 Agent 核心（ReAct 循环、工具系统、可观测性）均为手搓实现，不依赖 LangChain 等第三方 agent/LLM 框架。

---

## 架构设计

### 分层架构

```
┌───────────────────────────────────────────────┐
│              API 层（app/main.py）               │
│   FastAPI 路由 · Pydantic Schema · 依赖注入     │
│   · Agent 组装工厂（build_agent）               │
├───────────────────────────────────────────────┤
│           业务服务层（app/services/）             │
│   知识库管理（kb.py）· RAG 问答（rag.py）         │
│   · search_docs() 共享检索入口                   │
├───────────────────┬───────────────────────────┤
│  Agent 核心        │  基础设施层                  │
│  （agent/）        │  （infrastructure/）         │
│                    │                             │
│  · ReAct 循环      │  · llm_client.py（手搓）     │
│  · 系统提示词       │  · embeddings.py（手搓）     │
│  · 工具调用解析     │  · vector_store.py          │
│                    │  · text_splitter.py         │
│  工具系统(tools/)  │                             │
│  · BaseTool/ToolSpec│  可观测性（observability/） │
│  · ToolRegistry    │  · tracer.py（Trace）        │
│  · KnowledgeBaseTool│                             │
└───────────────────┴───────────────────────────┘
```

### 关键设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| LLM 框架 | 自实现，不依赖 LangChain | 完全控制调用流程，理解底层协议 |
| Embedding 调用 | 客户端主动发送 HTTP 请求 | 非黑盒，每一环可观测、可缓存、可测试 |
| 异步支持 | 全 async（FastAPI + httpx） | LLM 调用为长 I/O 等待，异步不阻塞 |
| 工具调用格式 | 文本 JSON + 容错解析（而非 Ollama native tool calling） | 本地小模型（如 qwen3:2b）的 native tool calling 不可靠，文本 JSON 方案模型无关 |
| Agent 循环 | 单线程 ReAct while-loop | 对齐 Claude Code 的 master loop 设计：简单、可调试、不依赖复杂编排 |

---

## 基础设施层（app/infrastructure/）

基础设施层是项目与外部系统交互的桥梁，四个组件全部自实现。

### 1. LLM 客户端（llm_client.py）

直接通过 HTTP API 调用 Ollama `/api/chat` 接口。

```python
class OllamaChatClient:
    async def chat(self, messages: list[Message]) -> str
    async def chat_with_response(self, messages: list[Message]) -> ChatResponse
```

**可配置参数**（通过 `.env` 注入）：
- `think`：思考型模型的 think 阶段开关。关闭可大幅提速——Agent 循环已有显式 thought 字段，模型内部思考是重复劳动
- `max_tokens`：单次生成上限；None 不限制
- `timeout`：HTTP 超时（默认 600 秒，CPU 推理慢模型留足余量）

**Ollama API 交互格式：**

```
POST /api/chat
{
    "model": "qwen3:2b",
    "messages": [{"role": "user", "content": "你好"}],
    "stream": false,
    "think": false,
    "options": {"temperature": 0.2}
}
```

### 2. Embedding 客户端（embeddings.py）

直接调用 Ollama `/api/embeddings` 接口将文本向量化。

**设计原则（客户端主动调用而非委托给向量库）：**
- Embedding 过程透明可控，每一步可观测、可异常处理
- 可添加缓存层（相同文本避免重复计算）
- 可替换 embedding 提供方而不影响向量存储层

### 3. 向量存储（vector_store.py）

基于 ChromaDB 原生 SDK，`add_documents` / `similarity_search` / `delete_collection` 三个核心操作，全部 async。

- Embedding 在客户端算好再传入，不在 ChromaDB 侧自动调用
- 使用 `PersistentClient`，数据持久化到 `data/vectorstore/<kb_name>/`
- 余弦距离（cosine）作为相似度度量

### 4. 文本分块（text_splitter.py）

递归字符分割器，纯函数实现。分隔符优先级：段落 → 行 → 中文句号 → 英文句号 → 空格 → 字符级。支持 chunk_size / chunk_overlap 配置，中英文混合优化。

---

## Agent 核心（app/agent/）

Agent 架构参考 Claude Code 的设计理念：**"简单的 while-loop + 强大的工具系统，把复杂性放在确定性的基础设施里"**。

### ReAct 循环（orchestrator.py）

```
收到任务 → 组装消息历史（system prompt + 工具列表 + 用户任务）
    → LLM 推理
    → 解析输出：工具调用 or 最终回答
        ├─ 工具调用 → 执行工具 → 结果写回消息历史 → 回到 LLM 推理
        └─ 最终回答 → 返回 AgentResult
```

**安全机制**：
- `max_steps`（默认 5）：防死循环
- `max_parse_failures`（默认 2）：连续工具错误自动终止
- 工具执行错误以结构化 `ToolResult` 返回给 LLM，由 LLM 自己决定重试/换词/放弃

**消息历史结构**（ReAct 的"思考-行动-观察"链）：

```
[system]    角色设定 + 工具列表 + 输出格式约束
[user]      用户任务
[assistant] {"thought": "...", "tool": "search_kb", "arguments": {...}}
[user]      [工具结果 search_kb] 检索到的文档...
[assistant] 最终回答
```

### 工具调用解析（schemas.py）

模型输出可能是纯 JSON，也可能是"独白 + JSON"混合文本（关闭 think 模式后的常见行为）。解析器两段式容错：

1. 整段尝试解析 → 成功且有 `tool` 字段 → 工具调用
2. 失败 → 提取第一个 `{` 到最后一个 `}` 的子串再解析
3. 都失败 → 视为最终回答

**设计参考**：Claude Code 依赖 Claude 模型的原生 `tool_use` 结构化输出块，不解析文本。本项目因使用本地小模型（native tool calling 不可靠），采用文本 JSON + 容错解析——这是模型约束下的正确取舍，容错层承担了 Claude Code 中模型原生能力承担的工作。

### 系统提示词（prompts.py）

prompt 是 Agent 的"宪法"，定义角色、工具列表、输出格式约束、行为准则。关键设计：
- 强制规则优于建议规则（"必须先调用工具"强于"建议调用工具"）
- few-shot 完整示例：给模型一个"思考-行动-观察-回答"的标准示范
- 明确禁止项："禁止在 JSON 前后输出思考过程"

---

## 工具系统（app/tools/）

### 工具抽象（base.py）

```python
@dataclass
class ToolSpec:      # 工具的"身份证"
    name: str
    description: str
    parameters: dict  # 完整 JSON Schema

@dataclass
class ToolResult:    # 结构化执行结果（对齐 Claude Code 的 tool_result + is_error）
    ok: bool
    content: str

class BaseTool(ABC):
    spec: ToolSpec
    async def run(self, **kwargs) -> str
    async def execute(self, kwargs) -> ToolResult  # 模板方法：校验 + 执行 + 异常兜底
```

**Schema 双重职责**（对齐 Claude Code 的 Zod inputSchema 设计）：
- `parameters`（JSON Schema）放在 system prompt 中约束 LLM 输出参数
- 同一 schema 在运行时用 `jsonschema` 校验模型传入的参数（LLM 输出不可信）

**ToolResult 结构化错误**（对齐 Claude Code 的 `tool_result` + `is_error` 标记）：
- orchestrator 用 `result.ok` 字段判断成功/失败，而非字符串嗅探
- 成功/失败以不同前缀写回消息历史（`[工具结果]` / `[工具错误]`），模型能明确区分

### 注册中心（registry.py）

- `register()`：工具注册（应用启动时）
- `schemas()`：生成工具列表给 LLM（放在 system prompt）
- `execute()`：按名称执行工具，返回 ToolResult

### 内置工具（builtin/）

**KnowledgeBaseTool（search_kb）**：将 `RagService.search_docs()` 包装为 Agent 工具。只检索不生成——生成是 Agent 循环中 LLM 自己的事。知识库名称在初始化时注入（用户已选好），调用时只需传 query。

---

## 可观测性（app/observability/）

### Tracer（tracer.py）

记录 Agent 执行每一步的事件，用于调试、展示、评估：

```json
{
  "run_id": "a1b2c3d4",
  "total_duration_ms": 134500.2,
  "events": [
    {"event": "llm", "detail": "{\"thought\": ...}", "duration_ms": 45230.5, "step": 0},
    {"event": "tool_call", "detail": "search_kb {'query': '...'}", "duration_ms": 1.2, "step": 1},
    {"event": "tool_result", "detail": "[1] (...)...", "duration_ms": 320.1, "step": 1}
  ]
}
```

**耗时统计**（每步评估的基础数据）：
- `llm` 事件的 `duration_ms`：单次 LLM 推理耗时（CPU 慢模型的大头）
- `tool_call → tool_result` 的 `duration_ms`：工具执行耗时
- `total_duration_ms`：任务总耗时，可对比不同模型/prompt 配置

**设计参考**：Claude Code 通过 trace/transcript 支撑"可复盘"与"改进闭环"。Tracer 为内存存储，生命周期与请求一致，每次执行返回完整 trace 供前端展示与调试。

---

## 业务服务层（app/services/）

### RAG 问答（rag.py）

两层 API，共享同一检索入口（`search_docs`），避免维护两套检索逻辑：

| 方法 | 职责 | 消费方 |
|------|------|--------|
| `search_docs(kb, query, top_k)` | 纯检索，不做生成 | `/chat` 和 Agent 的 KnowledgeBaseTool |
| `answer_question(kb, question, top_k)` | 检索 + 一轮 LLM 生成 | `POST /chat` 端点 |

### 知识库服务（kb.py）

创建/列出/删除知识库、文件上传、Git 仓库导入（支持 Token 认证、加速代理、重试策略）。

---

## 测试体系

52 项自动化测试，分层策略：

| 层级 | 测试对象 | 依赖 | 数量 |
|------|---------|------|------|
| 单元测试 | text_splitter（纯算法） | 无 | 9 |
| 单元测试 | embeddings / llm_client（响应解析） | mock | 6 |
| 单元测试 | schemas（工具调用解析，含脏输出容错） | 无 | 8 |
| 单元测试 | tool_registry（jsonschema 校验 + ToolResult） | 无 | 8 |
| 单元测试 | agent_orchestrator（ReAct 循环控制） | mock LLM | 4 |
| 单元测试 | tracer（事件 + 耗时统计） | 无 | 6 |
| 集成测试 | vector_store（ChromaDB 操作） | 真实 ChromaDB + mock embedding | 4 |
| 集成测试 | rag（管线编排） | mock embedding + LLM | 2 |
| 冒烟测试 | 项目状态、模块导入 | 无 | 4 |

**运行方式：**

```bash
pytest tests/ -v            # 全部
pytest tests/unit/ -v       # 单元测试（毫秒级，无外部依赖）
```

---

## 技术栈总览

| 类别 | 技术 | 用途 |
|------|------|------|
| 框架 | FastAPI | Web 服务 + 依赖注入 |
| 运行时 | Uvicorn | ASGI 服务器 |
| 语言 | Python 3.11+ | 主力开发语言 |
| LLM 服务 | Ollama | 本地大模型部署（对话 + 嵌入） |
| 向量数据库 | ChromaDB | 向量索引与相似度搜索 |
| HTTP 客户端 | httpx | 异步 HTTP 请求 |
| 数据校验 | jsonschema | 工具参数运行时校验 |
| Git 操作 | GitPython | 代码仓库克隆 |
| 配置管理 | Pydantic Settings | 环境变量 + 类型校验 |
| 测试 | pytest + pytest-asyncio | 自动化测试框架 |
| 容器化 | Docker + Docker Compose | 生产部署 |
| 反向代理 | Nginx | SSL + 负载均衡 |
