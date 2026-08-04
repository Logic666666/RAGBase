# AI RAG Knowledge — RAG + Agent 知识管理与研究助手

模块化的知识管理与研究系统，融合 RAG 问答与 ReAct Agent 两种问答模式。基础设施全部手搓（不依赖 LangChain 等框架），直接对接 Ollama API。

核心设计原则：

- **手搓基础设施**：LLM 客户端、Embedding 客户端、向量存储、文本分块全部自实现，掌握底层原理
- **Agent 架构对齐行业实践**：ReAct 循环、结构化工具系统、可观测 trace，参考 Claude Code 的 agent 架构设计
- **工程化导向**：类型安全、可测试、可观测、配置驱动、异步优先

---

## 项目结构

```
ai-rag-knowledge/
│
├── app/                                   # 核心应用包
│   ├── __init__.py
│   ├── main.py                            # FastAPI 入口 + 路由 + Agent 组装工厂
│   │
│   ├── api/                               # API 层
│   │   └── __init__.py
│   │
│   ├── core/                              # 核心配置
│   │   ├── __init__.py
│   │   └── config.py                      # Settings + 依赖注入
│   │
│   ├── infrastructure/                    # 基础设施层（全部手搓）
│   │   ├── __init__.py
│   │   ├── embeddings.py                  # Ollama Embedding 客户端
│   │   ├── llm_client.py                  # Ollama Chat 客户端（支持 think 开关/超时配置）
│   │   ├── text_splitter.py               # 递归文本分块（中英文混合优化）
│   │   └── vector_store.py                # ChromaDB 向量存储（async）
│   │
│   ├── services/                          # 业务服务层
│   │   ├── __init__.py
│   │   ├── kb.py                          # 知识库管理（文件/Git 导入）
│   │   └── rag.py                         # RAG 问答 + 共享检索（search_docs）
│   │
│   ├── agent/                             # Agent 核心
│   │   ├── __init__.py
│   │   ├── orchestrator.py                # ReAct 循环（思考 → 行动 → 观察）
│   │   ├── prompts.py                     # 系统提示词模板（工具列表 + 输出格式约束）
│   │   └── schemas.py                     # 工具调用解析（容错提取 JSON）
│   │
│   ├── tools/                             # 工具系统
│   │   ├── __init__.py
│   │   ├── base.py                        # BaseTool 抽象 + ToolSpec + ToolResult
│   │   ├── registry.py                    # 工具注册中心（注册/发现/执行）
│   │   └── builtin/
│   │       ├── __init__.py
│   │       └── knowledge_base.py          # 知识库检索工具（search_kb）
│   │
│   ├── observability/                     # 可观测性
│   │   ├── __init__.py
│   │   └── tracer.py                      # Trace 记录（事件 + 耗时统计）
│   │
│   └── utils/                             # 通用工具
│       └── __init__.py
│
├── tests/                                 # 测试（52 项）
│   ├── __init__.py
│   ├── conftest.py                        # 共享 fixture
│   ├── unit/                              # 单元测试（纯逻辑 + mock）
│   └── integration/                       # 集成测试（ChromaDB + 管线编排）
│
├── docs/                                  # 文档
│   ├── index.md                           # 项目总览
│   ├── deployment.md                      # 部署指南
│   └── tech-stack.md                      # 技术栈分析
│
├── deploy-configs/                        # 云部署配置
├── static/                                # Web UI（聊天界面）
├── data/                                  # 运行时数据
│
├── pyproject.toml                         # 项目元数据 + 依赖
├── environment.yml                        # Conda 环境定义
├── Makefile                               # 常用命令
├── Dockerfile
└── README.md
```

---

## 架构分层

```
┌───────────────────────────────────────────────────┐
│                 API 层（app/main.py）                │
│  路由定义 · 请求 Schema · 依赖注入 · Agent 组装工厂   │
├───────────────────────────────────────────────────┤
│           业务服务层（app/services/）                 │
│  知识库管理（kb.py）· RAG 问答（rag.py）             │
├───────────────────┬───────────────────────────────┤
│   Agent 核心       │     基础设施层                   │
│   （agent/）       │   （infrastructure/）            │
│                    │                                │
│   ReAct 循环       │   LLM 客户端（手搓）             │
│   系统提示词        │   Embedding 客户端（手搓）       │
│   工具调用解析      │   向量存储（ChromaDB）           │
│                    │   文本分块（手搓）               │
│  工具系统(tools/)  │                                │
│   BaseTool/ToolSpec│   可观测性（observability/）     │
│   ToolRegistry     │   Trace 事件 + 耗时统计          │
└───────────────────┴───────────────────────────────┘
```

---

## 功能特性

### 知识库管理

- 创建 / 列出 / 删除知识库
- 文件上传（`.txt` · `.md` · `.py` · `.java` · `.sql` · `.csv` · `.json`）
- Git 仓库摄取（HTTPS + 可选令牌认证，支持加速代理）

### 两种问答模式

| 模式                 | 后端                | 特点                                                         |
| -------------------- | ------------------- | ------------------------------------------------------------ |
| **普通问答**   | `POST /chat`      | 单轮检索 + 生成，快速直接                                    |
| **Agent 模式** | `POST /agent/run` | ReAct 多步推理：自主决定检索关键词和次数，综合多份文档后回答 |

### Agent 能力（参考 Claude Code 架构）

- **ReAct 循环**：思考 → 行动（工具调用）→ 观察 → 再思考，直到给出最终回答
- **结构化工具系统**：`ToolSpec`（JSON Schema 参数定义）同时约束模型输出和运行时校验；`ToolResult` 用 `ok` 字段标记成功/失败（对齐 Claude Code 的 `tool_result` + `is_error` 设计）
- **容错输出解析**：模型输出"独白 + JSON"混合文本时，自动提取工具调用（对齐 Claude Code 对脏输出的容错处理）
- **可观测 trace**：每步事件带耗时（`Thought for Xs`），前端以对话流形式展示思考过程
- **安全机制**：`max_steps` 防死循环、连续工具错误自动终止、jsonschema 参数校验

### Web UI（聊天界面）

- 对话流布局：用户消息右对齐，Agent 推理过程（Thought）可展开，工具调用与结果可视化
- 思考内容双通道展示：模型完整推理（think 事件）以可展开的 Thought 块呈现；工具调用 JSON 中的简短 thought 以普通文本气泡展示在流程中间，与最终回答同格式
- 输入框固定在底部，Enter 发送，多轮对话累积
- 模式切换（普通问答 / Agent 模式）与参数控制（top_k / max_steps）

---

## 快速开始

### 前置条件

1. **安装 Ollama**：[https://ollama.com/download](https://ollama.com/download)
2. **拉取模型**：

```bash
ollama pull bge-m3            # 嵌入模型
ollama pull qwen3:2b          # 对话模型（按机器性能选择）
```

### 安装与运行

```bash
# 方式一：Conda + pip
conda create -n ai-agent python=3.11 -y 
conda activate ai-agent
pip install -e ".[dev]"

# 方式二：venv + pip
python -m venv .venv && .venv/Scripts/activate
pip install -e ".[dev]"

# 配置本地 .env（模型、Ollama 地址等）
# 参考下方"环境配置"

# 启动服务（--reload-dir 限定监听代码目录，避免 data/ 写入触发重载中断请求）
uvicorn app.main:app --host 0.0.0.0 --port 8090 --reload --reload-dir app --reload-dir static
# 或 make run
```

### 验证

```bash
curl http://localhost:8090/health
# → {"ok":true,"model":"qwen3:2b","ollama":"http://localhost:11434"}
```

- **Web UI**：`http://localhost:8090/`
- **API 文档**：`http://localhost:8090/docs`

### 运行测试

```bash
pytest tests/ -v            # 全部 52 项
pytest tests/unit/ -v       # 单元测试（毫秒级，无外部依赖）
```

---

## API 概览

| 方法       | 路径                  | 说明                       |
| ---------- | --------------------- | -------------------------- |
| `GET`    | `/health`           | 健康检查                   |
| `POST`   | `/kb`               | 创建知识库                 |
| `GET`    | `/kb`               | 列出知识库                 |
| `DELETE` | `/kb/{name}`        | 删除知识库                 |
| `POST`   | `/kb/{name}/upload` | 上传文件                   |
| `POST`   | `/kb/{name}/git`    | 导入 Git 仓库              |
| `POST`   | `/chat`             | RAG 单轮问答               |
| `POST`   | `/agent/run`        | Agent 多步推理（含 trace） |

### `/agent/run` 响应结构

```json
{
  "answer": "最终回答（markdown）",
  "completed": true,
  "steps": 2,
  "trace": {
    "run_id": "a1b2c3d4",
    "total_duration_ms": 134500.2,
    "events": [
      {"event": "llm", "detail": "{\"thought\": ...}", "duration_ms": 45230.5, "step": 0},
      {"event": "tool_call", "detail": "search_kb {'query': '...'}", "duration_ms": 1.2, "step": 1},
      {"event": "tool_result", "detail": "[1] (...)...", "duration_ms": 320.1, "step": 1}
    ]
  }
}
```

---

## 环境配置

核心配置项通过环境变量或根目录 `.env` 文件管理：

| 配置项              | 说明                            | 默认值                     |
| ------------------- | ------------------------------- | -------------------------- |
| `OLLAMA_BASE_URL` | Ollama 服务地址                 | `http://localhost:11434` |
| `EMBEDDING_MODEL` | 文本嵌入模型                    | `bge-m3`                 |
| `CHAT_MODEL`      | 对话生成模型                    | `qwen3:2b`               |
| `LLM_THINK`       | 思考型模型的 think 开关         | `false`（关闭以提速）    |
| `LLM_MAX_TOKENS`  | 单次生成最大 token（空=不限制） | 空                         |
| `DATA_DIR`        | 数据存储目录                    | `./data`                 |
| `GIT_TIMEOUT`     | Git 操作超时（秒）              | `300`                    |

> 注意：`deploy-configs/.env` 是 Docker 部署配置，本地开发使用项目根目录 `.env`，两者字段不同。

---

## Docker 部署

```bash
# 应用单独构建
make docker-build
make docker-run

# 完整服务编排（应用 + Ollama + Nginx）
cd deploy-configs
cp env.example .env
docker compose up -d --build
```

详细部署指南请参考 [docs/deployment.md](docs/deployment.md)。

---

## 开发命令

```bash
make run          # 启动开发服务器（热重载）
make test         # 运行所有测试
make clean        # 清除 __pycache__
make shell        # Python shell 载入包
make docker-build # Docker 构建
make docker-run   # Docker 运行
```

---

## 注意事项

- Chroma 向量数据库持久化在 `data/vectorstore/<kb_name>/`
- **更换嵌入模型后需重建知识库索引**（向量维度/分布不同，新旧无法比较）
- 文件上传支持常见文本/代码格式，不支持二进制文件
- 私有 Git 仓库使用 HTTPS + Token 认证
- 本项目基础设施与 Agent 核心全部手搓，不依赖 LangChain 等框架
