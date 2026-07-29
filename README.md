# AI RAG Knowledge — RAG + Agent 知识管理与研究助手

模块化、可扩展的知识管理与研究系统。当前基于 RAG（Retrieval-Augmented Generation）实现知识库问答，基础设施全部手搓，直接对接 Ollama API。

核心设计原则：

- **手搓基础设施**：不依赖 LangChain 等框架，直接对接 LLM API，掌握底层原理
- **模块化架构**：基础设施层、Agent 核心、工具系统、业务服务清晰分层
- **工程化导向**：类型安全、可测试、可观测、配置驱动

---

## 项目结构

```
ai-rag-knowledge/
│
├── app/                                   # 核心应用包
│   ├── __init__.py
│   ├── main.py                            # FastAPI 入口 + 路由
│   │
│   ├── api/                               # API 层（路由和 schema）
│   │   └── __init__.py
│   │
│   ├── core/                              # 核心配置
│   │   ├── __init__.py
│   │   └── config.py                      # Settings + 依赖注入
│   │
│   ├── infrastructure/                    # 基础设施层（手搓）
│   │   ├── __init__.py
│   │   ├── embeddings.py                  # Ollama Embedding 客户端
│   │   ├── llm_client.py                  # Ollama Chat 客户端
│   │   ├── text_splitter.py               # 递归文本分块
│   │   └── vector_store.py                # ChromaDB 向量存储
│   │
│   ├── services/                          # 业务服务层
│   │   ├── __init__.py
│   │   ├── kb.py                          # 知识库管理（文件/Git 导入）
│   │   └── rag.py                         # RAG 问答管线
│   │
│   ├── agent/                             # Agent 核心（Phase 2）
│   │   └── __init__.py
│   │
│   ├── tools/                             # 工具系统（Phase 2）
│   │   ├── __init__.py
│   │   └── builtin/
│   │       └── __init__.py
│   │
│   └── utils/                             # 通用工具
│       └── __init__.py
│
├── tests/                                 # 测试
│   ├── __init__.py
│   ├── conftest.py                        # 共享 fixture
│   ├── unit/                              # 单元测试（14 tests）
│   └── integration/                       # 集成测试（12 tests）
│
├── docs/                                  # 文档
│   ├── index.md                           # 项目总览
│   ├── deployment.md                      # 部署指南
│   └── tech-stack.md                      # 技术栈分析
│
├── deploy-configs/                        # 云部署配置
├── static/                                # Web UI
├── data/                                  # 运行时数据
│
├── pyproject.toml                         # 项目元数据 + 依赖
├── environment.yml                        # Conda 环境定义
├── Makefile                               # 常用命令
├── Dockerfile
└── README.md
```

### 架构分层

```
┌───────────────────────────────────────────────────┐
│                    API 层 (api/)                    │
│  路由定义 · 请求/响应 Schema · 依赖注入              │
├───────────────────────────────────────────────────┤
│                业务服务层 (services/)                │
│  知识库管理 · RAG 问答管线                          │
├───────────────────┬───────────────────────────────┤
│   Agent 核心      │     基础设施层                   │
│   (agent/)        │   (infrastructure/)            │
│                    │                                │
│   · Agent Loop     │   · LLM 客户端（手搓）          │
│   · 记忆系统       │   · Embedding 客户端（手搓）     │
│   · 规划模块       │   · 向量存储（ChromaDB）        │
│                    │   · 文本分块                    │
│  工具系统 (tools/) │                                │
│   · 工具注册中心   │                                │
│   · BaseTool 抽象  │                                │
│   · 内置工具集     │                                │
└───────────────────┴───────────────────────────────┘
```

---

## 功能特性

### 当前（RAG 阶段）

- 创建 / 列出 / 删除知识库
- 文件上传（`.txt` · `.md` · `.py` · `.java` · `.sql` · `.csv` · `.json`）
- Git 仓库摄取（HTTPS + 可选令牌认证）
- 智能文本分块（中英文混合优化，递归字符分割）
- 向量检索增强生成（RAG 问答，来源引用追踪）
- 基础设施全部手搓：LLM 客户端、Embedding 客户端、向量存储、文本分块
- 完整的 async 异步支持
- 26 项自动化测试覆盖

### 规划中（Agent 阶段）

- ReAct Agent 循环（推理 → 行动 → 观察）
- 可插拔工具系统（Tool Registry + BaseTool）
- 分层记忆系统（短期 / 长期 / 工作记忆）
- 任务规划模块（Plan-and-Solve）
- 多源 Research Agent + 结构化报告

---

## 快速开始

### 前置条件

1. **安装 Ollama**：[https://ollama.com/download](https://ollama.com/download)
2. **拉取模型**：

```bash
ollama pull bge-m3            # 嵌入模型
ollama pull deepseek-r1:1.5b  # 对话模型
```

### 安装与运行

```bash
# 方式一：Conda
conda env create -f environment.yml
conda activate ai-rag

# 方式二：venv + pip
python -m venv .venv && .venv/Scripts/activate
pip install -e ".[dev]"

# 启动服务
uvicorn app.main:app --host 0.0.0.0 --port 8090 --reload
# 或 make run
```

### 验证

```bash
curl http://localhost:8090/health
# → {"ok":true,"model":"deepseek-r1:1.5b","ollama":"http://localhost:11434"}
```

- **API 文档**：`http://localhost:8090/docs`
- **Web UI**：`http://localhost:8090/static/index.html`

### 运行测试

```bash
pytest tests/ -v            # 全部测试
pytest tests/unit/ -v       # 只跑单元测试
```

---

## API 概览

| 方法       | 路径                  | 说明          |
| ---------- | --------------------- | ------------- |
| `GET`    | `/health`           | 健康检查      |
| `POST`   | `/kb`               | 创建知识库    |
| `GET`    | `/kb`               | 列出知识库    |
| `DELETE` | `/kb/{name}`        | 删除知识库    |
| `POST`   | `/kb/{name}/upload` | 上传文件      |
| `POST`   | `/kb/{name}/git`    | 导入 Git 仓库 |
| `POST`   | `/chat`             | RAG 问答      |

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

详细部署指南请参考 [docs/deployment.md](docs/deployment.md) 和 [deploy-configs/QUICK_START.md](deploy-configs/QUICK_START.md)。

---

## 环境配置

核心配置项通过环境变量或 `.env` 文件管理：

| 配置项              | 说明               | 默认值                     |
| ------------------- | ------------------ | -------------------------- |
| `OLLAMA_BASE_URL` | Ollama 服务地址    | `http://localhost:11434` |
| `EMBEDDING_MODEL` | 文本嵌入模型       | `bge-m3`                |
| `CHAT_MODEL`      | 对话生成模型       | `deepseek-r1:1.5b`      |
| `DATA_DIR`        | 数据存储目录       | `./data`                 |
| `GIT_TIMEOUT`     | Git 操作超时（秒） | `300`                    |

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
- 文件上传支持常见文本/代码格式，不支持二进制文件
- 私有 Git 仓库使用 HTTPS + Token 认证
- 本项目的 `infrastructure/` 模块为手搓实现，不依赖 LangChain
