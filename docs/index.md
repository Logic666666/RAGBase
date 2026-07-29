# AI RAG Knowledge — RAG + Agent 知识管理与研究助手

后端采用 **FastAPI**，基础设施全部手搓、不依赖 LangChain，直接对接 **Ollama API**（LLM 对话 + 文本嵌入），使用 **ChromaDB** 做向量存储。

### 项目定位

- 🧱 **手搓基础设施**：LLM 客户端、Embedding 客户端、向量存储、文本分块全部自实现
- 🧩 **模块化架构**：基础设施层、Agent 核心、工具系统、业务服务清晰分层
- 🔧 **工程化导向**：类型安全、可测试、可观测、配置驱动

### 功能特性

- 创建 / 列出 / 删除知识库
- 文件上传（`.txt`、`.md`、`.py`、`.java`、`.sql`、`.csv`、`.json`）
- Git 仓库摄取（HTTPS + 可选令牌认证）
- 智能文本分块（中英文混合优化，递归字符分割）
- 向量检索增强生成（RAG 问答，来源引用追踪）
- 基础设施全部手搓：LLM 客户端、Embedding 客户端、向量存储、文本分块
- 完整的 async 异步支持
- 26 项自动化测试覆盖

### 快速开始

#### 前置条件

1. **安装 Ollama**：[https://ollama.com/download](https://ollama.com/download)
2. **拉取模型**：

```bash
ollama pull bge-m3            # 嵌入模型
ollama pull deepseek-r1:1.5b  # 对话模型
```

#### Conda（推荐）

```bash
conda env create -f environment.yml
conda activate ai-rag
uvicorn app.main:app --host 0.0.0.0 --port 8090 --reload
```

#### venv + pip

```bash
python -m venv .venv && .venv/Scripts/activate
pip install -e ".[dev]"
uvicorn app.main:app --host 0.0.0.0 --port 8090 --reload
```

#### 验证

```bash
curl http://localhost:8090/health
# → {"ok":true,"model":"deepseek-r1:1.5b","ollama":"http://localhost:11434"}
```

浏览器访问：

- **API 文档**：`http://localhost:8090/docs`
- **Web UI**：`http://localhost:8090/static/index.html`

#### 运行测试

```bash
pytest tests/ -v       # 全部 26 项测试
pytest tests/unit/ -v  # 只跑单元测试
```

### Docker 部署

```bash
docker compose up -d --build
```

服务：

- 应用：`http://localhost:8090`
- Ollama：`http://localhost:11434`

### 环境配置

核心配置项通过环境变量或 `.env` 文件管理：

| 配置项              | 说明             | 默认值                     |
| ------------------- | ---------------- | -------------------------- |
| `OLLAMA_BASE_URL` | Ollama 服务地址  | `http://localhost:11434` |
| `EMBEDDING_MODEL` | 文本嵌入模型     | `bge-m3`                |
| `CHAT_MODEL`      | 对话生成模型     | `deepseek-r1:1.5b`      |
| `DATA_DIR`        | 数据存储目录     | `./data`                 |
| `GIT_TIMEOUT`     | Git 操作超时(秒) | `300`                    |

详细部署配置请参考 `deploy-configs/.env`。

### API 概览

| 方法     | 路径                  | 说明          |
| -------- | --------------------- | ------------- |
| `GET`  | `/health`           | 健康检查      |
| `POST` | `/kb`               | 创建知识库    |
| `GET`  | `/kb`               | 列出知识库    |
| `DELETE` | `/kb/{name}`       | 删除知识库    |
| `POST` | `/kb/{name}/upload` | 上传文件      |
| `POST` | `/kb/{name}/git`    | 导入 Git 仓库 |
| `POST` | `/chat`             | RAG 问答      |

### 注意事项

- Chroma 数据库持久化存储在 `data/vectorstore/<kb>`
- 文件上传支持常见文本/代码格式，不支持二进制文件
- 私有 Git 仓库使用 HTTPS + Token 认证
- 基础设施模块（`app/infrastructure/`）为手搓实现，不依赖 LangChain
