# AI RAG Knowledge — RAG + Agent 知识管理与研究助手

模块化的知识管理与研究系统，融合 RAG 问答与 ReAct Agent 两种问答模式。后端采用 **FastAPI**，基础设施全部手搓、不依赖 LangChain，直接对接 **Ollama API**（LLM 对话 + 文本嵌入），使用 **ChromaDB** 做向量存储。

### 项目定位

- 🧱 **手搓基础设施**：LLM 客户端、Embedding 客户端、向量存储、文本分块全部自实现
- 🤖 **Agent 架构对齐行业实践**：ReAct 循环、结构化工具系统、可观测 trace，参考 Claude Code 的 agent 架构设计
- 🔧 **工程化导向**：类型安全、可测试、可观测、配置驱动、异步优先

### 功能特性

- 创建 / 列出 / 删除知识库
- 文件上传（`.txt`、`.md`、`.py`、`.java`、`.sql`、`.csv`、`.json`）
- Git 仓库摄取（HTTPS + 可选令牌认证）
- 智能文本分块（中英文混合优化，递归字符分割）
- **普通问答**：单轮检索 + 生成（`POST /chat`）
- **Agent 模式**：ReAct 多步推理，自主决定检索策略并综合回答（`POST /agent/run`）
- Agent 思考过程可视化：模型完整推理以可展开的 Thought 块呈现，工具调用意图以普通气泡展示，附工具调用与耗时统计
- 完整的 async 异步支持
- 52 项自动化测试覆盖

### 快速开始

#### 前置条件

1. **安装 Ollama**：[https://ollama.com/download](https://ollama.com/download)
2. **拉取模型**：

```bash
ollama pull bge-m3            # 嵌入模型
ollama pull qwen3:2b          # 对话模型（按机器性能选择）
```

#### 环境配置与运行

```bash
# 方式一：Conda + pip
conda create -n ai-agent python=3.11 -y 
conda activate ai-agent
pip install -e ".[dev]"

# 方式二：venv + pip
python -m venv .venv && .venv/Scripts/activate
pip install -e ".[dev]"

uvicorn app.main:app --host 0.0.0.0 --port 8090 --reload
```

#### 验证

```bash
curl http://localhost:8090/health
# → {"ok":true,"model":"qwen3:2b","ollama":"http://localhost:11434"}
```

浏览器访问：

- **Web UI**：`http://localhost:8090/`
- **API 文档**：`http://localhost:8090/docs`

#### 运行测试

```bash
pytest tests/ -v       # 全部 52 项测试
pytest tests/unit/ -v  # 只跑单元测试
```

### 环境配置

核心配置项通过环境变量或根目录 `.env` 文件管理：

| 配置项              | 说明                            | 默认值                     |
| ------------------- | ------------------------------- | -------------------------- |
| `OLLAMA_BASE_URL` | Ollama 服务地址                 | `http://localhost:11434` |
| `EMBEDDING_MODEL` | 文本嵌入模型                    | `bge-m3`                 |
| `CHAT_MODEL`      | 对话生成模型                    | `qwen3:2b`               |
| `LLM_THINK`       | 思考型模型的 think 开关         | `false`（关闭以提速）    |
| `LLM_MAX_TOKENS`  | 单次生成最大 token（空=不限制） | 空                         |
| `DATA_DIR`        | 数据存储目录                    | `./data`                 |
| `GIT_TIMEOUT`     | Git 操作超时(秒)                | `300`                    |

### API 概览

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

### Docker 部署

```bash
docker compose up -d --build
```

服务：

- 应用：`http://localhost:8090`
- Ollama：`http://localhost:11434`

### 注意事项

- Chroma 数据库持久化存储在 `data/vectorstore/<kb>`
- **更换嵌入模型后需重建知识库索引**
- 文件上传支持常见文本/代码格式，不支持二进制文件
- 私有 Git 仓库使用 HTTPS + Token 认证
- 基础设施与 Agent 核心模块（`app/infrastructure/`、`app/agent/`）为手搓实现，不依赖 LangChain
