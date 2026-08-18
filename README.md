# AI RAG Knowledge — RAG + Agent 知识管理与研究助手

模块化的知识管理与研究系统,融合 RAG 单轮问答与 ReAct Agent 多步研究两种模式。基础设施与 Agent 核心全部手搓(不依赖 LangChain 等框架),直接对接 Ollama API。

核心设计原则:

- **手搓基础设施**:LLM 客户端、Embedding 客户端、向量存储、文本分块、重试与错误分类全部自实现,掌握底层原理
- **Agent 架构对齐行业实践**:规划轮(Explore → Plan)、ReAct 循环、收敛检测、上下文压缩、可观测 trace,参考 [Claude Code 的 agent 架构](https://code.claude.com/docs/zh-CN/how-claude-code-works)
- **面向小模型的防御体系**:10 层防御链(输出容错、收敛检测、提前收尾质疑、收尾确认)保证任务可完成、结果不编造
- **工程化导向**:类型安全、可测试(169 项)、可观测、配置驱动、异步优先

---

## 快速开始

### 前置条件

1. 安装 Ollama:<https://ollama.com/download>
2. 拉取模型:

```bash
ollama pull bge-m3            # 嵌入模型
ollama pull qwen3.5:2b        # 对话模型(按机器性能选择)
```

### 安装与运行

```bash
# 方式一:Conda + pip
conda create -n ai-agent python=3.11 -y
conda activate ai-agent
pip install -e ".[dev]"

# 方式二:venv + pip
python -m venv .venv && .venv/Scripts/activate
pip install -e ".[dev]"

# 启动服务(--reload-dir 限定监听目录,避免 data/ 写入触发重载)
uvicorn app.main:app --host 0.0.0.0 --port 8090 --reload --reload-dir app --reload-dir static
```

### 验证

```bash
curl http://localhost:8090/health
# → {"ok":true,"model":"qwen3.5:2b","ollama":"http://localhost:11434"}
```

- Web UI:`http://localhost:8090/`
- API 文档:`http://localhost:8090/docs`

### 运行测试

```bash
pytest tests/ -q       # 全部 169 项
pytest tests/unit/ -q  # 单元测试(毫秒级,无外部依赖)
```

---

## 功能特性

### 知识库管理

- 创建 / 列出 / 删除知识库
- 文件上传(`.txt` · `.md` · `.py` · `.java` · `.sql` · `.csv` · `.json` · `.pdf`)
- Git 仓库摄取(HTTPS + 可选令牌认证,支持加速镜像)

### 两种问答模式

| 模式 | 后端 | 特点 |
| --- | --- | --- |
| 普通问答 | `POST /chat` | 单轮检索 + 生成,快速直接 |
| Agent 模式 | `POST /agent/run` | 多步研究:规划 → 逐篇阅读/检索 → 收敛 → 结构化报告 |

### Agent 研究流程

```
提交任务 → 规划轮(list_files 探索 → 研究计划锚点)
→ ReAct 主循环(思考 → 工具调用 → 观察,收敛检测防死循环)
→ 上下文压缩(单轮裁剪 → 历史折叠 → 窗口兜底)
→ 提前收尾质疑(0 阅读硬拦,部分阅读须说明未读原因)
→ 收尾确认(要求纯文本报告,防"声明当报告")
→ 最终报告(五段结构)
```

工具集:`search_kb`(向量检索)· `grep_code` / `read_file` / `list_files`(代码分析)· `read_pdf`(论文阅读)· `note_take` / `read_note` / `list_notes`(工作区笔记)。

### 可观测与追溯

- Trace 全量记录(事件 + 耗时 + 结束原因)
- EventBus + SSE 实时推送(晚订阅历史重放)
- Transcript 落盘 + 历史会话回看
- 前端双通道:SSE 实时预览 → 完成时原地升级正式版,不重复渲染

---

## 项目结构

```
app/
├── main.py                    # FastAPI 入口 + 路由 + build_agent 工厂
├── core/config.py             # Settings(配置管理)
├── infrastructure/            # 手搓基础设施
│   ├── llm_client.py          # OllamaChatClient(重试/思考提取/截断诊断)
│   ├── embeddings.py          # OllamaEmbeddingClient
│   ├── vector_store.py        # ChromaDB(共享 client 池 + 显式 close)
│   └── text_splitter.py       # 递归文本分块(纯函数)
├── agent/
│   ├── orchestrator.py        # ReAct 循环(规划轮 + 收敛检测 + 提前收尾质疑)
│   ├── context.py             # ContextCompressor + TrimCompressor + SummaryCompressor
│   ├── prompts.py             # PLANNING_PROMPT + AGENT_RULES + system_prompt
│   └── schemas.py             # parse_tool_call(容错解析)+ ToolCall/AgentResult
├── tools/
│   ├── base.py                # ToolSpec/ToolResult/BaseTool(jsonschema 校验)
│   ├── registry.py            # ToolRegistry
│   ├── path_utils.py          # resolve_codebase_path(公共路径解析)
│   └── builtin/               # search_kb / grep_code / read_file / list_files / read_pdf / 笔记
├── reliability/               # errors.py(错误分类)+ retry.py(指数退避)
├── events/                    # bus.py(EventBus 发布-订阅 + 历史重放)
├── sessions/                  # SessionRecord/SessionStore/SessionManager
├── observability/tracer.py    # Trace(事件 + 耗时 + done_reason)
├── workspace.py               # FileWorkspace(工作区笔记)
└── services/                  # kb.py(知识库管理)+ rag.py(RAG 问答)
```

---

## API 概览

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/health` | 健康检查 |
| `POST` | `/kb` | 创建知识库 |
| `GET` | `/kb` | 列出知识库 |
| `DELETE` | `/kb/{name}` | 删除知识库 |
| `POST` | `/kb/{name}/upload` | 上传文件 |
| `POST` | `/kb/{name}/git` | 导入 Git 仓库 |
| `POST` | `/chat` | RAG 单轮问答 |
| `POST` | `/agent/run` | 提交 Agent 研究任务(异步,返回 session_id) |
| `GET` | `/agent/events/{session_id}` | SSE 实时事件流 |
| `GET` | `/agent/status/{session_id}` | 查询任务状态 |
| `GET` | `/agent/result/{session_id}` | 获取结果 + trace + transcript |
| `GET` | `/agent/sessions` | 历史会话列表 |
| `DELETE` | `/agent/sessions/{session_id}` | 删除历史会话 |

---

## 环境配置

核心配置项通过环境变量或根目录 `.env` 文件管理:

| 配置项 | 说明 | 默认值 |
| --- | --- | --- |
| `OLLAMA_BASE_URL` | Ollama 服务地址 | `http://localhost:11434` |
| `EMBEDDING_MODEL` | 文本嵌入模型 | `bge-m3` |
| `CHAT_MODEL` | 对话生成模型 | `qwen3.5:2b` |
| `LLM_THINK` | 思考型模型的 think 开关 | `false`(关闭以提速) |
| `LLM_MAX_TOKENS` | 单次生成最大 token | `8192` |
| `LLM_NUM_CTX` | 上下文窗口大小 | `16384` |
| `DATA_DIR` | 数据存储目录 | `./data` |
| `GIT_TIMEOUT` | Git 操作超时(秒) | `300` |
| `GIT_ACCELERATORS` | Git 加速镜像列表 | 可配置 |

> 上下文压缩触发阈值与 `LLM_NUM_CTX` 联动(轮次预算 ≈ 窗口 60%)。
> `deploy-configs/.env` 是 Docker 部署配置,本地开发使用项目根目录 `.env`,两者字段不同。

---

## 开发命令

```bash
make run          # 启动开发服务器(热重载)
make test         # 运行所有测试
make clean        # 清除 __pycache__
make shell        # Python shell 载入包
make docker-build # Docker 构建
make docker-run   # Docker 运行
```

---

## 文档

- [docs/index.md](docs/index.md) — 项目总览与快速开始
- [docs/tech-stack.md](docs/tech-stack.md) — 技术栈深度分析(架构、防御链、上下文管理、可观测)
- [docs/deployment.md](docs/deployment.md) — 云服务器部署指南

## 注意事项

- Chroma 向量数据库持久化在 `data/vectorstore/<kb_name>/`
- 更换嵌入模型后需重建知识库索引(向量维度/分布不同,新旧无法比较)
- `read_file` 拒绝二进制/PDF 文件(返回明确错误并引导 `read_pdf`)
- 私有 Git 仓库使用 HTTPS + Token 认证
- 基础设施与 Agent 核心全部手搓,不依赖 LangChain 等框架
