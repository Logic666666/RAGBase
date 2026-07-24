# AI RAG Knowledge — RAG + Agent 知识管理与研究助手

> 从 RAG 知识库向自主 Agent 架构演进的工程化项目

---

## 项目定位

一个**模块化、可扩展**的知识管理与研究系统。当前基于 RAG（Retrieval-Augmented Generation）实现知识库问答，正在逐步演进为**多源 Research Agent**，支持自主规划、多工具协作和结构化报告生成。

核心设计原则：

- **手搓基础设施**：不依赖 LangChain 等框架，直接对接 LLM API，掌握底层原理
- **模块化架构**：基础设施层、Agent 核心、工具系统、业务服务清晰分层
- **工程化导向**：类型安全、可测试、可观测、配置驱动

---

## 项目结构

```
ai-rag-knowledge/
│
├── app/                                   # ★ 核心应用包
│   ├── __init__.py
│   ├── main.py                            # FastAPI 入口
│   │
│   ├── api/                               # API 层
│   │   └── __init__.py                    # （后续拆分路由和 schema）
│   │
│   ├── core/                              # 核心配置
│   │   ├── __init__.py
│   │   └── config.py                      # Settings + 依赖注入
│   │
│   ├── infrastructure/                    # ★ 基础设施层（手搓目标）
│   │   ├── __init__.py
│   │   ├── vector_store.py                # 向量数据库封装（Chroma）
│   │   └── text_splitter.py               # 文本分块策略
│   │
│   ├── services/                          # 业务服务层
│   │   ├── __init__.py
│   │   ├── kb.py                          # 知识库管理（创建/文件/Git 导入）
│   │   └── rag.py                         # RAG 问答管线
│   │
│   ├── agent/                             # ★ Agent 核心（规划中）
│   │   └── __init__.py
│   │
│   ├── tools/                             # ★ 工具系统（规划中）
│   │   ├── __init__.py
│   │   └── builtin/
│   │       └── __init__.py
│   │
│   └── utils/                             # 通用工具
│       └── __init__.py
│
├── tests/                                 # ★ 测试（新增）
│   ├── __init__.py
│   ├── unit/                              # 单元测试
│   └── integration/                       # 集成测试
│
├── docs/                                  # ★ 文档集中管理
│   ├── index.md                           # 项目总览（原 README 迁移）
│   ├── deployment.md                      # 部署指南（原 DEPLOY.md 迁移）
│   └── tech-stack.md                      # 技术栈分析（原技术栈.md 迁移）
│
├── scripts/                               # ★ 构建脚本
│   ├── build.sh                           # Docker 构建 + 推送（原 build-and-push.sh）
│   └── quick-build.sh                     # 快速构建
│
├── deploy-configs/                        # 云部署配置
│   ├── docker-compose.yml
│   ├── docker-compose.prod.yml
│   ├── .env                               # 部署环境变量
│   ├── env.example
│   ├── deploy.sh
│   ├── nginx/
│   │   ├── nginx-http.conf
│   │   ├── nginx-https.conf
│   │   ├── entrypoint.sh
│   │   └── ssl/
│   └── ...
│
├── static/
│   └── index.html                         # 极简 Web UI
│
├── data/                                  # 运行时数据
│   ├── kb/
│   └── vectorstore/
│
├── pyproject.toml                         # ★ 项目元数据
├── Makefile                               # ★ 常用命令入口
├── requirements.txt
├── Dockerfile
├── .gitignore
└── README.md
```

### 架构分层

```
┌───────────────────────────────────────────────────┐
│                    API 层 (api/)                    │
│  路由定义 · 请求/响应 Schema · 依赖注入             │
├───────────────────────────────────────────────────┤
│                业务服务层 (services/)                │
│  知识库管理 · RAG 问答管线                          │
├───────────────────┬───────────────────────────────┤
│   Agent 核心      │     基础设施层                   │
│   (agent/)        │   (infrastructure/)            │
│                    │                                │
│   · Agent Loop     │   · LLM 客户端（手搓）           │
│   · 记忆系统       │   · Embedding 客户端（手搓）     │
│   · 规划模块       │   · 向量存储                    │
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
- 智能文本分块（中英文混合优化）
- 向量检索增强生成（RAG 问答）
- 来源引用追踪

### 规划中（Agent 阶段）

- 手搓 LLM / Embedding / VectorStore 基础设施
- ReAct Agent 循环（推理 → 行动 → 观察）
- 可插拔工具系统（Tool Registry + BaseTool）
- 分层记忆系统（短期 / 长期 / 工作记忆）
- 任务规划模块（Plan-and-Solve）
- 多源 Research Agent + 结构化报告

---

## 快速开始（本地开发）

### 前置条件

1. **安装 Ollama**：[https://ollama.com/download](https://ollama.com/download)
2. **拉取模型**（可根据需要选择）：

```bash
ollama pull bge-m3          # 嵌入模型
ollama pull deepseek-r1:1.5b  # 对话模型
```

### 安装与运行

```bash
# 1. 创建虚拟环境
python -m venv .venv && .venv/Scripts/activate

# 2. 安装依赖
pip install -r requirements.txt

# 3. 启动服务
uvicorn app.main:app --host 0.0.0.0 --port 8090 --reload

# 或使用 Makefile
make run
```

### 验证

```bash
curl http://localhost:8090/health
# → {"ok":true,"model":"deepseek-r1:1.5b","ollama":"http://localhost:11434"}
```

浏览器访问：

- **API 文档**：`http://localhost:8090/docs`
- **Web UI**：`http://localhost:8090/static/index.html`

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

### 应用单独构建

```bash
# 构建镜像
make docker-build

# 运行
make docker-run

# 或手动
docker build -t ai-rag-app:latest .
docker run -d --name ai-rag -p 8090:8090 ai-rag-app:latest
```

### 完整服务编排（应用 + Ollama + Nginx）

```bash
cd deploy-configs
cp env.example .env    # 编辑配置
docker compose up -d --build
```

### 构建脚本

```bash
# 快速构建 + 推送
./scripts/quick-build.sh

# 完整构建流程（含测试）
./scripts/build.sh
```

详细部署指南请参考：

- [部署文档](docs/deployment.md)
- [部署配置说明](deploy-configs/QUICK_START.md)

---

## 环境配置

核心配置项通过环境变量或 `.env` 文件管理：

| 配置项              | 说明               | 默认值                     |
| ------------------- | ------------------ | -------------------------- |
| `OLLAMA_BASE_URL` | Ollama 服务地址    | `http://localhost:11434` |
| `EMBEDDING_MODEL` | 文本嵌入模型       | `deepseek-r1:1.5b`       |
| `CHAT_MODEL`      | 对话生成模型       | `deepseek-r1:1.5b`       |
| `DATA_DIR`        | 数据存储目录       | `./data`                 |
| `GIT_TIMEOUT`     | Git 操作超时（秒） | `300`                    |

部署环境配置（Docker 端口、SSL、镜像仓库等）请使用 `deploy-configs/.env`。

---

## 开发命令

```bash
make run          # 启动开发服务器（热重载）
make test         # 运行测试
make clean        # 清除 __pycache__
make shell        # Python shell 快速载入包
make docker-build # Docker 构建
make docker-run   # Docker 运行
```

---

## 演进路线

| 阶段              | 内容                                            | 状态      |
| ----------------- | ----------------------------------------------- | --------- |
| **Phase 0** | 项目结构重构，代码整理                          | ✅ 完成   |
| **Phase 1** | 脱框：移除 LangChain，手搓基础设施              | 📝 规划中 |
| **Phase 2** | Agent 骨架：Tool System + Agent Loop + Memory   | 📝 规划中 |
| **Phase 3** | 多源 Research Agent：规划 + 工具协作 + 报告生成 | 📝 规划中 |

---

## 注意事项

- Chroma 向量数据库持久化在 `data/vectorstore/<kb_name>/`
- 文件上传支持常见文本/代码格式，不支持二进制文件
- 私有 Git 仓库使用 HTTPS + Token 认证
- 本项目的 `infrastructure/` 和 `agent/` 模块为手搓实现，不依赖 LangChain
