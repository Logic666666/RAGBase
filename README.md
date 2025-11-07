## AI RAG 知识库系统 (FastAPI + LangChain + Ollama DeepSeek)

一个生产就绪、可云端部署的 RAG 服务：
- 模型分离：独立配置嵌入模型(Embedding)和对话模型(Chat)
- 后端：FastAPI
- 嵌入模型：通过 `langchain-ollama` 使用 Ollama (DeepSeek) 生成向量
- 对话模型：独立配置的 Ollama 模型用于高质量回答生成
- 向量数据库：Chroma 持久化存储
- 数据摄取：上传文本文件或解析 Git 仓库
- RAG 聊天：基于选定知识库的检索增强生成

### 功能特性
- 创建/列出/删除知识库 (KB)
- 上传文件 (`.txt,.md,.py,.java,.sql,.csv,.json`)
- Git 仓库摄取 (HTTPS + 可选令牌)
- 使用 `RecursiveCharacterTextSplitter` 进行文本分块
- 通过 Ollama 使用 DeepSeek 进行上下文查询
- 本地测试的极简网页界面

### 快速开始 (本地)
1) 安装 Ollama：访问 `https://ollama.com/download`
2) 拉取 DeepSeek 模型（选择一个）：
```bash
ollama pull deepseek-r1:1.5b
# 或：ollama pull deepseek-r1:7b
```
3) Python 环境
```bash
python -m venv .venv && .venv/Scripts/activate
pip install -r requirements.txt
```
4) 运行 API
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8090 --reload
```
5) 打开文档：`http://localhost:8090/docs`

### Docker (应用 + Ollama)
```bash
docker compose up -d --build
# 首次运行可能需要一些时间来拉取模型。
```

服务：
- 应用：`http://localhost:8090`
- Ollama：`http://localhost:11434`

### 环境配置
创建 `.env` 文件（可选）：
```
OLLAMA_BASE_URL=http://localhost:11434
# 嵌入模型（用于文本向量化）
EMBEDDING_MODEL=deepseek-r1:1.5b
# 对话模型（用于生成回答）
CHAT_MODEL=deepseek-r1:1.5b
DATA_DIR=./data
```

### API 概览
- POST `/kb` 创建知识库
- GET `/kb` 列出知识库
- DELETE `/kb/{kb}` 删除知识库
- POST `/kb/{kb}/upload` 多部分文件上传
- POST `/kb/{kb}/git` JSON { repo_url, branch?, username?, token? }
- POST `/chat` JSON { kb, question, top_k? }

### 极简 Web UI
打开 `http://localhost:8090/` 获取一个轻量级页面来尝试上传和提问。

### 部署到云端 (Ubuntu 示例)
1) 配置 4GB+ RAM 的虚拟机
2) 安装 Docker、Docker Compose
3) 设置 DNS/域名并开放端口 80/443/11434/8090（或放在反向代理后面）
4) `docker compose up -d --build`
5) 可选择在前端放置 Nginx，配置 TLS 并代理到 `app:8090`

### 注意事项
- Chroma 数据库持久化存储在 `data/vectorstore/<kb>`
- 摄取支持常见文本/代码文件；根据需要扩展加载器
- 对于私有 Git，在 URL 中使用令牌或提供 `username` + `token`
