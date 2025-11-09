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

项目使用 `.env` 文件管理配置，支持自定义模型、端口、SSL 等多种设置。

#### 配置文件说明

- **`.env.example`**：配置模板文件，包含所有可用配置项
- **`.env`**：实际使用的配置文件（从 `.env.example` 复制而来）

#### 快速配置

```bash
# 1. 复制配置模板
cp env.example .env

# 2. 编辑配置文件，根据需求修改各项参数
nano .env
```

#### 核心配置项说明

| 配置项              | 说明            | 默认值                  | 可选值                 |
| ------------------- | --------------- | ----------------------- | ---------------------- |
| `OLLAMA_BASE_URL` | Ollama 服务地址 | `http://ollama:11434` | 根据部署环境调整       |
| `EMBEDDING_MODEL` | 文本嵌入模型    | `deepseek-r1:1.5b`    | 任意 Ollama 支持的模型 |
| `CHAT_MODEL`      | 对话生成模型    | `deepseek-r1:1.5b`    | 任意 Ollama 支持的模型 |
| `HTTP_PORT`       | HTTP 端口       | `80`                  | 1024-65535             |
| `HTTPS_PORT`      | HTTPS 端口      | `443`                 | 1024-65535             |
| `USE_SSL`         | 是否启用 HTTPS  | `false`               | `true`/`false`     |
| `DATA_DIR`        | 数据存储目录    | `/data`               | 自定义路径             |

**支持的特性**：

- **智能默认值**：代码中预设了合理的默认配置
- **自动目录创建**：自动创建必要的数据目录结构
- **单例模式**：使用 `@lru_cache` 确保配置只加载一次
- **类型安全**：使用 Pydantic 进行配置验证

#### 模型配置建议

**开发环境**：推荐使用较小的模型以提高响应速度

```bash
EMBEDDING_MODEL=deepseek-r1:1.5b
CHAT_MODEL=deepseek-r1:1.5b
```

**生产环境**：推荐使用较大的模型以获得更好的效果

```bash
EMBEDDING_MODEL=deepseek-r1:7b
CHAT_MODEL=deepseek-r1:7b
```

#### 多环境配置

你可以为不同环境创建不同的配置文件：

```bash
# 开发环境配置
cp env.example .env.dev
# 修改 .env.dev 中的开发环境参数

# 生产环境配置
cp env.example .env.prod
# 修改 .env.prod 中的生产环境参数

# 使用特定配置启动
docker compose --env-file .env.dev up -d
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

### 部署到云服务器

项目已配置完整的云服务器部署方案，支持 Docker 容器化部署、Nginx 反向代理、SSL 证书等生产环境特性。

#### 快速部署（推荐）

我们提供了**部署配置文件包**，包含所有必需的配置文件，与Docker镜像分离管理：

```bash
# 1. 下载部署配置包
# 上传 deploy-configs/ 目录到云服务器

# 2. 进入部署目录
cd deploy-configs

# 3. 配置环境变量
cp env.example .env
nano .env  # 设置镜像地址等参数

# 4. 一键部署
chmod +x deploy.sh && ./deploy.sh
```

#### 部署配置文件包结构

```
deploy-configs/
├── docker-compose.yml          # 服务编排配置
├── docker-compose.prod.yml     # 生产环境优化
├── .env                       # 环境变量（需自定义）
├── nginx/                     # Nginx配置文件
│   ├── nginx-http.conf       # HTTP配置
│   ├── nginx-https.conf      # HTTPS配置
│   ├── entrypoint.sh         # 启动脚本
│   └── ssl/                  # SSL证书目录
├── deploy.sh                 # 一键部署脚本
└── QUICK_START.md           # 详细部署指南
```

#### 部署包特点

- **分离管理**：配置文件与Docker镜像分离
- **即拿即用**：包含所有必需的配置和脚本
- **灵活配置**：支持HTTP/HTTPS一键切换
- **生产优化**：内置性能优化和日志管理

#### 详细部署指南

请参考以下文档获取完整信息：

- [DEPLOY.md](DEPLOY.md) - 完整的服务器环境准备和优化指南
- [deploy-configs/QUICK_START.md](deploy-configs/QUICK_START.md) - 部署包使用指南

#### 部署验证

部署完成后，可以通过以下方式验证：

```bash
# 检查服务状态
docker compose ps

# 查看服务日志
docker compose logs -f

# 测试健康检查
curl http://localhost/health
```

### 镜像打包和推送

项目支持构建自定义 Docker 镜像并推送到镜像仓库：

#### 镜像构建

```bash
# 1. 配置镜像信息（在 .env 文件中设置）
DOCKER_REGISTRY=your-registry.com
DOCKER_NAMESPACE=your-project
DOCKER_IMAGE_NAME=ai-rag-app
DOCKER_IMAGE_TAG=latest

# 2. 构建镜像
docker build -t ${DOCKER_REGISTRY}/${DOCKER_NAMESPACE}/${DOCKER_IMAGE_NAME}:${DOCKER_IMAGE_TAG} .

# 3. 本地测试
docker run -d -p 8090:8090 ${DOCKER_REGISTRY}/${DOCKER_NAMESPACE}/${DOCKER_IMAGE_NAME}:${DOCKER_IMAGE_TAG}
curl http://localhost:8090/health
```

#### 镜像推送

```bash
# 1. 登录镜像仓库
docker login ${DOCKER_REGISTRY}

# 2. 推送镜像
docker push ${DOCKER_REGISTRY}/${DOCKER_NAMESPACE}/${DOCKER_IMAGE_NAME}:${DOCKER_IMAGE_TAG}

# 3. 可选：推送版本标签
docker tag ${DOCKER_REGISTRY}/${DOCKER_NAMESPACE}/${DOCKER_IMAGE_NAME}:${DOCKER_IMAGE_TAG} \
           ${DOCKER_REGISTRY}/${DOCKER_NAMESPACE}/${DOCKER_IMAGE_NAME}:1.0.0
docker push ${DOCKER_REGISTRY}/${DOCKER_NAMESPACE}/${DOCKER_IMAGE_NAME}:1.0.0
```

#### 使用构建脚本（推荐）

```bash
# 快速构建和推送
./quick-build.sh

# 或完整构建流程（包含测试）
./build-and-push.sh
```

```bash
# 查看服务状态
docker compose -f docker-compose.prod.yml ps

# 查看日志
docker compose -f docker-compose.prod.yml logs -f app
```

#### 详细部署文档

请参考 [DEPLOY.md](DEPLOY.md) 获取完整的部署指南，包括：

- 服务器准备和 Docker 安装
- SSL 证书配置（Let's Encrypt / 自签名）
- 防火墙配置
- 生产环境优化
- 故障排查
- 监控和备份

#### 服务访问

- **HTTP**: `http://your-server-ip`
- **HTTPS**: `https://your-domain.com` (需配置 SSL)
- **API 文档**: `http://your-server-ip/docs`
- **健康检查**: `http://your-server-ip/health`

### 注意事项

- Chroma 数据库持久化存储在 `data/vectorstore/<kb>`
- 摄取支持常见文本/代码文件；根据需要扩展加载器
- 对于私有 Git，在 URL 中使用令牌或提供 `username` + `token`
