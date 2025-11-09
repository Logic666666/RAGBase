# 部署配置文件清单

## 必需文件

### Docker Compose 配置
- `docker-compose.yml` - 基础服务配置
- `docker-compose.prod.yml` - 生产环境优化配置

### 环境变量
- `env.example` - 环境变量模板
- `.env` - 实际使用的配置文件（从env.example复制）

### Nginx 配置
- `nginx/nginx.conf` - Nginx主配置文件
- `nginx/nginx-http.conf` - HTTP配置
- `nginx/nginx-https.conf` - HTTPS配置
- `nginx/entrypoint.sh` - 容器启动脚本
- `nginx/ssl/` - SSL证书目录（可选）

### 部署脚本
- `deploy.sh` - 一键部署脚本
- `nginx/generate-self-signed-cert.sh` - 自签名证书生成脚本

## 文件说明

### 1. docker-compose.yml
定义了三个服务：
- **ollama**: AI模型服务
- **app**: FastAPI应用服务
- **nginx**: 反向代理服务

### 2. .env文件
需要配置的关键参数：
- `DOCKER_REGISTRY`: 镜像仓库地址
- `DOCKER_NAMESPACE`: 镜像命名空间
- `DOCKER_IMAGE_NAME`: 镜像名称
- `EMBEDDING_MODEL`: 嵌入模型
- `CHAT_MODEL`: 对话模型
- `HTTP_PORT`: HTTP端口
- `HTTPS_PORT`: HTTPS端口
- `USE_SSL`: 是否启用HTTPS

### 3. Nginx配置
- 支持HTTP和HTTPS两种模式
- 自动根据USE_SSL环境变量切换
- 包含静态文件服务和反向代理

## 部署步骤

1. 上传所有配置文件到云服务器
2. 复制并编辑.env文件：`cp env.example .env`
3. 运行部署脚本：`chmod +x deploy.sh && ./deploy.sh`

## SSL证书配置

### 选项1：自签名证书（测试用）
```bash
cd nginx
bash generate-self-signed-cert.sh
cd ..
```

### 选项2：Let's Encrypt证书（生产用）
```bash
# 申请证书
sudo certbot certonly --standalone -d your-domain.com

# 复制到项目目录
sudo cp /etc/letsencrypt/live/your-domain.com/fullchain.pem nginx/ssl/cert.pem
sudo cp /etc/letsencrypt/live/your-domain.com/privkey.pem nginx/ssl/key.pem