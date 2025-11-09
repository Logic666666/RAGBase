# 云服务器部署指南

本文档详细说明如何将 AI RAG Knowledge Base 项目部署到云服务器上。

## 前置要求

### 服务器要求

- **操作系统**: Ubuntu 20.04+ / CentOS 7+ / Debian 10+
- **内存**: 最低 4GB RAM（推荐 8GB+）
- **CPU**: 2 核心以上（推荐 4 核心+）
- **磁盘**: 至少 20GB 可用空间
- **网络**: 能够访问互联网，开放端口 80 和 443（或自定义端口）

### 软件要求

- Docker 20.10+
- Docker Compose 2.0+
- Git（用于克隆项目）

## 部署步骤

### 1. 服务器准备

#### 安装 Docker

**Ubuntu/Debian:**

```bash
# 更新系统
sudo apt update && sudo apt upgrade -y

# 安装 Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# 安装 Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# 将当前用户添加到 docker 组
sudo usermod -aG docker $USER
newgrp docker
```

**CentOS/RHEL:**

```bash
# 安装 Docker
sudo yum install -y docker
sudo systemctl start docker
sudo systemctl enable docker

# 安装 Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

### 2. 获取项目代码

```bash
# 克隆项目（如果使用 Git）
git clone <your-repo-url>
cd ai-rag-knowledge

# 或者上传项目文件到服务器
```

### 3. 配置环境变量

```bash
# 复制环境变量模板
cp env.example .env

# 编辑环境变量文件
nano .env
```

主要配置项：

```env
# Ollama 配置
OLLAMA_BASE_URL=http://ollama:11434

# 模型配置
EMBEDDING_MODEL=deepseek-r1:1.5b
CHAT_MODEL=deepseek-r1:1.5b

# Nginx 端口配置
HTTP_PORT=80
HTTPS_PORT=443

# SSL 配置
USE_SSL=false  # 设置为 true 启用 HTTPS
```

### 4. SSL 证书配置（可选，推荐生产环境使用）

#### 方案 1: 使用 Let's Encrypt（推荐）

```bash
# 安装 certbot
sudo apt install certbot -y

# 申请证书（需要域名）
sudo certbot certonly --standalone -d your-domain.com

# 复制证书到项目目录
sudo mkdir -p nginx/ssl
sudo cp /etc/letsencrypt/live/your-domain.com/fullchain.pem nginx/ssl/cert.pem
sudo cp /etc/letsencrypt/live/your-domain.com/privkey.pem nginx/ssl/key.pem
sudo chmod 644 nginx/ssl/cert.pem
sudo chmod 600 nginx/ssl/key.pem
```

#### 方案 2: 使用自签名证书（仅用于测试）

```bash
cd nginx
bash generate-self-signed-cert.sh
cd ..
```

### 5. 部署应用

#### 方式 1: 使用部署脚本（推荐）

```bash
# Linux/Mac
chmod +x deploy.sh
./deploy.sh
```

#### 方式 2: 手动部署

```bash
# 构建镜像
docker compose build

# 启动服务（开发环境）
docker compose up -d

# 或启动服务（生产环境，使用生产配置）
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### 6. 验证部署

```bash
# 检查服务状态
docker compose ps

# 查看日志
docker compose logs -f

# 检查健康状态
curl http://localhost/health
```

### 7. 配置防火墙

```bash
# Ubuntu/Debian (UFW)
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable

# CentOS/RHEL (firewalld)
sudo firewall-cmd --permanent --add-service=http
sudo firewall-cmd --permanent --add-service=https
sudo firewall-cmd --reload
```

## 生产环境优化

### 1. 资源限制

编辑 `docker-compose.prod.yml` 调整资源限制：

```yaml
services:
  app:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2G
```

### 2. 日志管理

```bash
# 配置日志轮转
docker compose logs --tail=100 -f app

# 或使用 Docker 日志驱动
# 在 docker-compose.yml 中添加:
logging:
  driver: "json-file"
  options:
    max-size: "10m"
    max-file: "3"
```

### 3. 数据备份

```bash
# 备份数据卷
docker run --rm -v ai-rag-knowledge_app_data:/data -v $(pwd):/backup alpine tar czf /backup/app_data_backup.tar.gz /data
```

### 4. 监控和告警

建议配置监控系统（如 Prometheus + Grafana）监控：

- 容器资源使用情况
- API 响应时间
- 错误率
- Ollama 模型加载状态

## 常用管理命令

```bash
# 查看服务状态
docker compose ps

# 查看日志
docker compose logs -f app
docker compose logs -f nginx
docker compose logs -f ollama

# 重启服务
docker compose restart app

# 停止服务
docker compose down

# 更新服务
docker compose pull
docker compose up -d --build

# 进入容器
docker exec -it rag-app bash

# 清理未使用的资源
docker system prune -a
```

## 故障排查

### 1. 服务无法启动

```bash
# 查看详细日志
docker compose logs app
docker compose logs nginx
docker compose logs ollama

# 检查端口占用
sudo netstat -tulpn | grep :80
sudo netstat -tulpn | grep :443
```

### 2. Ollama 模型未加载

```bash
# 进入 Ollama 容器
docker exec -it rag-ollama bash

# 检查模型列表
ollama list

# 拉取模型
ollama pull deepseek-r1:1.5b
```

### 3. Nginx 502 错误

```bash
# 检查应用服务是否运行
docker compose ps app

# 检查应用健康状态
curl http://app:8090/health

# 检查 Nginx 配置
docker exec -it rag-nginx nginx -t
```

### 4. SSL 证书问题

```bash
# 检查证书文件
ls -la nginx/ssl/

# 验证证书
openssl x509 -in nginx/ssl/cert.pem -text -noout
```

## 安全建议

1. **使用 HTTPS**: 生产环境必须启用 HTTPS
2. **防火墙配置**: 只开放必要端口
3. **定期更新**: 定期更新 Docker 镜像和系统
4. **访问控制**: 配置 Nginx 访问限制（如 IP 白名单）
5. **数据加密**: 敏感数据加密存储
6. **日志审计**: 启用访问日志和错误日志

## 性能优化

1. **增加工作进程**: 在 `docker-compose.prod.yml` 中调整 worker 数量
2. **启用缓存**: 配置 Redis 缓存（如需要）
3. **CDN 加速**: 静态资源使用 CDN
4. **数据库优化**: 优化向量数据库查询性能

## 更新部署

```bash
# 1. 拉取最新代码
git pull

# 2. 备份数据
# （执行备份脚本）

# 3. 更新服务
docker compose pull
docker compose up -d --build

# 4. 验证服务
curl http://localhost/health
```

## 支持与帮助

如遇到问题，请查看：

- 项目 README.md
- Docker 日志: `docker compose logs`
- GitHub Issues
