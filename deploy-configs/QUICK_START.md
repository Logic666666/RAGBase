# 快速部署指南

## 准备工作

1. **云服务器要求**：
   - Docker 20.10+
   - Docker Compose 2.0+
   - 最低4GB内存（推荐8GB+）

2. **上传配置文件**：
   ```bash
   # 将整个 deploy-configs 目录上传到云服务器
   scp -r deploy-configs user@your-server:/path/to/
   ```

## 部署步骤

### 1. 进入部署目录
```bash
cd /path/to/deploy-configs
```

### 2. 配置环境变量
```bash
# 复制环境变量模板
cp env.example .env

# 编辑配置文件（必须修改以下参数）
nano .env
```

**必须修改的参数**：
- `DOCKER_REGISTRY`: 你的镜像仓库地址
- `DOCKER_NAMESPACE`: 镜像命名空间
- `EMBEDDING_MODEL`: 嵌入模型（如 deepseek-r1:1.5b）
- `CHAT_MODEL`: 对话模型（如 deepseek-r1:1.5b）

### 3. 运行部署脚本
```bash
# 给脚本执行权限
chmod +x deploy.sh

# 一键部署
./deploy.sh
```

## SSL证书配置（可选）

### HTTP模式（默认）
无需额外配置，直接使用。

### HTTPS模式
1. 编辑 `.env` 文件，设置 `USE_SSL=true`
2. 选择证书方式：

**方式1：自签名证书（测试用）**
```bash
cd nginx
bash generate-self-signed-cert.sh
cd ..
```

**方式2：Let's Encrypt证书（生产用）**
```bash
# 申请证书
sudo certbot certonly --standalone -d your-domain.com

# 复制证书
sudo cp /etc/letsencrypt/live/your-domain.com/fullchain.pem nginx/ssl/cert.pem
sudo cp /etc/letsencrypt/live/your-domain.com/privkey.pem nginx/ssl/key.pem
```

## 验证部署

部署完成后，访问：
- HTTP: `http://your-server-ip`
- HTTPS: `https://your-server-ip`（如果配置了SSL）

## 管理命令

```bash
# 查看服务状态
docker compose ps

# 查看日志
docker compose logs -f

# 停止服务
docker compose down

# 重启服务
docker compose restart
```

## 故障排查

1. **服务无法启动**：检查端口是否被占用
2. **健康检查失败**：查看应用日志 `docker compose logs app`
3. **SSL证书问题**：确认证书文件路径和权限
4. **模型下载慢**：首次启动需要下载AI模型，请耐心等待