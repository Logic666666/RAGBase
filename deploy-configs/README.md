# 部署配置文件包

这个文件夹包含了部署到云服务器所需的所有配置文件，与Docker镜像分开管理。

## 文件结构

```
deploy-configs/
├── docker-compose.yml          # Docker Compose主配置文件
├── docker-compose.prod.yml     # 生产环境配置
├── .env                       # 环境变量配置（需要自定义）
├── nginx/                     # Nginx配置文件
│   ├── nginx-http.conf       # HTTP配置
│   ├── nginx-https.conf      # HTTPS配置
│   ├── nginx.conf            # 主配置
│   ├── entrypoint.sh         # 容器启动脚本
│   └── ssl/                  # SSL证书目录（可选）
├── deploy.sh                 # 部署脚本
├── deploy-configs.md         # 本说明文档
└── FILE_LIST.md             # 文件清单
```

## 使用说明

1. 将这些配置文件上传到云服务器
2. 编辑 `.env` 文件，配置您的参数
3. 运行 `./deploy.sh` 开始部署

## 配置文件说明

### Docker Compose文件
- `docker-compose.yml`: 基础服务配置
- `docker-compose.prod.yml`: 生产环境优化配置

### Nginx配置
- 支持HTTP和HTTPS两种模式
- 自动根据环境变量切换配置
- 包含SSL证书支持

### 环境变量
- 从 `env.example` 复制并重命名为 `.env`
- 配置数据库、端口、SSL等参数

## 与镜像的关系

- **Docker镜像**：包含应用代码和运行环境
- **配置文件**：包含服务编排和个性化设置
- **部署时**：两者结合，镜像提供运行环境，配置文件提供运行参数