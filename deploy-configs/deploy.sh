#!/bin/bash
# 云服务器部署脚本

set -e

echo "=========================================="
echo "AI RAG Knowledge Base 部署脚本"
echo "=========================================="

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查 Docker 是否安装
if ! command -v docker &> /dev/null; then
    echo -e "${RED}错误: Docker 未安装，请先安装 Docker${NC}"
    exit 1
fi

# 检查 Docker Compose 是否安装
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo -e "${RED}错误: Docker Compose 未安装，请先安装 Docker Compose${NC}"
    exit 1
fi

# 检查 .env 文件
if [ ! -f .env ]; then
    echo -e "${YELLOW}警告: .env 文件不存在，从 env.example 创建${NC}"
    if [ -f env.example ]; then
        cp env.example .env
    else
        echo -e "${RED}错误: env.example 文件不存在${NC}"
        exit 1
    fi
    echo -e "${GREEN}请编辑 .env 文件配置相关参数${NC}"
fi

# 检查 SSL 证书（如果启用 HTTPS）
USE_SSL=${USE_SSL:-false}
if [ "$USE_SSL" = "true" ]; then
    if [ ! -f nginx/ssl/cert.pem ] || [ ! -f nginx/ssl/key.pem ]; then
        echo -e "${YELLOW}警告: SSL 证书不存在${NC}"
        echo -e "${YELLOW}选项 1: 使用 Let's Encrypt 生成证书（推荐）${NC}"
        echo -e "${YELLOW}选项 2: 使用自签名证书（仅用于测试）${NC}"
        read -p "是否生成自签名证书？(y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            cd nginx
            bash generate-self-signed-cert.sh
            cd ..
        else
            echo -e "${YELLOW}请手动配置 SSL 证书到 nginx/ssl/ 目录${NC}"
            exit 1
        fi
    fi
fi

# 拉取最新镜像
echo -e "${GREEN}拉取 Docker 镜像...${NC}"
docker compose pull

# 构建应用镜像
echo -e "${GREEN}构建应用镜像...${NC}"
docker compose build --no-cache

# 停止现有服务
echo -e "${GREEN}停止现有服务...${NC}"
docker compose down

# 启动服务
echo -e "${GREEN}启动服务...${NC}"
if [ "$USE_SSL" = "true" ]; then
    docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
else
    # 使用 HTTP 配置
    docker compose up -d
fi

# 等待服务启动
echo -e "${GREEN}等待服务启动...${NC}"
sleep 10

# 检查服务状态
echo -e "${GREEN}检查服务状态...${NC}"
docker compose ps

# 检查健康状态
echo -e "${GREEN}检查应用健康状态...${NC}"
sleep 5
if curl -f http://localhost/health > /dev/null 2>&1; then
    echo -e "${GREEN}✓ 应用健康检查通过${NC}"
else
    echo -e "${YELLOW}警告: 应用健康检查失败，请检查日志${NC}"
    docker compose logs app
fi

# 显示访问信息
echo ""
echo -e "${GREEN}=========================================="
echo "部署完成！"
echo "==========================================${NC}"
echo ""
echo "服务访问地址:"
if [ "$USE_SSL" = "true" ]; then
    echo -e "  - HTTPS: ${GREEN}https://localhost${NC}"
    echo -e "  - HTTP:  ${GREEN}http://localhost (自动重定向到 HTTPS)${NC}"
else
    echo -e "  - HTTP:  ${GREEN}http://localhost${NC}"
fi
echo ""
echo "管理命令:"
echo "  - 查看日志: docker compose logs -f"
echo "  - 停止服务: docker compose down"
echo "  - 重启服务: docker compose restart"
echo "  - 查看状态: docker compose ps"
echo ""
echo -e "${YELLOW}注意: 首次运行需要下载 Ollama 模型，可能需要一些时间${NC}"
echo ""