#!/bin/bash

# RAG项目Docker镜像构建和推送脚本
set -e

# 镜像信息
# 从环境变量读取配置（如果没有则使用默认值）
DOCKER_REGISTRY=${DOCKER_REGISTRY:-your-registry.com}
DOCKER_NAMESPACE=${DOCKER_NAMESPACE:-your-project}
DOCKER_IMAGE_NAME=${DOCKER_IMAGE_NAME:-ai-rag-app}
DOCKER_IMAGE_TAG=${DOCKER_IMAGE_TAG:-latest}

IMAGE_NAME="${DOCKER_REGISTRY}/${DOCKER_NAMESPACE}/${DOCKER_IMAGE_NAME}"
VERSION="${DOCKER_IMAGE_TAG}"
FULL_IMAGE="${IMAGE_NAME}:${VERSION}"

echo "🚀 开始构建RAG项目Docker镜像..."

# 1. 清理旧的构建缓存
echo "🧹 清理构建缓存..."
docker builder prune -f

# 2. 构建镜像
echo "🔨 构建镜像 ${FULL_IMAGE}..."
docker build -t ${FULL_IMAGE} .

# 3. 本地测试
echo "🧪 本地测试..."
docker run -d --name test-rag -p 8090:8090 ${FULL_IMAGE}
sleep 10

# 健康检查
if curl -f http://localhost:8090/health; then
    echo "✅ 健康检查通过"
else
    echo "❌ 健康检查失败"
    docker logs test-rag
    docker stop test-rag && docker rm test-rag
    exit 1
fi

# 清理测试容器
docker stop test-rag && docker rm test-rag

# 4. 登录镜像仓库
echo "🔐 登录镜像仓库..."
docker login ${DOCKER_REGISTRY}

# 5. 推送镜像
echo "📤 推送镜像到仓库..."
docker push ${FULL_IMAGE}

# 6. 可选：推送latest标签
docker tag ${FULL_IMAGE} ${IMAGE_NAME}:latest
docker push ${IMAGE_NAME}:latest

echo "✅ 镜像构建和推送完成！"
echo "📋 镜像地址: ${FULL_IMAGE}"

# 7. 显示镜像信息
echo "📊 镜像信息:"
docker images ${IMAGE_NAME}