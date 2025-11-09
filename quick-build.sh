#!/bin/bash

# 快速构建和推送脚本（简化版）
set -e

# 从环境变量读取配置（如果没有则使用默认值）
DOCKER_REGISTRY=${DOCKER_REGISTRY:-your-registry.com}
DOCKER_NAMESPACE=${DOCKER_NAMESPACE:-your-project}
DOCKER_IMAGE_NAME=${DOCKER_IMAGE_NAME:-ai-rag-app}
DOCKER_IMAGE_TAG=${DOCKER_IMAGE_TAG:-latest}

IMAGE_NAME="${DOCKER_REGISTRY}/${DOCKER_NAMESPACE}/${DOCKER_IMAGE_NAME}"
VERSION="${DOCKER_IMAGE_TAG}"

echo "🚀 开始构建镜像..."
docker build -t ${IMAGE_NAME}:${VERSION} .
docker tag ${IMAGE_NAME}:${VERSION} ${IMAGE_NAME}:latest

echo "📤 推送到仓库..."
docker login ${DOCKER_REGISTRY}
docker push ${IMAGE_NAME}:${VERSION}
docker push ${IMAGE_NAME}:latest

echo "✅ 完成！镜像地址: ${IMAGE_NAME}:${VERSION}"