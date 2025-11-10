#!/bin/sh
# Nginx 容器入口脚本

# 根据环境变量选择配置文件
if [ "${USE_SSL}" = "true" ]; then
    echo "启用 HTTPS 配置"
    # 检查 SSL 证书是否存在
    if [ -f /etc/nginx/ssl/cert.pem ] && [ -f /etc/nginx/ssl/key.pem ]; then
        # 替换 HTTP 配置为 HTTPS 配置
        sed -i 's|include /etc/nginx/nginx-http.conf;|include /etc/nginx/nginx-https.conf;|' /etc/nginx/nginx.conf
        echo "HTTPS 配置已启用"
    else
        echo "警告: SSL 证书不存在，使用 HTTP 配置"
    fi
else
    echo "使用 HTTP 配置"
fi

# 删除 nginx 所有可能的默认欢迎页面文件
rm -f /usr/share/nginx/html/index.html 2>/dev/null || true
rm -f /var/www/html/index.html 2>/dev/null || true
rm -f /var/www/index.html 2>/dev/null || true
echo "已清理所有默认欢迎页面文件"

# 确保 nginx 使用我们的自定义配置
# 删除所有可能的默认服务器配置
rm -f /etc/nginx/conf.d/*.conf 2>/dev/null || true
echo "已删除所有默认服务器配置"

# 确保我们的配置是唯一有效的配置
# 复制我们的配置文件到 conf.d 目录
cp /etc/nginx/nginx-http.conf /etc/nginx/conf.d/custom.conf 2>/dev/null || true

# 简化配置：直接使用 Docker 内部 DNS，让 nginx 处理连接重试
echo "应用服务配置：使用 Docker 内部 DNS 解析"
echo "nginx 已配置连接超时和重试机制，无需额外等待"

# 确保静态文件目录可读（虽然不在nginx容器中，但保持兼容性）
chmod -R +r /app/static 2>/dev/null || true
echo "已设置静态文件目录权限（兼容性）"

# 测试 nginx 配置
if nginx -t; then
    echo "nginx 配置测试成功"
else
    echo "nginx 配置测试失败，请检查配置"
    exit 1
fi

# 启动 nginx
exec nginx -g 'daemon off;'