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

# 启动 nginx
exec nginx -g 'daemon off;'