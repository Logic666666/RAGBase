#!/bin/bash
# 生成自签名SSL证书（仅用于测试）

echo "生成自签名SSL证书..."

# 创建ssl目录
mkdir -p ssl

# 生成私钥
openssl genrsa -out ssl/key.pem 2048

# 生成证书签名请求
openssl req -new -key ssl/key.pem -out ssl/cert.csr -subj "/C=CN/ST=Beijing/L=Beijing/O=Test/CN=localhost"

# 生成自签名证书
openssl x509 -req -days 365 -in ssl/cert.csr -signkey ssl/key.pem -out ssl/cert.pem

# 删除证书签名请求
rm ssl/cert.csr

# 设置权限
chmod 600 ssl/key.pem
chmod 644 ssl/cert.pem

echo "自签名证书生成完成！"
echo "证书文件：ssl/cert.pem"
echo "私钥文件：ssl/key.pem"
echo "注意：自签名证书仅用于测试，生产环境请使用有效的SSL证书"