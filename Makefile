.PHONY: run test clean shell

# 开发运行
# --reload-dir 限定监听代码目录，避免 data/ 下导入文件触发重载中断请求
run:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8090 --reload-dir app --reload-dir static

# 运行测试
test:
	python -m pytest tests/ -v

# 清除 __pycache__
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Python 交互式 shell
shell:
	python -c "import app; print('AI RAG Knowledge — app package loaded')"

# Docker 构建
docker-build:
	docker build -t ai-rag-app:latest .

# Docker 运行
docker-run:
	docker run -d --name ai-rag -p 8090:8090 ai-rag-app:latest
