# 生产级 RAG 知识库 - Docker 镜像
# 用法：
#   docker compose up -d                      # 一键启动（推荐，见 docker-compose.yml）
#   docker build -t rag-knowledge-base .      # 或手动构建
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.hf

WORKDIR /app

# 先装依赖（利用 Docker 层缓存）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 拷贝源码、示例语料与已构建索引（开箱即用，也可自行重建）
COPY src ./src
COPY scripts ./scripts
COPY data ./data
COPY eval_data ./eval_data

# 非 root 运行（安全加固）
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=150s --retries=3 \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["python", "-m", "scripts.serve"]
