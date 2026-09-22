# 生产级 RAG 知识库 - 可复现构建
# 用法：docker build -t rag-knowledge-base . && docker run -p 8000:8000 rag-knowledge-base
FROM python:3.12-slim

WORKDIR /app

# 先装依赖（利用 Docker 层缓存）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 拷贝源码、示例语料与已构建索引（开箱即用，也可自行重建）
COPY src ./src
COPY scripts ./scripts
COPY data ./data
COPY eval_data ./eval_data

EXPOSE 8000

CMD ["python", "-m", "scripts.serve"]
