"""全局配置：模型、路径、检索参数。所有可调项集中在这里，便于复现实验。"""
import os
from pathlib import Path

# ---------- 路径 ----------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DOCS_DIR = DATA_DIR / "docs"          # 原始文档（PDF/DOCX/TXT/MD）
INDEX_DIR = DATA_DIR / "index"        # 构建产物（向量索引 + BM25 + 分块元数据）
EVAL_DIR = BASE_DIR / "eval_data"     # 离线评估测试集

# ---------- 向量模型 / 重排序（HuggingFace 自动下载） ----------
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")
# BGE 系列检索 query 前缀（官方要求，不加会掉分）
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："

# ---------- 分块 ----------
CHUNK_SIZE = 400          # 中文按字符计
CHUNK_OVERLAP = 50
CHUNK_SEPARATORS = ["\n\n", "\n", "。", "；", "，", " ", ""]

# ---------- 检索 ----------
VECTOR_TOP_K = 10         # 向量路召回数
BM25_TOP_K = 10           # BM25 路召回数
RRF_K = 60                # RRF 融合常数（论文推荐 60）
HYBRID_FINAL_K = 10       # 混合后取 top-k
RERANK_FINAL_K = 5        # 重排后最终取 top-k

# ---------- LLM（问答生成；纯检索演示可不填） ----------
# 支持任何 OpenAI 兼容 API：DeepSeek / 通义千问 / OpenAI / Ollama(localhost:11434/v1)
LLM_API_BASE = os.getenv("LLM_API_BASE", "https://api.deepseek.com/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")
LLM_TEMPERATURE = 0.3

# ---------- 服务 ----------
SERVE_HOST = os.getenv("HOST", "0.0.0.0")
# 端口支持环境变量覆盖：HuggingFace Spaces 要求监听 7860，本地默认 8000
SERVE_PORT = int(os.getenv("PORT", "8000"))

# ---------- 安全（公网部署建议开启） ----------
# 设置 API_KEY 后，/retrieve /chat 等业务接口需携带 Authorization: Bearer <key>；
# 留空则开放访问（适合本地演示）。演示页支持在页面右上角填入 Key。
API_KEY = os.getenv("API_KEY", "")
