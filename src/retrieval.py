"""检索管线总装：向量 / 混合 / 混合+Rerank 三种模式，统一返回带来源引用的结果。

对外暴露：
- retrieve(query, mode)         检索，返回 [RetrievedDoc]
- build_retrieval_components()  加载索引（供问答与 API 复用）

RetrievedDoc 携带 source/heading，供前端展示"引用来源"（面试要点之一）。
"""
from __future__ import annotations

from dataclasses import dataclass

from .config import HYBRID_FINAL_K, INDEX_DIR, RERANK_FINAL_K
from .embeddings import get_embedding_model
from .hybrid_search import BM25Index, hybrid_retrieve
from .reranker import get_reranker
from .vector_store import FAISSVectorStore


@dataclass
class RetrievedDoc:
    text: str
    source: str
    heading: str
    score: float
    rank: int


def load_index() -> tuple[FAISSVectorStore, BM25Index]:
    faiss_path = INDEX_DIR / "faiss"
    bm25_path = INDEX_DIR / "bm25"
    # 云端（Streamlit Cloud）无持久磁盘，索引二进制不入库；启动时检测到缺失则自动从 data/docs 重建
    if not (faiss_path / "index.bin").exists() or not (bm25_path / "bm25.pkl").exists():
        print("[auto-build] 索引不存在，开始从 data/docs/ 自动重建（首次会下载 embedding 模型）...")
        from scripts.build_index import main as build_main
        build_main()
    store = FAISSVectorStore.load(faiss_path)
    bm25 = BM25Index.load(bm25_path)
    return store, bm25


def retrieve(query: str, mode: str = "hybrid_rerank",
             top_k: int | None = None, source_filter: str | None = None) -> list[RetrievedDoc]:
    """mode:
      - vector         纯向量（baseline）
      - hybrid         向量 + BM25 + RRF
      - hybrid_rerank  混合 + Cross-Encoder 精排（默认，生产推荐）
    """
    store, bm25 = load_index()
    embedder = get_embedding_model()
    k = top_k or RERANK_FINAL_K

    if mode == "vector":
        q_vec = embedder.encode_query(query)
        hits = store.search(q_vec, top_k=max(k, 10), source_filter=source_filter)
        ranked = [(c, s) for c, s in hits[:k]]
    else:
        candidates = hybrid_retrieve(
            store, bm25, embedder, query,
            final_k=max(HYBRID_FINAL_K, k), source_filter=source_filter,
        )
        if mode == "hybrid":
            ranked = [(c, s) for c, s in candidates[:k]]
        else:  # hybrid_rerank
            reranker = get_reranker()
            texts = [c.text for c, _ in candidates]
            reranked = reranker.rerank(query, texts, top_k=k)
            ranked = [(candidates[i][0], s) for i, s in reranked]

    return [
        RetrievedDoc(text=c.text, source=c.source, heading=c.heading_path,
                     score=round(s, 4), rank=i + 1)
        for i, (c, s) in enumerate(ranked)
    ]
