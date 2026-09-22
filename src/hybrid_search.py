"""混合检索：向量（稠密） + BM25（稀疏） + RRF 融合。

面试可讲的三点：
1. 纯向量检索对专有名词/编号/缩写不敏感（如"HRP""SPD""科室成本"），BM25 精确命中补位；
2. RRF（Reciprocal Rank Fusion）只按排名融合，不依赖两路分数尺度一致，免调参；
3. 两路各召回 top-k，RRF 融合后取前 N，再交给 Rerank 精排。
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import jieba
from rank_bm25 import BM25Okapi

from .config import BM25_TOP_K, INDEX_DIR, RRF_K, VECTOR_TOP_K
from .ingestion import Chunk
from .vector_store import FAISSVectorStore


class BM25Index:
    """基于 jieba 分词 + rank_bm25 的关键词检索索引。"""

    def __init__(self, corpus: list[str]):
        tokenized = [list(jieba.cut(doc)) for doc in corpus]
        self.bm25 = BM25Okapi(tokenized)
        self.corpus = corpus

    @classmethod
    def from_chunks(cls, chunks: list[Chunk]) -> "BM25Index":
        return cls([c.text for c in chunks])

    def search(self, query: str, top_k: int = BM25_TOP_K,
               source_filter: str | None = None) -> list[tuple[int, float]]:
        q_tokens = list(jieba.cut(query))
        scores = self.bm25.get_scores(q_tokens)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        results = []
        for i in order:
            if source_filter and self.corpus[i].startswith(source_filter):
                continue
            results.append((i, float(scores[i])))
            if len(results) >= top_k:
                break
        return results

    def save(self, path: Path = INDEX_DIR / "bm25"):
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "bm25.pkl", "wb") as f:
            pickle.dump({"corpus": self.corpus}, f)

    @staticmethod
    def load(path: Path = INDEX_DIR / "bm25") -> "BM25Index":
        with open(path / "bm25.pkl", "rb") as f:
            data = pickle.load(f)
        return BM25Index(data["corpus"])


def rrf_merge(vector_hits: list[tuple[int, float]],
              bm25_hits: list[tuple[int, float]],
              k: int = RRF_K) -> list[int]:
    """RRF 融合：score = Σ 1/(k + rank)。返回按融合分降序的 chunk 下标列表。"""
    fused: dict[int, float] = {}
    for rank, (idx, _score) in enumerate(vector_hits):
        fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank + 1)
    for rank, (idx, _score) in enumerate(bm25_hits):
        fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(fused, key=fused.get, reverse=True)


def hybrid_retrieve(
    store: FAISSVectorStore,
    bm25: BM25Index,
    embedder,
    query: str,
    final_k: int = 10,
    source_filter: str | None = None,
) -> list[tuple[Chunk, float]]:
    """向量 + BM25 两路召回 -> RRF 融合 -> 返回最终 top-k Chunk 列表。

    返回的 score 为 RRF 融合分（非相似度），仅用于排序。
    """
    # 1) 向量路
    q_vec = embedder.encode_query(query)
    vec_hits = store.search(q_vec, top_k=VECTOR_TOP_K, source_filter=source_filter)
    vec_ranked = [(store.chunks.index(c), s) for c, s in vec_hits]

    # 2) BM25 路
    bm25_hits = bm25.search(query, top_k=BM25_TOP_K, source_filter=source_filter)
    # 3) RRF 融合
    fused_ids = rrf_merge(vec_ranked, bm25_hits)

    results = []
    for idx in fused_ids:
        if source_filter and store.chunks[idx].source != source_filter:
            continue
        results.append((store.chunks[idx], 0.0))
        if len(results) >= final_k:
            break
    return results
