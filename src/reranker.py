"""Rerank 精排：Cross-Encoder（BGE-Reranker）对"问题+候选chunk"逐对打分。

为什么需要：向量相似度是粗排，双塔模型看不出 query 与 chunk 的细粒度交互；
Cross-Encoder 把 query 和 chunk 拼在一起过 Transformer，精度更高但慢，
所以只对混合检索召回的前 N 个候选精排。
"""
from __future__ import annotations

from functools import lru_cache

from .config import RERANK_MODEL


class Reranker:
    def __init__(self, model_name: str = RERANK_MODEL, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._model = None

    def _load(self):
        from FlagEmbedding import FlagReranker
        self._model = FlagReranker(self.model_name, use_fp16=False, device=self.device)
        return self._model

    def rerank(self, query: str, candidates: list[str], top_k: int = 5) -> list[tuple[int, float]]:
        """返回按相关度降序的 (candidate_idx, score) 列表，取前 top_k。"""
        if not candidates:
            return []
        model = self._model or self._load()
        pairs = [[query, c] for c in candidates]
        scores = model.compute_score(pairs, normalize=True)
        if isinstance(scores, float):
            scores = [scores]
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [(i, float(scores[i])) for i in order[:top_k]]


@lru_cache(maxsize=1)
def get_reranker() -> Reranker:
    return Reranker()
