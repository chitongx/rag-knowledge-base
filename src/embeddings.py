"""Embedding 封装：sentence-transformers 加载 BGE 中文向量模型（CPU 可跑）。

BGE 检索的官方要求：query 需要加指令前缀，文档侧不加。本封装在
embed_query 里自动加前缀，embed_documents 不加。
"""
from __future__ import annotations

from functools import lru_cache

from .config import EMBEDDING_MODEL, QUERY_INSTRUCTION


class EmbeddingModel:
    def __init__(self, model_name: str = EMBEDDING_MODEL, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._model = None

    def _load(self):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        model = self._model or self._load()
        vecs = model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False, batch_size=32
        )
        return vecs.tolist()

    def encode_query(self, query: str) -> list[float]:
        model = self._model or self._load()
        vecs = model.encode(
            [QUERY_INSTRUCTION + query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vecs[0].tolist()


@lru_cache(maxsize=1)
def get_embedding_model() -> EmbeddingModel:
    return EmbeddingModel()
