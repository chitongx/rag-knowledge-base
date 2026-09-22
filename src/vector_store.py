"""向量库：基于原生 FAISS（faiss-cpu）封装。

- 使用 IndexFlatIP（内积 + 归一化向量 = 余弦相似度），精确检索，无 ANN 召回损失；
- 支持按 source / heading 过滤（metadata 过滤），对应 Nova 里"按 Project 隔离"的思路；
- 索引与分块元数据落盘到 data/index/，重建可复现。
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import faiss
import numpy as np

from .config import INDEX_DIR
from .embeddings import EmbeddingModel
from .ingestion import Chunk


class FAISSVectorStore:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatIP(dim)
        self.chunks: list[Chunk] = []  # 顺序与 index 中向量一一对应

    # ---------- 写入 ----------
    def add_chunks(self, chunks: list[Chunk], embedder: EmbeddingModel):
        if not chunks:
            return
        vecs = np.asarray(
            embedder.encode_documents([c.text for c in chunks]), dtype="float32"
        )
        self.index.add(vecs)
        for i, c in enumerate(chunks):
            c.chunk_id = f"c{len(self.chunks) + i}"
        self.chunks.extend(chunks)

    # ---------- 检索 ----------
    def search(self, query_vec: list[float], top_k: int = 10,
               source_filter: str | None = None) -> list[tuple[Chunk, float]]:
        q = np.asarray([query_vec], dtype="float32")
        scores, ids = self.index.search(q, top_k * 4)  # 多召回用于过滤
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            c = self.chunks[int(idx)]
            if source_filter and c.source != source_filter:
                continue
            results.append((c, float(score)))
            if len(results) >= top_k:
                break
        return results

    # ---------- 持久化 ----------
    def save(self, path: Path = INDEX_DIR / "faiss"):
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / "index.bin"))
        with open(path / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)
        with open(path / "meta.json", "w", encoding="utf-8") as f:
            json.dump({"dim": self.index.d, "count": len(self.chunks)}, f,
                      ensure_ascii=False, indent=2)

    @staticmethod
    def load(path: Path = INDEX_DIR / "faiss") -> "FAISSVectorStore":
        store = FAISSVectorStore.__new__(FAISSVectorStore)
        store.index = faiss.read_index(str(path / "index.bin"))
        with open(path / "chunks.pkl", "rb") as f:
            store.chunks = pickle.load(f)
        return store

    def __len__(self):
        return len(self.chunks)
