"""一键构建索引：解析 data/docs 全部文档 -> 分块 -> Embedding -> FAISS 向量库 + BM25 索引。

用法：python -m scripts.build_index
输出：data/index/{faiss,bm25}
"""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DOCS_DIR, INDEX_DIR
from src.embeddings import get_embedding_model
from src.hybrid_search import BM25Index
from src.ingestion import chunk_document
from src.vector_store import FAISSVectorStore


def document_fingerprints() -> dict[str, str]:
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(DOCS_DIR.glob("*"))
        if p.is_file() and p.suffix.lower() in (".pdf", ".docx", ".txt", ".md", ".markdown")
    }


def main():
    fingerprints = document_fingerprints()
    docs = [DOCS_DIR / name for name in fingerprints]
    if not docs:
        raise ValueError(f"没有找到可解析的文档，请先放文件到 {DOCS_DIR}")

    all_chunks = []
    for path in docs:
        chunks = chunk_document(path)
        print(f"  [{path.name}] 分块 {len(chunks)} 个")
        all_chunks.extend(chunks)

    print(f"\n共 {len(all_chunks)} 个 chunk，开始向量化（首次会下载模型，较慢）...")
    embedder = get_embedding_model()

    # 小批量写入向量库，避免一次编码占用过多内存
    store = FAISSVectorStore(dim=embedder.encode_documents(["测试"])[0].__len__())
    BATCH = 64
    for i in range(0, len(all_chunks), BATCH):
        store.add_chunks(all_chunks[i:i + BATCH], embedder)
        print(f"  已向量化 {min(i + BATCH, len(all_chunks))}/{len(all_chunks)}")

    bm25 = BM25Index.from_chunks(all_chunks)
    store.save(INDEX_DIR / "faiss")
    bm25.save(INDEX_DIR / "bm25")
    (INDEX_DIR / "documents.sha256.json").write_text(
        json.dumps(fingerprints, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(f"\n索引构建完成：向量 {len(store)} 条，BM25 {len(bm25.corpus)} 条 -> {INDEX_DIR}")


if __name__ == "__main__":
    main()
