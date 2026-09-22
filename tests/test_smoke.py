"""冒烟测试：验证核心链路可运行（不依赖 LLM）。

运行：python -m pytest tests/ -q  或  python tests/test_smoke.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DOCS_DIR
from src.ingestion import chunk_document
from src.retrieval import retrieve


def test_ingestion():
    docs = list(DOCS_DIR.glob("*.md"))
    assert docs, "data/docs 下没有示例文档"
    chunks = [c for d in docs for c in chunk_document(d)]
    assert len(chunks) >= 8, f"分块数量过少: {len(chunks)}"


def test_retrieve():
    docs = retrieve("什么是SPD模式？", mode="hybrid_rerank", top_k=3)
    assert docs, "检索结果为空"
    assert docs[0].source.endswith("02_医院物资管理流程.md"), \
        f"期望命中物资管理文档，实际 {docs[0].source}"
    assert all(d.rank == i + 1 for i, d in enumerate(docs))


if __name__ == "__main__":
    test_ingestion()
    print("ingestion OK")
    test_retrieve()
    print("retrieve OK")
    print("冒烟测试全部通过 ✅")
