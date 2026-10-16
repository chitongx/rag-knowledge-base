"""Real temporary FAISS/BM25 artifacts must follow the current document set."""
import hashlib
import json
from unittest.mock import patch

import pytest

from scripts import build_index
from src import retrieval


class TinyEmbedder:
    def encode_documents(self, texts):
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]


@pytest.fixture
def index_files(tmp_path):
    docs = tmp_path / "docs"
    index = tmp_path / "index"
    docs.mkdir()
    (docs / "first.md").write_text("# 第一篇\n\n原始内容", encoding="utf-8")
    (docs / "second.txt").write_text("第二篇内容", encoding="utf-8")
    (docs / "ignored.log").write_text("不支持的文件类型", encoding="utf-8")
    with patch.object(build_index, "DOCS_DIR", docs), \
            patch.object(build_index, "INDEX_DIR", index), \
            patch.object(retrieval, "INDEX_DIR", index), \
            patch.object(build_index, "get_embedding_model", return_value=TinyEmbedder()):
        yield docs, index


def test_builder_saves_only_supported_document_fingerprints(index_files):
    docs, index = index_files
    build_index.main()
    manifest = index / "documents.sha256.json"
    assert manifest.exists()
    assert json.loads(manifest.read_text(encoding="utf-8")) == {
        name: hashlib.sha256((docs / name).read_bytes()).hexdigest()
        for name in ("first.md", "second.txt")
    }


@pytest.mark.parametrize("change", ["add", "modify", "delete", "missing_manifest", "invalid_manifest"])
def test_loader_rebuilds_stale_index_using_actual_documents(index_files, change):
    docs, index = index_files
    build_index.main()
    if change == "add":
        (docs / "third.md").write_text("# 新增资料\n\n新增内容", encoding="utf-8")
    elif change == "modify":
        (docs / "first.md").write_text("# 第一篇\n\n更新后的内容", encoding="utf-8")
    elif change == "delete":
        (docs / "second.txt").unlink()
    elif change == "missing_manifest":
        (index / "documents.sha256.json").unlink(missing_ok=True)
    else:
        (index / "documents.sha256.json").write_text("{invalid", encoding="utf-8")
    with patch.object(build_index, "main", wraps=build_index.main) as build:
        store, bm25 = retrieval.load_index()
    assert build.call_count == 1
    assert len(store.chunks) == len(bm25.corpus)
    sources = {chunk.source for chunk in store.chunks}
    assert sources == {p.name for p in docs.iterdir() if p.suffix in (".md", ".txt")}
    if change == "modify":
        assert any("更新后的内容" in chunk.text for chunk in store.chunks)


def test_loader_does_not_rebuild_when_only_unsupported_file_changes(index_files):
    docs, _ = index_files
    build_index.main()
    (docs / "ignored.log").write_text("更新但仍不支持", encoding="utf-8")
    with patch.object(build_index, "main", wraps=build_index.main) as build:
        store, _ = retrieval.load_index()
    assert build.call_count == 0
    assert len(store.chunks) == 2


def test_deleting_all_documents_does_not_return_old_index(index_files):
    docs, _ = index_files
    build_index.main()
    (docs / "first.md").unlink()
    (docs / "second.txt").unlink()
    with pytest.raises(ValueError, match="没有找到可解析的文档"):
        retrieval.load_index()
