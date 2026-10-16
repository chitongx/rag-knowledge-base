"""离线评估：不依赖 LLM，只评估"检索"质量（面试里最有说服力的数字）。

判定口径（chunk 级）：chunk 来源符合 expected_sources，且文本包含 expected_key（原文子串）。
相比"文档级命中"，chunk 级判定更严格，能区分"检索到了对的文档但找错了段落"。

指标口径（简历/面试直接可讲）：
- Hit@K：top-K 结果中是否出现相关 chunk（0/1 平均）
- Recall@K：已召回相关 chunk 数 / 完整语料中的相关 chunk 数
- MRR：第一个相关 chunk 的排名倒数平均
- NDCG@K：二元相关性的对数折扣 DCG / 完整语料相关性得到的 IDCG

实验设计：vector / hybrid / hybrid_rerank 三种模式在同一测试集、同一 top-k 下对比，
自动输出相对 baseline（vector）的提升百分比，可直接用于简历结果论表述。
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from .config import EVAL_DIR
from .retrieval import retrieve


def _load_tests(path: Path | None = None) -> list[dict]:
    path = path or EVAL_DIR / "test_questions.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["questions"]


def _relevant(doc, q: dict) -> bool:
    """chunk 级相关判定：文本包含 expected_key；无 key 时退化为 source 匹配。"""
    sources = q.get("expected_sources", [])
    if sources and doc.source not in sources:
        return False
    key = q.get("expected_key", "")
    if key:
        return key in doc.text
    return doc.source in q.get("expected_sources", [])


# ---------------- 指标 ----------------
def hit_at_k(docs, q: dict, k: int) -> float:
    return 1.0 if any(_relevant(d, q) for d in docs[:k]) else 0.0


def recall_at_k(docs, q: dict, k: int, total_relevant: int) -> float:
    if not total_relevant:
        return 0.0
    return sum(_relevant(d, q) for d in docs[:k]) / total_relevant


def mrr(docs, q: dict) -> float:
    for rank, d in enumerate(docs, 1):
        if _relevant(d, q):
            return 1.0 / rank
    return 0.0


def ndcg_at_k(docs, q: dict, k: int, total_relevant: int) -> float:
    """二元相关性：rank i 的增益为 1/log2(i+1)，理想排序涵盖所有相关 chunk。"""
    dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(docs[:k]) if _relevant(d, q))
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(total_relevant, k)))
    return dcg / ideal if ideal else 0.0


def _experiment_metadata(k: int, chunk_count: int, questions_path: Path | None = None) -> dict:
    """记录公开模型、配置与版本；不读取或保存认证配置。"""
    import hashlib
    import platform
    from datetime import datetime, timezone
    from importlib.metadata import PackageNotFoundError, version
    from . import config

    packages = {}
    for name in ("langchain", "langchain-core", "langchain-openai", "langchain-text-splitters",
                 "sentence-transformers", "FlagEmbedding", "transformers", "torch",
                 "faiss-cpu", "rank-bm25", "jieba", "numpy", "streamlit", "fastapi"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    files = sorted(config.DOCS_DIR.glob("*")) + sorted((config.BASE_DIR / "src").glob("*.py"))
    files += [questions_path or EVAL_DIR / "test_questions.json", config.BASE_DIR / "requirements.txt", config.BASE_DIR / "scripts/build_index.py"]
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "top_k": k,
        "chunk_count": chunk_count,
        "models": {"embedding": config.EMBEDDING_MODEL, "reranker": config.RERANK_MODEL},
        "config": {name: getattr(config, name) for name in (
            "CHUNK_SIZE", "CHUNK_OVERLAP", "CHUNK_SEPARATORS", "QUERY_INSTRUCTION",
            "VECTOR_TOP_K", "BM25_TOP_K", "RRF_K", "HYBRID_FINAL_K", "RERANK_FINAL_K")},
        "sha256": {str(p.relative_to(config.BASE_DIR) if p.is_relative_to(config.BASE_DIR) else p): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in files if p.is_file()},
        "relevance": "expected_sources + expected_key substring; binary chunk-level labels",
        "ndcg_discount": "1/log2(rank+1), IDCG from full-corpus relevant chunk count",
    }


# ---------------- 实验 ----------------
def run_experiment(k: int = 5, modes: tuple[str, ...] = ("vector", "hybrid", "hybrid_rerank"),
                   *, questions_path: Path | None = None) -> dict:
    import time
    from .retrieval import load_index

    started = time.perf_counter()
    questions_path = (questions_path or EVAL_DIR / "test_questions.json").resolve()
    dataset = json.loads(questions_path.read_text(encoding="utf-8"))
    tests = dataset["questions"]
    categories = ("answerable", "unanswerable", "out_of_scope")
    counts = {category: sum(q.get("category", "answerable") == category for q in tests) for category in categories}
    if not counts["answerable"]:
        raise ValueError("评估集至少需要一个可回答问题")
    for q in tests:
        category = q.get("category", "answerable")
        if category not in categories:
            raise ValueError(f"未知问题分类：{category}")
        if category != "answerable" and (q.get("expected_key") or q.get("expected_sources")):
            raise ValueError("无答案或越界问题不能标注相关答案片段")
    store, _ = load_index()
    results = {m: {"hit": [], "recall": [], "mrr": [], "ndcg": []} for m in modes}
    per_question = []
    negative_counts = {m: 0 for m in modes}

    for q in tests:
        question = q["question"]
        category = q.get("category", "answerable")
        relevant_chunks = [c for c in store.chunks if _relevant(c, q)]
        if category == "answerable" and not relevant_chunks:
            raise ValueError(f"评估标注未命中任何语料分块：{question}")
        detail = {
            "question": question,
            "expected_key": q.get("expected_key", ""),
            "expected_sources": q.get("expected_sources", []),
            "relevant_chunk_ids": [c.chunk_id for c in relevant_chunks],
            "relevant_chunk_count": len(relevant_chunks),
            "modes": {},
        }
        if "category" in q:
            detail.update({name: q.get(name) for name in ("id", "category", "corpus_group", "answer_points", "expected_behavior")})
        for m in modes:
            docs = retrieve(question, mode=m, top_k=k)
            if category != "answerable":
                negative_counts[m] += bool(docs)
                detail["modes"][m] = {
                    "returned_candidates": len(docs),
                    "retrieved": [{**d.__dict__, "relevant": False} for d in docs],
                }
                continue
            r = results[m]
            r["hit"].append(hit_at_k(docs, q, k))
            r["recall"].append(recall_at_k(docs, q, k, len(relevant_chunks)))
            r["mrr"].append(mrr(docs, q))
            r["ndcg"].append(ndcg_at_k(docs, q, k, len(relevant_chunks)))
            detail["modes"][m] = {
                "hit": r["hit"][-1], "recall": r["recall"][-1],
                "mrr": r["mrr"][-1], "ndcg": r["ndcg"][-1],
                "retrieved": [{**d.__dict__, "relevant": _relevant(d, q)} for d in docs],
            }
        per_question.append(detail)
        print(f"[{len(per_question)}/{len(tests)}] {category}: {question}", flush=True)

    summary = {}
    for m, r in results.items():
        n = len(r["hit"])
        summary[m] = {
            "Hit@%d" % k: round(sum(r["hit"]) / n, 4),
            "Recall@%d" % k: round(sum(r["recall"]) / n, 4),
            "MRR": round(sum(r["mrr"]) / n, 4),
            "NDCG@%d" % k: round(sum(r["ndcg"]) / n, 4),
            "samples": n,
        }

    # 相对 baseline 提升（用于简历结果论）
    base = summary[modes[0]]
    summary["_improvement_vs_baseline"] = {
        m: {
            "Hit@%d" % k: _pct(base["Hit@%d" % k], summary[m]["Hit@%d" % k]),
            "MRR": _pct(base["MRR"], summary[m]["MRR"]),
            "NDCG@%d" % k: _pct(base["NDCG@%d" % k], summary[m]["NDCG@%d" % k]),
        }
        for m in modes[1:]
    }
    summary["_per_question"] = per_question
    summary["_metadata"] = _experiment_metadata(k, len(store.chunks), questions_path)
    if "version" in dataset:
        metadata = summary["_metadata"]
        metadata["dataset_version"] = dataset["version"]
        metadata["provenance"] = dataset.get("provenance", {})
        metadata["classification_counts"] = counts
        metadata["answerable_metric_denominator"] = counts["answerable"]
        metadata["corpus"] = dataset.get("corpus", {})
        metadata["historical_reference"] = dataset.get("historical_reference", {})
        metadata["generation_evaluated"] = False
        metadata["question_file"] = str(questions_path.relative_to(EVAL_DIR.parent) if questions_path.is_relative_to(EVAL_DIR.parent) else questions_path)
        summary["_unanswerable"] = {
            "samples": counts["unanswerable"] + counts["out_of_scope"],
            "interpretation": "仅记录无答案/越界题的检索候选返回；不计入可回答题的Hit/Recall/MRR/NDCG，不代表模型拒答结果。",
            "modes": {m: {"queries_returning_candidates": negative_counts[m], "rejection_accuracy": None} for m in modes},
        }
    summary["_metadata"]["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    return summary


def _pct(base: float, cur: float) -> str:
    if base <= 0:
        return "n/a"
    return f"{(cur - base) / base * 100:+.1f}%"


def main(argv: list[str] | None = None):
    import argparse
    parser = argparse.ArgumentParser(description="检索评估；无答案题单列，不计入有答案指标分母")
    parser.add_argument("k", nargs="?", type=int, default=5)
    parser.add_argument("--questions", type=Path, default=EVAL_DIR / "test_questions.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    legacy_questions = (EVAL_DIR / "test_questions.json").resolve()
    is_versioned = args.questions.resolve() != legacy_questions
    out = args.output or EVAL_DIR / ("eval_result_v2.json" if is_versioned else "eval_result.json")
    if is_versioned and out.resolve() == (EVAL_DIR / "eval_result.json").resolve():
        parser.error("新版题集不能覆盖28题历史结果，请使用独立输出文件")
    k = args.k
    n = len(_load_tests(args.questions))
    print(f"=== 检索评估实验（top-k={k}，共 {n} 个问题，chunk 级判定） ===\n")
    summary = run_experiment(k=k, questions_path=args.questions)
    header = ["模式", "Hit@%d" % k, "Recall@%d" % k, "MRR", "NDCG@%d" % k]
    print(f"{header[0]:<16}{header[1]:>10}{header[2]:>12}{header[3]:>10}{header[4]:>12}")
    print("-" * 62)
    for m in ("vector", "hybrid", "hybrid_rerank"):
        if m not in summary:
            continue
        s = summary[m]
        print(f"{m:<16}{s['Hit@%d' % k]:>10.4f}{s['Recall@%d' % k]:>12.4f}"
              f"{s['MRR']:>10.4f}{s['NDCG@%d' % k]:>12.4f}")

    print("\n相对 baseline（vector）提升：")
    for m, imp in summary.get("_improvement_vs_baseline", {}).items():
        print(f"  {m}: Hit@{k} {imp['Hit@%d' % k]}, MRR {imp['MRR']}, NDCG@{k} {imp['NDCG@%d' % k]}")

    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已保存: {out}")


if __name__ == "__main__":
    main()
