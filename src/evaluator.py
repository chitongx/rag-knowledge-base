"""离线评估：不依赖 LLM，只评估"检索"质量（面试里最有说服力的数字）。

判定口径（chunk 级）：一个 chunk 相关 = 其文本包含该问题的 expected_key（原文子串）。
相比"文档级命中"，chunk 级判定更严格，能区分"检索到了对的文档但找错了段落"。

指标口径（简历/面试直接可讲）：
- Hit@K：top-K 结果中是否出现相关 chunk（0/1 平均）
- Recall@K：相关 chunk 被召回的覆盖率（本数据集每题 1 个相关 chunk）
- MRR：第一个相关 chunk 的排名倒数平均
- NDCG@K：考虑排序位置的检索质量（相关 chunk 在 rank i 得 1/i）

实验设计：vector / hybrid / hybrid_rerank 三种模式在同一测试集、同一 top-k 下对比，
自动输出相对 baseline（vector）的提升百分比，可直接用于简历结果论表述。
"""
from __future__ import annotations

import json
from pathlib import Path

from .config import EVAL_DIR
from .retrieval import retrieve


def _load_tests() -> list[dict]:
    path = EVAL_DIR / "test_questions.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["questions"]


def _relevant(doc, q: dict) -> bool:
    """chunk 级相关判定：文本包含 expected_key；无 key 时退化为 source 匹配。"""
    key = q.get("expected_key", "")
    if key:
        return key in doc.text
    return doc.source in q.get("expected_sources", [])


# ---------------- 指标 ----------------
def hit_at_k(docs, q: dict, k: int) -> float:
    return 1.0 if any(_relevant(d, q) for d in docs[:k]) else 0.0


def recall_at_k(docs, q: dict, k: int) -> float:
    return 1.0 if any(_relevant(d, q) for d in docs[:k]) else 0.0


def mrr(docs, q: dict) -> float:
    for rank, d in enumerate(docs, 1):
        if _relevant(d, q):
            return 1.0 / rank
    return 0.0


def ndcg_at_k(docs, q: dict, k: int) -> float:
    """DCG = Σ 相关位置 1/(rank)；ideal 按期望相关 chunk 数计算（本数据集为 1）。"""
    dcg = sum(1.0 / (i + 1) for i, d in enumerate(docs[:k]) if _relevant(d, q))
    ideal = 1.0  # 每题期望 1 个相关 chunk
    return dcg / ideal if ideal > 0 else 0.0


# ---------------- 实验 ----------------
def run_experiment(k: int = 5, modes: tuple[str, ...] = ("vector", "hybrid", "hybrid_rerank")) -> dict:
    tests = _load_tests()
    results = {m: {"hit": [], "recall": [], "mrr": [], "ndcg": []} for m in modes}

    for q in tests:
        question = q["question"]
        for m in modes:
            docs = retrieve(question, mode=m, top_k=k)
            r = results[m]
            r["hit"].append(hit_at_k(docs, q, k))
            r["recall"].append(recall_at_k(docs, q, k))
            r["mrr"].append(mrr(docs, q))
            r["ndcg"].append(ndcg_at_k(docs, q, k))

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
    return summary


def _pct(base: float, cur: float) -> str:
    if base <= 0:
        return "n/a"
    return f"+{(cur - base) / base * 100:.1f}%"


def main():
    import sys
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    n = len(_load_tests())
    print(f"=== 检索评估实验（top-k={k}，共 {n} 个问题，chunk 级判定） ===\n")
    summary = run_experiment(k=k)
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

    out = EVAL_DIR / "eval_result.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已保存: {out}")


if __name__ == "__main__":
    main()
