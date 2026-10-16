"""Save real RAG generation evidence; automated checks are proxies, not accuracy.

Example: python -m scripts.evaluate_generation --run --limit 3 \
    --output eval_data/generation_2026-10-16.json
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config

MODES = ("vector", "hybrid", "hybrid_rerank")
REFUSAL_MARKERS = ("无法回答", "资料不足", "未提供", "没有相关", "未检索到")


def evaluate_generation(questions_path, output_path, *, mode="hybrid_rerank", limit=None, ids=None):
    if not config.LLM_API_KEY.strip():
        raise ValueError("未配置模型凭据；停止，不生成结果文件。")
    if mode not in MODES or (limit is not None and limit < 1):
        raise ValueError("模式或题目数量无效。")
    raw = Path(questions_path).read_bytes()
    dataset = json.loads(raw)
    questions = dataset["questions"]
    if ids:
        if set(ids) - {q["id"] for q in questions}:
            raise ValueError("题目 ID 不存在。")
        questions = [q for q in questions if q["id"] in ids]
    questions = questions[:limit]
    report = {
        "metadata": {"created_at": datetime.now(timezone.utc).isoformat(), "model_id": config.LLM_MODEL,
                     "mode": mode, "top_k": 5, "dataset_version": dataset.get("version"),
                     "questions_sha256": hashlib.sha256(raw).hexdigest(), "selected_count": len(questions)},
        "limitations": ["字面要点覆盖仅为 proxy，不是事实准确率。",
                        "引用编号仅检查格式和越界，不验证引用支持结论。",
                        "拒答短语仅记录 marker；事实、引用支持与拒答成功均待人工核查。"],
        "rows": [],
    }
    # Exclusive creation protects every existing record, including retrieval JSON.
    with Path(output_path).open("x", encoding="utf-8") as destination:
        from src.retrieval import retrieve
        from src.qa import stream_answer

        for question in questions:
            started = perf_counter()
            row = {**question, "sources": [], "answer": "", "status": "error",
                   "model_called": False, "error_type": None,
                   "manual_review": {"factual_accuracy": "pending", "citation_support": "pending",
                                     "refusal_success": "pending"}}
            interrupted = False
            try:
                docs = retrieve(question["question"], mode=mode, top_k=5)
                row["sources"] = [
                    {"citation": i, **{key: getattr(doc, key) for key in ("source", "heading", "text", "rank", "score")}}
                    for i, doc in enumerate(docs, 1)
                ]
                row["status"] = "no_sources"
                if docs:
                    row["status"], row["model_called"] = "incomplete", True
                    for event in stream_answer(question["question"], mode=mode, docs=docs):
                        if event["type"] == "token":
                            row["answer"] += event["data"]
                    if row["answer"].strip():
                        row["status"] = "completed"
            except (Exception, KeyboardInterrupt) as error:
                row["error_type"] = type(error).__name__  # Never retain exception bodies or credentials.
                interrupted = isinstance(error, KeyboardInterrupt)
            row["elapsed_seconds"] = round(perf_counter() - started, 4)
            points, answer = question.get("answer_points", []), row["answer"]
            matched = [point for point in points if point in answer]
            row["literal_coverage_proxy"] = {
                "matched": matched, "missing": [point for point in points if point not in answer],
                "rate": len(matched) / len(points) if points else None,
            }
            citations = sorted({int(value) for value in re.findall(r"\[(\d+)\]", answer)})
            row["citation_format"] = {"numbers_found": citations,
                                      "out_of_range": [i for i in citations if not 1 <= i <= len(row["sources"])]}
            row["refusal_markers_found"] = [marker for marker in REFUSAL_MARKERS if marker in answer]
            report["rows"].append(row)
            report["metadata"]["updated_at"] = datetime.now(timezone.utc).isoformat()
            report["metadata"]["interrupted"] = interrupted
            destination.seek(0)
            json.dump(report, destination, ensure_ascii=False, indent=2)
            destination.truncate()
            destination.flush()
            if interrupted:
                break
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="真实生成会消耗模型额度；结果必须人工核查。")
    parser.add_argument("--run", action="store_true", help="明确确认执行真实模型调用")
    parser.add_argument("--questions", type=Path, default=ROOT / "eval_data" / "test_questions_v2.json")
    parser.add_argument("--output", type=Path, required=True, help="独立的新结果 JSON，禁止覆盖")
    parser.add_argument("--mode", choices=MODES, default="hybrid_rerank")
    parser.add_argument("--limit", type=int, help="仅执行前 N 题，建议先使用 3")
    parser.add_argument("--ids", nargs="+", help="仅执行这些题目 ID，可组合正例和负例")
    args = parser.parse_args(argv)
    if not args.run:
        parser.error("必须明确传入 --run 才会调用模型；建议先 --limit 3。")
    if not config.LLM_API_KEY.strip():
        print("未配置模型凭据；未运行，未创建结果文件。", file=sys.stderr)
        return 2
    try:
        report = evaluate_generation(args.questions, args.output, mode=args.mode, limit=args.limit, ids=args.ids)
    except Exception as error:
        print(f"停止：{type(error).__name__}；未输出异常正文。", file=sys.stderr)
        return 2
    print(f"已保存 {len(report['rows'])} 条实际运行记录；完整性和人工复核状态请查看独立结果文件。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
