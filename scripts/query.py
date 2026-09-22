"""命令行问答/检索演示。

用法：
  python -m scripts.query "什么是HRP系统"                # 检索（不调 LLM）
  python -m scripts.query "什么是HRP系统" --chat         # 完整问答（需配置 LLM）
  python -m scripts.query "什么是HRP系统" --mode vector  # 指定检索模式
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", help="问题")
    ap.add_argument("--mode", default="hybrid_rerank",
                    choices=["vector", "hybrid", "hybrid_rerank"])
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--chat", action="store_true", help="调用 LLM 生成回答")
    args = ap.parse_args()

    if args.chat:
        from src.qa import answer_question
        result = answer_question(args.query, mode=args.mode, top_k=args.top_k)
        print(f"\nQ: {result['question']}\n")
        print(f"A: {result['answer']}\n")
        print("引用来源：")
        for s in result["sources"]:
            print(f"  - {s['source']} | {s['heading'] or '正文'}")
        return

    from src.retrieval import retrieve
    docs = retrieve(args.query, mode=args.mode, top_k=args.top_k)
    print(f"\nQ: {args.query}  [mode={args.mode}]\n")
    for d in docs:
        print(f"#{d.rank} [{d.source} | {d.heading or '正文'} | score={d.score}]")
        print(f"   {d.text[:120].replace(chr(10), ' ')}...\n")


if __name__ == "__main__":
    main()
