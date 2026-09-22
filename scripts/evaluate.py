"""评估脚本入口。

用法：
  python -m scripts.evaluate            # top-k=5
  python -m scripts.evaluate 10         # top-k=10
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluator import main as eval_main

if __name__ == "__main__":
    eval_main()
