"""Small metric regressions; no model downloads or retrieval dependencies required."""
import importlib
import math
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


# The metrics operate on text/source values; importing them must not load models.
_config = types.ModuleType("src.config")
_config.EVAL_DIR = Path(__file__).resolve().parent.parent / "eval_data"
_retrieval = types.ModuleType("src.retrieval")
_retrieval.retrieve = lambda *args, **kwargs: None
with patch.dict(sys.modules, {"src.config": _config, "src.retrieval": _retrieval}):
    evaluator = importlib.import_module("src.evaluator")


class RetrievalMetricsTests(unittest.TestCase):
    def setUp(self):
        self.question = {"expected_key": "answer", "expected_sources": ["right.md"]}
        self.relevant = SimpleNamespace(text="answer", source="right.md")
        self.irrelevant = SimpleNamespace(text="other", source="right.md")

    def test_recall_uses_all_relevant_chunks(self):
        self.assertEqual(evaluator.recall_at_k([self.relevant], self.question, 1, 2), 0.5)

    def test_ndcg_uses_logarithmic_rank_discount(self):
        docs = [self.irrelevant, self.relevant]
        self.assertAlmostEqual(evaluator.ndcg_at_k(docs, self.question, 5, 1), 1 / math.log2(3))

    def test_ndcg_normalizes_multiple_relevant_chunks(self):
        docs = [self.relevant, self.relevant]
        self.assertAlmostEqual(evaluator.ndcg_at_k(docs, self.question, 5, 2), 1.0)

    def test_ndcg_ideal_uses_full_corpus_relevance(self):
        self.assertAlmostEqual(
            evaluator.ndcg_at_k([self.relevant], self.question, 5, 2),
            1 / (1 + 1 / math.log2(3)),
        )

    def test_no_relevant_corpus_chunks_have_zero_recall_and_ndcg(self):
        self.assertEqual(evaluator.recall_at_k([self.irrelevant], self.question, 5, 0), 0.0)
        self.assertEqual(evaluator.ndcg_at_k([self.irrelevant], self.question, 5, 0), 0.0)

    def test_relevance_requires_expected_source(self):
        wrong_source = SimpleNamespace(text="answer", source="other.md")
        self.assertEqual(evaluator.hit_at_k([wrong_source], self.question, 5), 0.0)


if __name__ == "__main__":
    unittest.main()
