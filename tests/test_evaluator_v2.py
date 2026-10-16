"""Versioned evaluation keeps answerability boundaries and historical artifacts."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src import evaluator
from src.ingestion import chunk_document


ROOT = Path(__file__).resolve().parents[1]
def test_no_answer_questions_are_not_in_retrieval_hit_denominator(tmp_path):
    path = tmp_path / "questions.json"
    path.write_text(json.dumps({"version": "test-v2", "questions": [
        {"id": "a1", "category": "answerable", "question": "有答案的问题",
         "expected_key": "answer", "expected_sources": ["right.md"], "answer_points": ["answer"]},
        {"id": "n1", "category": "unanswerable", "question": "未知数据",
         "expected_key": "", "expected_sources": [], "answer_points": []},
        {"id": "n2", "category": "out_of_scope", "question": "临床诊断",
         "expected_key": "", "expected_sources": [], "answer_points": []},
    ]}), encoding="utf-8")
    doc = SimpleNamespace(text="answer", source="right.md", heading="", score=0.8, rank=1)
    chunk = SimpleNamespace(text="answer", source="right.md", chunk_id="c1")
    with patch("src.retrieval.load_index", return_value=(SimpleNamespace(chunks=[chunk]), None)), \
            patch.object(evaluator, "retrieve", return_value=[doc]):
        result = evaluator.run_experiment(modes=("vector",), questions_path=path)
    assert result["vector"]["samples"] == 1
    assert result["vector"]["Hit@5"] == 1.0
    assert result["_unanswerable"]["samples"] == 2
    assert result["_unanswerable"]["modes"]["vector"]["queries_returning_candidates"] == 2
    assert result["_unanswerable"]["modes"]["vector"]["rejection_accuracy"] is None
    assert result["_metadata"]["classification_counts"] == {"answerable": 1, "unanswerable": 1, "out_of_scope": 1}
    assert result["_metadata"]["sha256"]["scripts/build_index.py"] == hashlib.sha256((ROOT / "scripts/build_index.py").read_bytes()).hexdigest()
    assert "hit" not in result["_per_question"][1]["modes"]["vector"]


def test_versioned_cli_cannot_overwrite_legacy_result(tmp_path):
    path = tmp_path / "new_questions.json"
    path.write_text('{"questions": []}', encoding="utf-8")
    protected = ROOT / "eval_data/eval_result.json"
    before = protected.read_bytes()
    with patch.object(evaluator, "run_experiment") as run:
        with pytest.raises(SystemExit):
            evaluator.main(["5", "--questions", str(path), "--output", str(protected)])
    run.assert_not_called()
    assert protected.read_bytes() == before


def test_regression_percentage_has_a_single_negative_sign():
    assert evaluator._pct(0.8, 0.7) == "-12.5%"


def test_v2_data_has_supported_answers_distinct_questions_and_honest_provenance():
    data = json.loads((ROOT / "eval_data/test_questions_v2.json").read_text(encoding="utf-8"))
    old = json.loads((ROOT / "eval_data/test_questions.json").read_text(encoding="utf-8"))
    rows = data["questions"]
    assert data["version"] == "hrp-retrieval-v2"
    assert data["provenance"]["author"]
    assert data["provenance"]["independent_third_party"] is False
    assert len(data["corpus"]["original"]) == 8
    assert len(data["corpus"]["extension"]) >= 4
    assert len({q["id"] for q in rows}) == len(rows)
    assert len({q["question"] for q in rows}) == len(rows)
    assert not {q["question"] for q in rows} & {q["question"] for q in old["questions"]}
    answerable = [q for q in rows if q["category"] == "answerable"]
    assert 40 <= len(answerable) <= 60
    assert len(rows) - len(answerable) >= 10
    for name in data["corpus"]["extension"]:
        text = (ROOT / "data/docs" / name).read_text(encoding="utf-8")
        assert "作者编写" in text and "示例规则" in text and "非真实医院制度" in text
    for q in rows:
        assert q["answer_points"] if q["category"] == "answerable" else q["expected_behavior"]
        if q["category"] == "answerable":
            assert q["corpus_group"] in ("original", "extension")
            assert q["expected_key"] and q["expected_sources"]
            chunks = [c for source in q["expected_sources"] for c in chunk_document(ROOT / "data/docs" / source)]
            assert any(q["expected_key"] in c.text for c in chunks), q["id"]
        else:
            assert q["expected_sources"] == [] and q["expected_key"] == ""


def test_legacy_data_and_original_documents_remain_frozen():
    manifest = json.loads((ROOT / "eval_data/test_questions_v2.json").read_text(encoding="utf-8"))["historical_reference"]["sha256"]
    assert len(manifest) == 10
    for relative, expected in manifest.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected, relative
