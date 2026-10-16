"""Generation runner contracts; all model/retrieval calls are local fakes."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src import config, qa, retrieval

ROOT = Path(__file__).resolve().parents[1]


def test_real_generation_runner_entrypoint_exists():
    assert (ROOT / "scripts" / "evaluate_generation.py").exists(), "The real-generation evidence runner is missing"


@pytest.fixture
def runner(monkeypatch):
    path = ROOT / "scripts" / "evaluate_generation.py"
    if not path.exists():
        pytest.skip("Runner is not implemented yet")
    spec = importlib.util.spec_from_file_location("generation_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(config, "LLM_API_KEY", "unit-test-only-never-save")
    monkeypatch.setattr(config, "LLM_MODEL", "hrp-qwen35-4b")
    return module


@pytest.fixture
def questions(tmp_path):
    path = tmp_path / "questions.json"
    path.write_text(json.dumps({"version": "test-v2", "questions": [
        {"id": "a1", "category": "answerable", "question": "Question one",
         "answer_points": ["验收", "批号"], "expected_sources": ["物资.md"]},
        {"id": "a2", "category": "answerable", "question": "Question two", "answer_points": []},
        {"id": "n1", "category": "unanswerable", "question": "No source answer", "answer_points": []},
    ]}, ensure_ascii=False), encoding="utf-8")
    return path


@pytest.fixture
def documents(monkeypatch):
    docs = [SimpleNamespace(source="物资.md", heading="验收", text="完整资料" * 100 + "结尾证据",
                            rank=1, score=0.9)]
    retrieve = Mock(return_value=docs)
    monkeypatch.setattr(retrieval, "retrieve", retrieve)
    return docs, retrieve


def test_no_key_stops_before_creating_a_result(runner, questions, documents, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "LLM_API_KEY", "")
    output = tmp_path / "generation.json"
    with pytest.raises(ValueError):
        runner.evaluate_generation(questions, output)
    assert not output.exists()
    documents[1].assert_not_called()


def test_cli_requires_explicit_run_before_model_work(runner, documents, tmp_path):
    output = tmp_path / "generation.json"
    with pytest.raises(SystemExit):
        runner.main(["--output", str(output)])
    assert not output.exists()
    documents[1].assert_not_called()


def test_cli_missing_key_explains_no_run_without_sensitive_output(runner, questions, documents, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(config, "LLM_API_KEY", "")
    output = tmp_path / "generation.json"
    assert runner.main(["--run", "--questions", str(questions), "--output", str(output), "--limit", "1"]) == 2
    assert "未配置模型凭据" in capsys.readouterr().err
    assert not output.exists()
    documents[1].assert_not_called()


def test_existing_results_and_historical_retrieval_are_never_overwritten(runner, questions, documents, tmp_path):
    output = tmp_path / "old.json"
    output.write_text("old evidence", encoding="utf-8")
    historical = ROOT / "eval_data" / "eval_result.json"
    for target in [output, historical, questions]:
        before = target.read_bytes()
        with pytest.raises(FileExistsError):
            runner.evaluate_generation(questions, target)
        assert target.read_bytes() == before
    documents[1].assert_not_called()


def test_complete_run_preserves_same_full_sources_and_only_reports_proxies(runner, questions, documents, tmp_path, monkeypatch):
    docs, retrieve = documents
    stream = Mock(return_value=iter([{"type": "token", "data": "先验收"},
                                   {"type": "token", "data": "。[1][9]"}]))
    monkeypatch.setattr(qa, "stream_answer", stream)
    output = tmp_path / "generation.json"
    report = runner.evaluate_generation(questions, output, mode="vector", limit=1)
    assert report == json.loads(output.read_text(encoding="utf-8"))
    assert report["metadata"]["questions_sha256"] == hashlib.sha256(questions.read_bytes()).hexdigest()
    assert report["metadata"]["model_id"] == "hrp-qwen35-4b"
    row = report["rows"][0]
    assert (row["id"], row["category"], row["status"], row["answer"]) == ("a1", "answerable", "completed", "先验收。[1][9]")
    assert row["sources"][0]["text"].endswith("结尾证据")
    assert row["sources"][0]["citation"] == 1
    assert stream.call_args.kwargs["docs"] is docs
    assert stream.call_args.kwargs["mode"] == "vector"
    retrieve.assert_called_once_with("Question one", mode="vector", top_k=5)
    assert row["literal_coverage_proxy"] == {"matched": ["验收"], "missing": ["批号"], "rate": 0.5}
    assert row["citation_format"] == {"numbers_found": [1, 9], "out_of_range": [9]}
    assert all(value == "pending" for value in row["manual_review"].values())
    assert row["elapsed_seconds"] >= 0
    assert "unit-test-only-never-save" not in output.read_text(encoding="utf-8")


@pytest.mark.parametrize("exception", [TimeoutError("secret response body"), ValueError("cut off at token limit")])
def test_interrupted_or_truncated_stream_preserves_partial_without_error_body(runner, questions, documents, tmp_path, monkeypatch, exception):
    def stream(*args, **kwargs):
        yield {"type": "token", "data": "部分回答"}
        raise exception
    monkeypatch.setattr(qa, "stream_answer", stream)
    output = tmp_path / "generation.json"
    row = runner.evaluate_generation(questions, output, limit=1)["rows"][0]
    assert row["status"] == "incomplete"
    assert row["answer"] == "部分回答"
    assert row["error_type"] == type(exception).__name__
    assert row["sources"][0]["text"].endswith("结尾证据")
    assert str(exception) not in output.read_text(encoding="utf-8")


def test_empty_stream_is_incomplete(runner, questions, documents, tmp_path, monkeypatch):
    monkeypatch.setattr(qa, "stream_answer", Mock(return_value=iter([{"type": "token", "data": "  "}])))
    row = runner.evaluate_generation(questions, tmp_path / "generation.json", limit=1)["rows"][0]
    assert row["status"] == "incomplete"
    assert row["answer"] == "  "


def test_retrieval_error_does_not_call_model_or_save_exception_body(runner, questions, documents, tmp_path, monkeypatch, capsys):
    documents[1].side_effect = ConnectionError("provider credential and response body")
    stream = Mock()
    monkeypatch.setattr(qa, "stream_answer", stream)
    output = tmp_path / "generation.json"
    row = runner.evaluate_generation(questions, output, limit=1)["rows"][0]
    assert row["status"] == "error"
    assert row["error_type"] == "ConnectionError"
    stream.assert_not_called()
    assert "provider credential" not in output.read_text(encoding="utf-8") + capsys.readouterr().out


def test_refusal_marker_remains_pending_manual_judgment_and_ids_limit_scope(runner, questions, documents, tmp_path, monkeypatch):
    monkeypatch.setattr(qa, "stream_answer", Mock(return_value=iter([{"type": "token", "data": "根据现有资料无法回答。[1]"}])))
    report = runner.evaluate_generation(questions, tmp_path / "generation.json", ids=["n1"], limit=1)
    row = report["rows"][0]
    assert row["id"] == "n1"
    assert row["refusal_markers_found"] == ["无法回答"]
    assert row["manual_review"]["refusal_success"] == "pending"
    assert row["literal_coverage_proxy"]["rate"] is None


def test_no_sources_does_not_present_local_fallback_as_model_generation(runner, questions, documents, tmp_path, monkeypatch):
    documents[1].return_value = []
    stream = Mock()
    monkeypatch.setattr(qa, "stream_answer", stream)
    row = runner.evaluate_generation(questions, tmp_path / "generation.json", limit=1)["rows"][0]
    assert row["status"] == "no_sources"
    assert row["answer"] == ""
    assert row["model_called"] is False
    stream.assert_not_called()
