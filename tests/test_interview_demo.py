"""Interview demo behavior, without downloading models or calling an API."""
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from streamlit.testing.v1 import AppTest

from src import config, qa, retrieval


@contextmanager
def demo_app(*, api_key="", retrieve=None, stream_answer=None):
    retrieve = retrieve or Mock(return_value=[
        SimpleNamespace(source="02_医院物资管理流程.md", heading="入库验收",
                        text="验收不合格不得入库", score=0.91, rank=1)
    ])
    with patch.object(config, "LLM_API_KEY", api_key), \
            patch.object(retrieval, "retrieve", retrieve), \
            patch.object(qa, "stream_answer", stream_answer or Mock(return_value=iter([]))):
        yield AppTest.from_file(str(Path(__file__).resolve().parents[1] / "streamlit_app.py")), retrieve


def test_missing_generation_service_is_clear_before_question_and_cannot_be_enabled():
    with demo_app() as (app, retrieve):
        app.run()
        assert not app.exception
        assert app.checkbox[0].disabled
        assert any("回答生成未配置" in item.value for item in app.info)
        app.text_input[0].set_value("   ")
        app.button[0].click().run()
        retrieve.assert_not_called()
        assert any("请输入问题" in item.value for item in app.warning)


def test_representative_question_fills_input_without_starting_model_work():
    with demo_app() as (app, retrieve):
        app.run()
        example = app.selectbox(key="example_question")
        example.set_value("药品和耗材入库要核对什么？").run()
        assert not app.exception
        assert app.text_input[0].value == "药品和耗材入库要核对什么？"
        retrieve.assert_not_called()


def test_result_survives_controls_rerun_and_keeps_the_mode_that_produced_it():
    with demo_app() as (app, retrieve):
        app.run()
        app.text_input[0].set_value("医院物资入库流程")
        app.button[0].click().run()
        assert not app.exception
        app.selectbox(key="retrieval_mode").set_value("vector").run()
        assert not app.exception
        assert retrieve.call_count == 1
        assert any("验收不合格不得入库" in item.value for item in app.markdown)
        captions = "\n".join(item.value for item in app.caption)
        assert "本次结果：混合检索 + 精排" in captions
        assert "检索耗时" in captions


def test_historical_comparison_uses_recorded_full_sources_without_live_retrieval():
    with demo_app(retrieve=Mock(side_effect=RuntimeError("model unavailable"))) as (app, retrieve):
        app.run()
        app.text_input[0].set_value("医院物资入库流程")
        app.button[0].click().run()
        assert not app.exception
        assert any("历史评估回放" in item.value for item in app.error)
        app.radio[0].set_value("历史评估回放").run()
        app.selectbox(key="recorded_question").set_value("耗材是怎么直接送到科室的？").run()
        assert not app.exception
        assert retrieve.call_count == 1
        rendered = "\n".join(item.value for item in app.markdown)
        assert "SPD 模式" in rendered
        assert "2026-09-30" in "\n".join(item.value for item in app.caption)
        assert any("不调用模型" in item.value for item in app.info)
        assert [item.value for item in app.metric] == ["0.00", "0.25", "1.00"]
        assert len(app.get("download_button")) == 1


def test_expanded_record_separates_answerable_metrics_from_negative_candidates():
    import json
    record = json.loads((Path(__file__).resolve().parents[1] / "eval_data" / "eval_result_v2.json").read_text())
    negative = next(row for row in record["_per_question"] if row["category"] == "unanswerable")
    with demo_app() as (app, retrieve):
        app.run()
        app.radio[0].set_value("历史评估回放").run()
        assert any(item.key == "evaluation_version" for item in app.selectbox), "Expanded evidence must be selectable"
        app.selectbox(key="evaluation_version").set_value("扩展60题（12篇资料）").run()
        app.selectbox(key="recorded_question").set_value(negative["question"]).run()
        assert not app.exception
        retrieve.assert_not_called()
        captions = "\n".join(item.value for item in app.caption)
        assert "48 个可回答问题" in captions and "12 个无答案或越界问题" in captions
        assert not app.metric, "Negative candidates cannot receive answerable ranking metrics"
        assert any("未验收生成拒答" in item.value for item in app.warning)
        assert any("扩展评估" in item.value for item in app.markdown)


def test_generation_failure_retains_sources_and_does_not_report_completed_answer():
    def interrupted(*args, **kwargs):
        yield {"type": "token", "data": "未完成的回答"}
        raise TimeoutError("service unavailable")

    with demo_app(api_key="EMPTY", stream_answer=interrupted) as (app, retrieve):
        app.run()
        app.text_input[0].set_value("医院物资入库流程")
        app.checkbox[0].check()
        app.button[0].click().run()
        assert not app.exception
        assert any("生成未完成" in item.value for item in app.error)
        app.selectbox(key="retrieval_mode").set_value("hybrid").run()
        assert not app.exception
        assert retrieve.call_count == 1
        assert any("验收不合格不得入库" in item.value for item in app.markdown)
        assert any("生成未完成" in item.value for item in app.error)


@pytest.mark.parametrize("tokens", [[], [" ", "\n"]])
def test_empty_generation_marks_incomplete_and_retains_sources(tokens):
    def empty_answer(*args, **kwargs):
        for token in tokens:
            yield {"type": "token", "data": token}

    with demo_app(api_key="EMPTY", stream_answer=empty_answer) as (app, retrieve):
        app.run()
        app.text_input[0].set_value("医院物资入库流程")
        app.checkbox[0].check()
        app.button[0].click().run()
        assert not app.exception
        assert any("生成未完成" in item.value for item in app.error)
        assert app.session_state.rag_result["answer"] is None
        assert any("验收不合格不得入库" in item.value for item in app.markdown)
        app.selectbox(key="retrieval_mode").set_value("hybrid").run()
        assert not app.exception
        assert retrieve.call_count == 1
        assert any("生成未完成" in item.value for item in app.error)
