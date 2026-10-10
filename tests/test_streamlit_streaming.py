"""Verify the public UI consumes answer chunks, without calling a model service."""
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import streamlit as st
from streamlit.testing.v1 import AppTest


def test_question_uses_streamed_tokens_and_shows_sources():
    config = types.ModuleType("src.config")
    config.LLM_API_KEY = "EMPTY"
    retrieval = types.ModuleType("src.retrieval")
    retrieval.retrieve = lambda *args, **kwargs: [
        SimpleNamespace(source="02_医院物资管理流程.md", heading="入库", text="验收后入库" + "流程说明" * 50 + "验收不合格不得入库")
    ]
    qa = types.ModuleType("src.qa")
    qa.answer_question = lambda *args, **kwargs: {"answer": "整段回答"}
    emitted = []
    supplied_docs = []

    def stream_answer(*args, **kwargs):
        supplied_docs.append(kwargs.get("docs"))
        yield {"type": "sources", "data": []}
        for token in ["先验收，", "再入库。[1]"]:
            emitted.append(token)
            yield {"type": "token", "data": token}

    qa.stream_answer = stream_answer
    app_path = Path(__file__).resolve().parents[1] / "streamlit_app.py"
    with patch.dict(sys.modules, {"src.config": config, "src.retrieval": retrieval, "src.qa": qa}):
        with patch.object(st, "write_stream", wraps=st.write_stream) as streamed:
            app = AppTest.from_file(str(app_path)).run()
            app.text_input[0].set_value("医院物资入库流程")
            app.checkbox[0].check()
            app.button[0].click().run()
    assert not app.exception
    assert streamed.call_count == 1, "UI must consume the stream instead of waiting for a complete answer"
    assert emitted == ["先验收，", "再入库。[1]"]
    assert supplied_docs[0][0].text.endswith("验收不合格不得入库"), "Generation must use the exact displayed retrieval results"
    rendered = "\n".join(element.value for element in app.markdown)
    assert "先验收，再入库。[1]" in rendered
    assert "02_医院物资管理流程.md" in rendered
    assert "验收不合格不得入库" in rendered, "The complete retrieved evidence must be inspectable"
