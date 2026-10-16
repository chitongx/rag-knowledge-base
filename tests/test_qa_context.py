from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from src.qa import answer_question, stream_answer
from src.retrieval import RetrievedDoc


def test_stream_reuses_displayed_docs_and_citation_order():
    docs = [RetrievedDoc("验收合格后入库", "物资.md", "入库", 0.9, 1)]
    prompts = []

    def generate(prompt):
        prompts.append(prompt.to_messages())
        return AIMessage(content="验收合格后入库。[1]", response_metadata={"finish_reason": "stop"})

    with patch("src.qa.retrieve") as retrieval, patch("src.qa._build_llm", return_value=RunnableLambda(generate)):
        events = list(stream_answer("入库流程", docs=docs))
    retrieval.assert_not_called()
    assert events[0]["data"][0]["source"] == "物资.md"
    assert "[1] 来源：物资.md（入库）\n验收合格后入库" in prompts[0][0].content
    assert events[1] == {"type": "token", "data": "验收合格后入库。[1]"}


def test_explicit_empty_docs_does_not_retrieve_or_call_model():
    with patch("src.qa.retrieve") as retrieval, patch("src.qa._build_llm") as model:
        events = list(stream_answer("入库流程", docs=[]))
    retrieval.assert_not_called()
    model.assert_not_called()
    assert events == [{"type": "sources", "data": []}, {"type": "token", "data": "未检索到相关资料。"}]


def test_stream_does_not_accept_token_limit_as_a_complete_answer():
    docs = [RetrievedDoc("验收合格后入库", "物资.md", "入库", 0.9, 1)]
    limited = RunnableLambda(lambda prompt: AIMessage(
        content="只输出了部分流程", response_metadata={"finish_reason": "length"}))
    with patch("src.qa._build_llm", return_value=limited):
        events = stream_answer("入库流程", docs=docs)
        assert next(events)["type"] == "sources"
        with pytest.raises(ValueError, match="上限|不完整"):
            list(events)


def test_nonstream_does_not_accept_token_limit_as_a_complete_answer():
    docs = [RetrievedDoc("验收合格后入库", "物资.md", "入库", 0.9, 1)]
    limited = RunnableLambda(lambda prompt: AIMessage(
        content="只输出了部分流程", response_metadata={"finish_reason": "length"}))
    with patch("src.qa.retrieve", return_value=docs), patch("src.qa._build_llm", return_value=limited):
        with pytest.raises(ValueError, match="上限|不完整"):
            answer_question("入库流程")


@pytest.mark.parametrize("reason", [None, "content_filter"])
def test_stream_requires_a_successful_terminal_status(reason):
    docs = [RetrievedDoc("验收合格后入库", "物资.md", "入库", 0.9, 1)]
    metadata = {} if reason is None else {"finish_reason": reason}
    interrupted = RunnableLambda(lambda prompt: AIMessage(content="半截回答", response_metadata=metadata))
    with patch("src.qa._build_llm", return_value=interrupted):
        with pytest.raises(ValueError, match="不完整|完成"):
            list(stream_answer("入库流程", docs=docs))


def test_nonstream_requires_a_successful_terminal_status():
    docs = [RetrievedDoc("验收合格后入库", "物资.md", "入库", 0.9, 1)]
    interrupted = RunnableLambda(lambda prompt: AIMessage(content="没有终态的回答"))
    with patch("src.qa.retrieve", return_value=docs), patch("src.qa._build_llm", return_value=interrupted):
        with pytest.raises(ValueError, match="不完整|完成"):
            answer_question("入库流程")
