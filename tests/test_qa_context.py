from unittest.mock import patch

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from src.qa import stream_answer
from src.retrieval import RetrievedDoc


def test_stream_reuses_displayed_docs_and_citation_order():
    docs = [RetrievedDoc("验收合格后入库", "物资.md", "入库", 0.9, 1)]
    prompts = []

    def generate(prompt):
        prompts.append(prompt.to_messages())
        return AIMessage(content="验收合格后入库。[1]")

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
