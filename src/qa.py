"""问答生成：LangChain LCEL 组装（RAG 链）。

- 默认走 OpenAI 兼容 API（DeepSeek/通义/OpenAI/Ollama），配置见 .env；
- 检索结果按来源引用拼进 prompt，约束模型"无法回答时如实说明"（降幻觉）；
- 返回 answer + 引用的来源列表，供前端展示。
"""
from __future__ import annotations

from .config import LLM_API_BASE, LLM_API_KEY, LLM_MODEL, LLM_TEMPERATURE
from .retrieval import RetrievedDoc, retrieve

SYSTEM_PROMPT = """你是一名严谨的企业知识库问答助手。请只依据给定的资料片段回答问题，不要编造。

规则：
1. 回答前先看资料是否覆盖问题；资料中没有答案时，直接说"根据现有资料无法回答"，不要猜测。
2. 回答中如引用资料内容，用[1][2]标注编号，编号对应下方资料序号。
3. 保持简洁，使用中文。

资料片段：
{context}"""


def _build_llm():
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=LLM_MODEL,
        api_key=LLM_API_KEY or "EMPTY",
        base_url=LLM_API_BASE,
        temperature=LLM_TEMPERATURE,
        streaming=True,
    )


def _format_context(docs: list[RetrievedDoc]) -> str:
    parts = []
    for i, d in enumerate(docs, 1):
        parts.append(f"[{i}] 来源：{d.source}（{d.heading or '正文'}）\n{d.text}")
    return "\n\n".join(parts)


def answer_question(query: str, mode: str = "hybrid_rerank",
                    top_k: int = 5) -> dict:
    """返回 {question, answer, sources:[{source,heading,text}]}。"""
    docs = retrieve(query, mode=mode, top_k=top_k)
    if not docs:
        return {"question": query, "answer": "未检索到相关资料。", "sources": []}

    llm = _build_llm()
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", "问题：{question}")]
    )
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({
        "context": _format_context(docs),
        "question": query,
    })

    return {
        "question": query,
        "answer": answer,
        "sources": [
            {"source": d.source, "heading": d.heading, "text": d.text[:200]}
            for d in docs
        ],
    }


def stream_answer(query: str, mode: str = "hybrid_rerank", top_k: int = 5):
    """SSE 流式版本：先 yield 引用来源，再逐 token 流式输出答案。"""
    docs = retrieve(query, mode=mode, top_k=top_k)
    if not docs:
        yield {"type": "sources", "data": []}
        yield {"type": "token", "data": "未检索到相关资料。"}
        return

    yield {"type": "sources", "data": [
        {"source": d.source, "heading": d.heading, "rank": d.rank} for d in docs
    ]}

    llm = _build_llm()
    from langchain_core.prompts import ChatPromptTemplate
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", "问题：{question}")]
    )
    chain = prompt | llm
    for chunk in chain.stream({
        "context": _format_context(docs),
        "question": query,
    }):
        if chunk.content:
            yield {"type": "token", "data": chunk.content}
