"""Streamlit 在线演示版（供 Streamlit Community Cloud 免费公网部署）。

与 FastAPI 版共用同一套检索管线（src/retrieval），指标一致：
- 三种模式对比：vector / hybrid / hybrid_rerank
- 检索结果带来源引用
- 可选 LLM 生成回答（配置 LLM_API_BASE / LLM_API_KEY 后启用）

部署：把本仓库接入 Streamlit Cloud（登录 streamlit.io → New app → 选本仓库 → Main file 选 streamlit_app.py）→ 得到公网地址。
"""
from __future__ import annotations

import streamlit as st
import json
from pathlib import Path

from src.config import LLM_API_KEY
from src.retrieval import retrieve
from src.qa import stream_answer

st.set_page_config(page_title="HRP 知识库问答 · RAG 演示", page_icon="🏥", layout="centered")

st.title("🏥 HRP 医院运营知识库问答")
st.caption("HRP RAG 演示：LangChain + FAISS + BM25 混合检索 + BGE-Rerank 精排，检索结果带来源。首次提问会下载并加载模型。")

q = st.text_input("你的问题", placeholder="例如：什么是VMI？")
col1, col2 = st.columns([1, 1])
mode = col1.selectbox("检索模式", ["hybrid_rerank", "hybrid", "vector"],
                      help="hybrid_rerank（默认，推荐）：混合检索+精排；hybrid：混合检索；vector：纯向量")
use_llm = col2.checkbox("生成回答（需配置 LLM Key）", value=False)

if st.button("提问", type="primary", use_container_width=True) and q:
    try:
        with st.spinner("检索中（首次加载模型需要等待）..."):
            docs = retrieve(q, mode=mode, top_k=5)
    except Exception:
        st.error("该模式加载或检索失败，请尝试 vector / hybrid 模式，或由部署者查看构建日志。")
        st.stop()
    if not docs:
        st.warning("未检索到相关内容，请换个问法。")
    else:
        st.markdown("#### 引用来源")
        for i, d in enumerate(docs, 1):
            st.markdown(
                f"**{i}. {d.source}**\n\n"
                f"{d.heading}\n\n{d.text[:160]}…",
            )
            with st.expander(f"查看来源 [{i}] 完整片段"):
                st.markdown(d.text)
        if use_llm and LLM_API_KEY:
            st.markdown("#### 回答")
            def answer_tokens():
                for event in stream_answer(q, mode=mode, docs=docs):
                    if event["type"] == "token":
                        yield event["data"]
            try:
                st.write_stream(answer_tokens())
            except Exception:
                st.error("模型服务生成失败，请由部署者检查模型服务配置和日志。")
        elif use_llm:
            st.info("未配置 LLM_API_KEY，当前仅展示检索结果与来源。")

st.divider()
result_path = Path(__file__).resolve().parent / "eval_data" / "eval_result.json"
if result_path.exists():
    results = json.loads(result_path.read_text(encoding="utf-8"))
    st.caption("28 题 HRP 评估集的真实检索结果；关键词与来源共同判定相关分块。该样本结果不代表真实业务全量表现。")
    st.dataframe([{"mode": m, **results[m]} for m in ("vector", "hybrid", "hybrid_rerank")], hide_index=True)
else:
    st.caption("离线评估：运行 python -m scripts.evaluate 5 后生成三种模式的真实结果。")
