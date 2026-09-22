"""Streamlit 在线演示版（供 Streamlit Community Cloud 免费公网部署）。

与 FastAPI 版共用同一套检索管线（src/retrieval），指标一致：
- 三种模式对比：vector / hybrid / hybrid_rerank
- 检索结果带来源引用
- 可选 LLM 生成回答（配置 LLM_API_BASE / LLM_API_KEY 后启用）

部署：把本仓库接入 Streamlit Cloud（登录 streamlit.io → New app → 选本仓库 → Main file 选 streamlit_app.py）→ 得到公网地址。
"""
from __future__ import annotations

import streamlit as st

from src.config import LLM_API_KEY
from src.retrieval import retrieve
from src.qa import answer_question

st.set_page_config(page_title="HRP 知识库问答 · RAG 演示", page_icon="🏥", layout="centered")

st.title("🏥 HRP 医院运营知识库问答")
st.caption("生产级 RAG 演示：LangChain + FAISS + BM25 混合检索 + BGE-Rerank 精排，回答附来源引用。")


@st.cache_resource(show_spinner="首次加载模型（约 1-2 分钟）...")
def _warmup():
    """触发模型加载并缓存，避免重复下载/加载。"""
    retrieve("预热", mode="hybrid_rerank", top_k=1)
    return True


_warmup()

q = st.text_input("你的问题", placeholder="例如：什么是VMI？")
col1, col2 = st.columns([1, 1])
mode = col1.selectbox("检索模式", ["hybrid_rerank", "hybrid", "vector"],
                      help="hybrid_rerank（默认，推荐）：混合检索+精排；hybrid：混合检索；vector：纯向量")
use_llm = col2.checkbox("生成回答（需配置 LLM Key）", value=False)

if st.button("提问", type="primary", use_container_width=True) and q:
    with st.spinner("检索中..."):
        docs = retrieve(q, mode=mode, top_k=5)
    if not docs:
        st.warning("未检索到相关内容，请换个问法。")
    else:
        st.markdown("#### 引用来源")
        for i, d in enumerate(docs, 1):
            st.markdown(
                f"**{i}. {d.source}** · 相关度 {d.score:.4f}\n\n"
                f"<span style='color:#6b7280;font-size:12px'>{d.heading}</span>\n\n{d.text[:160]}…",
                unsafe_allow_html=True,
            )
        if use_llm and LLM_API_KEY:
            with st.spinner("生成回答中..."):
                ans = answer_question(q, mode=mode)
            st.markdown("#### 回答")
            st.write(ans.get("answer", ""))
        elif use_llm:
            st.info("未配置 LLM_API_KEY，仅展示检索结果（检索质量已由离线评估量化：Hit@5 92.9%→100%、MRR +12.4%）。")

st.divider()
st.caption("同一测试集（28 问）三种模式评估：纯向量 Hit@5 92.86% → 混合检索 100% → +Rerank 精排 MRR 0.9464（+12.4%）、NDCG@5 0.9732（+13.0%）")
