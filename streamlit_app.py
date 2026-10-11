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
from time import perf_counter

from src.config import LLM_API_KEY
from src.retrieval import retrieve
from src.qa import stream_answer

st.set_page_config(page_title="HRP 知识库问答 · RAG 演示", page_icon="🏥", layout="centered")

st.title("🏥 HRP 医院运营知识库问答")
st.caption("8 篇医院运营示例文档 · 三种检索模式 · 原文来源可追溯")

MODE_LABELS = {"vector": "纯向量", "hybrid": "混合检索", "hybrid_rerank": "混合检索 + 精排"}
result_path = Path(__file__).resolve().parent / "eval_data" / "eval_result.json"
results = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else None
demo_type = st.radio("演示方式", ["实时检索", "历史评估回放"], horizontal=True)

if demo_type == "实时检索":
    if LLM_API_KEY:
        st.info("回答生成已配置；勾选后调用模型服务，实际可用性以本次请求为准。")
    else:
        st.info("回答生成未配置：当前演示实时检索与原文来源，暂不生成回答。")
    st.caption("首次提问需下载并加载模型；候选资料不代表问题一定有答案。遇到冷启动或网络故障，可切换历史评估回放。")

    def use_example():
        selected = st.session_state.example_question
        st.session_state.question = "" if selected == "自定义问题" else selected

    st.selectbox("演示问题", ["自定义问题", "药品和耗材入库要核对什么？",
                          "耗材是怎么直接送到科室的？", "科室用钱超过预算了怎么办？"],
                 key="example_question", on_change=use_example)
    q = st.text_input("你的问题", placeholder="例如：医院物资入库流程", key="question").strip()
    col1, col2 = st.columns([1, 1])
    mode = col1.selectbox("检索模式", ["hybrid_rerank", "hybrid", "vector"],
                         format_func=MODE_LABELS.get, key="retrieval_mode",
                         help="向量找语义，BM25 找关键词，RRF 融合排名；精排进一步调整候选顺序。")
    use_llm = col2.checkbox("生成回答", value=False, disabled=not bool(LLM_API_KEY))
    submitted = st.button("提问", type="primary", use_container_width=True)
    if submitted:
        st.session_state.rag_result = None
        if not q:
            st.warning("请输入问题，再点击提问。")
        else:
            try:
                started = perf_counter()
                with st.spinner("检索中（首次加载模型需要等待）..."):
                    docs = retrieve(q, mode=mode, top_k=5)
                st.session_state.rag_result = {
                    "question": q, "mode": mode, "docs": docs,
                    "elapsed": perf_counter() - started, "answer": None, "generation_error": False,
                }
            except Exception:
                st.error("加载或检索失败。可改用纯向量 / 混合检索重新提问，或切换历史评估回放继续演示。")
    result = st.session_state.get("rag_result")
    if result:
        st.caption(f"本次结果：{MODE_LABELS[result['mode']]} · 问题：{result['question']} · 检索耗时 {result['elapsed']:.2f} 秒（含本次加载时间）")
        docs = result["docs"]
        if not docs:
            st.warning("未检索到资料片段，请换个问法。")
        else:
            st.markdown("#### 引用来源")
            for i, d in enumerate(docs, 1):
                preview = d.text[:160] + ("…" if len(d.text) > 160 else "")
                st.markdown(f"**[{i}] {d.source}**\n\n{d.heading}\n\n{preview}")
                with st.expander(f"查看来源 [{i}] 完整片段"):
                    st.markdown(d.text)
            if submitted and use_llm and LLM_API_KEY:
                st.markdown("#### 回答")

                def answer_tokens():
                    for event in stream_answer(result["question"], mode=result["mode"], docs=docs):
                        if event["type"] == "token":
                            yield event["data"]

                try:
                    answer = st.write_stream(answer_tokens())
                    if not isinstance(answer, str) or not answer.strip():
                        raise ValueError("Model service returned an empty answer")
                    result["answer"] = answer
                except Exception:
                    result["generation_error"] = True
            elif result["answer"] is not None:
                st.markdown("#### 回答")
                st.markdown(result["answer"])
            if result["generation_error"]:
                st.error("生成未完成，已显示的文字可能不完整。原文来源仍可核对；可关闭生成后重新提问。")
else:
    st.info("历史评估回放：展示已保存的真实检索记录，不调用模型，不是本次实时检索或生成回答。")
    if results:
        records = results["_per_question"]
        question = st.selectbox("回放问题", [r["question"] for r in records], key="recorded_question",
                                index=4)
        record = next(r for r in records if r["question"] == question)
        st.caption(f"记录日期：{results['_metadata']['created_at'][:10]} · 对同一个问题比较三种模式的 Top 5。")
        st.caption("相关片段由来源文档与标注原文共同判定；排名越靠前，倒数排名越高。不同模式的原始分数不可直接比较。")
        for column, (mode, label) in zip(st.columns(3), MODE_LABELS.items()):
            with column:
                st.markdown(f"**{label}**")
                mode_result = record["modes"][mode]
                st.metric("本题倒数排名", f"{mode_result['mrr']:.2f}")
                for i, doc in enumerate(mode_result["retrieved"], 1):
                    state = "相关" if doc["relevant"] else "未匹配标注"
                    with st.expander(f"[{i}] {doc['heading'] or '正文'} · {state}"):
                        st.caption(doc["source"])
                        st.markdown(doc["text"])
    else:
        st.warning("暂无历史评估记录。")

st.divider()
if results:
    st.markdown("#### 28 题历史评估")
    st.caption("关键词与来源共同判定相关分块；样本只覆盖有答案问题，不代表全量业务表现、生成准确率或无答案拒答能力。")
    st.dataframe([{"检索模式": MODE_LABELS[m], **results[m]} for m in MODE_LABELS], hide_index=True)
    with st.expander("指标含义与复现信息"):
        st.markdown("Hit@5：前五个片段是否命中；MRR：首个相关片段的倒数排名均值；NDCG@5：前五个片段的排序质量。")
        st.json(results["_metadata"])
    st.download_button("下载完整评估记录", result_path.read_bytes(), file_name="rag-evaluation.json", mime="application/json")
else:
    st.caption("离线评估：运行 python -m scripts.evaluate 5 后生成三种模式的真实结果。")
