"""FastAPI 演示服务：检索 + 问答（SSE 流式）。

接口：
- GET  /health
- POST /retrieve    {query, mode, top_k} -> 检索结果（带来源引用）
- POST /chat        {query, mode}       -> 完整回答（sources + answer）
- POST /chat/stream {query, mode}       -> SSE 流式（先来源后 token）
"""
from __future__ import annotations

import json

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .retrieval import retrieve
from .qa import answer_question, stream_answer

app = FastAPI(title="生产级 RAG 知识库", version="1.0.0")


class RetrieveReq(BaseModel):
    query: str = Field(..., description="用户问题")
    mode: str = Field("hybrid_rerank", description="vector / hybrid / hybrid_rerank")
    top_k: int = Field(5, ge=1, le=20)


class ChatReq(BaseModel):
    query: str
    mode: str = "hybrid_rerank"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/retrieve")
def api_retrieve(req: RetrieveReq):
    docs = retrieve(req.query, mode=req.mode, top_k=req.top_k)
    return {
        "query": req.query,
        "mode": req.mode,
        "results": [d.__dict__ for d in docs],
    }


@app.post("/chat")
def api_chat(req: ChatReq):
    return answer_question(req.query, mode=req.mode)


@app.post("/chat/stream")
def api_chat_stream(req: ChatReq):
    async def gen():
        for event in stream_answer(req.query, mode=req.mode):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")
