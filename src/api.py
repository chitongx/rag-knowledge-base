"""FastAPI 演示服务：检索 + 问答（SSE 流式）+ 浏览器演示页。

接口：
- GET  /                 浏览器演示页（推荐直接打开）
- GET  /health
- POST /retrieve    {query, mode, top_k} -> 检索结果（带来源引用）
- POST /chat        {query, mode}       -> 完整回答（sources + answer）
- POST /chat/stream {query, mode}       -> SSE 流式（先来源后 token）
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import API_KEY
from .retrieval import retrieve
from .qa import answer_question, stream_answer

app = FastAPI(title="生产级 RAG 知识库", version="1.0.0")

STATIC_DIR = Path(__file__).resolve().parent / "static"

# 公网安全：配置 API_KEY 后，业务接口需携带 Bearer Token（演示页与健康检查除外）
PUBLIC_PATHS = {"/", "/health", "/docs", "/openapi.json", "/redoc"}


@app.middleware("http")
async def api_key_auth(request, call_next):
    if API_KEY and request.url.path not in PUBLIC_PATHS and not request.url.path.startswith("/static"):
        auth = request.headers.get("authorization", "")
        expected = f"Bearer {API_KEY}"
        if auth != expected and request.headers.get("x-api-key") != API_KEY:
            return JSONResponse(status_code=401, content={"detail": "无效或缺失 API Key"})
    return await call_next(request)


class RetrieveReq(BaseModel):
    query: str = Field(..., description="用户问题")
    mode: str = Field("hybrid_rerank", description="vector / hybrid / hybrid_rerank")
    top_k: int = Field(5, ge=1, le=20)


class ChatReq(BaseModel):
    query: str
    mode: str = "hybrid_rerank"


@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


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


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
