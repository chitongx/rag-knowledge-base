# 生产级 RAG 知识库问答系统

基于 **LangChain + FAISS + BM25 混合检索 + BGE-Rerank** 构建的企业知识库问答系统，完整覆盖
「文档解析 → 语义分块 → 向量化 → 混合检索 → 精排 → 生成 → 离线评估」全链路。
**检索质量可用真实指标量化**（见下文评估结果），适合作为 AI 应用开发工程师的简历项目。

## 核心亮点（面试/简历可直接讲）

1. **混合检索（向量 + BM25 + RRF）**：向量路解决语义相似，BM25 路精确命中专有名词/缩写
   （如 "SPD"、"VMI"、"三单匹配"），RRF（Reciprocal Rank Fusion）按排名融合两路结果，无需调参。
2. **Rerank 精排**：Cross-Encoder（bge-reranker）对候选逐对打分，把最相关的段落提到最前。
3. **离线评估闭环**：不依赖 LLM 的检索评估（Hit@K / Recall@K / MRR / NDCG），
   同一测试集对比三种检索模式，用数字证明优化有效。
4. **来源引用与降幻觉**：回答强制附引用来源，资料不足时明确拒答。

## 当前真实评估结果（28 个问题，top-k=5）

| 模式 | Hit@5 | Recall@5 | MRR | NDCG@5 |
|---|---|---|---|---|
| 纯向量（baseline） | 92.86% | 92.86% | 0.8423 | 0.8613 |
| 混合检索（+BM25+RRF） | **100%** | **100%** | 0.9077（+7.8%） | 0.9268（+7.6%） |
| 混合 + Rerank 精排 | **100%** | **100%** | **0.9464（+12.4%）** | **0.9732（+13.0%）** |

> 结果文件：`eval_data/eval_result.json`（可复现：`python -m scripts.evaluate 5`）

## 上线部署（两条路径，任选其一）

### 路径 A：Docker Compose 一键部署（本机 / 任何云服务器）

```bash
docker compose up -d          # 构建并启动（首次约 5-10 分钟）
# 浏览器打开 http://localhost:8000 即见演示页；/docs 为 API 文档
# 可选：cp .env.example .env 填入 LLM Key 开启完整问答
./deploy.sh                   # 或一键部署 + 健康自检
```

云服务器部署后把 `localhost` 换成服务器公网 IP 即可对外访问（记得在安全组放行 8000 端口）。

### 路径 B：HuggingFace Spaces 免费公网托管（简历可直接放链接）

1. 登录 [huggingface.co](https://huggingface.co) → **New Space**；
2. SDK 选择 **Docker**，填好名称（如 `rag-knowledge-base`）；
3. 把本目录文件上传（**Dockerfile 需重命名为 `Dockerfile.hf`**，或在 Space 的 Files 里把 `Dockerfile.hf` 的内容粘到根目录 `Dockerfile`）；
4. 点 **Create** 等待构建完成（CPU 免费实例即可，首次请求自动加载模型约 1-2 分钟）；
5. 完成后得到公网地址：`https://<你的用户名>-rag-knowledge-base.hf.space`，**可直接放进简历**。

> 提示：Space 免费 CPU 实例 16GB 内存足够；把模型缓存挂到 Space 的持久存储可避免每次重启重新下载。

## 公网安全（部署前必读）

**风险确认**：本项目公网部署的暴露面 = 只读检索接口 + 浏览器演示页 + 示例语料（均为公开示例数据，不含真实企业数据），无写操作、无用户可控外呼（LLM 仅指向部署者配置的固定 API），**示例语料下整体风险低**。但公网/简历上线前请执行：

1. **开启 API Key 鉴权**（推荐）：Space Settings → Variables 添加 `API_KEY=<随机密钥>`（`openssl rand -hex 16` 生成）。
   开启后 `/retrieve` `/chat` 接口需携带 `Authorization: Bearer <key>`（或 `X-API-Key`），演示页右上角输入框可自动附带；未带 Key 返回 401；
2. **不传真实企业数据**：`data/docs/` 为公开示例语料；替换真实文档前请自行脱敏；LLM Key 一律放 **Space Secrets**，不要写进代码/镜像；
3. **保持接口只读**：本项目无写入/删除接口；二次开发新增写接口时必须加鉴权与速率限制。

## 架构

```
用户提问
   │
   ▼
┌─────────────────────────────────────────────────────────┐
│ 检索管线                                                 │
│  ┌──────────┐   ┌──────────┐                            │
│  │ FAISS    │   │ BM25     │   向量路：BGE 向量相似度    │
│  │ 向量检索 │ + │ 关键词检索│   BM25路：jieba 分词命中    │
│  └────┬─────┘   └────┬─────┘                            │
│       └──────┬───────┘                                  │
│              ▼ RRF 融合                                  │
│       ┌─────────────┐                                   │
│       │ Cross-Encoder│  bge-reranker 精排               │
│       │ Rerank      │                                   │
│       └──────┬──────┘                                   │
│              ▼ Top-K 段落 + 来源引用                     │
│       ┌─────────────┐                                   │
│       │ LLM 生成回答 │   OpenAI 兼容 API / Ollama        │
│       │ (LCEL 链)   │                                   │
│       └──────┬──────┘                                   │
└──────────────┼──────────────────────────────────────────┘
               ▼
      回答 + 引用来源（SSE 流式）
```

## 快速开始

```bash
# 1. 安装依赖（Python 3.10+）
pip install -r requirements.txt

# 2. 放入文档：把你的 PDF/DOCX/TXT/MD 复制到 data/docs/（自带 HRP 医院运营示例语料）

# 3. 构建索引（首次自动下载 BGE 向量模型，CPU 可跑）
python -m scripts.build_index

# 4. 命令行检索 / 问答
python -m scripts.query "SPD是什么模式？"                # 检索（不调 LLM）
python -m scripts.query "SPD是什么模式？" --chat         # 问答（需配置 LLM Key）
python -m scripts.query "SPD是什么模式？" --mode vector  # 对比纯向量

# 5. 离线评估（产出简历指标）
python -m scripts.evaluate 5

# 6. 启动 Web 服务（http://localhost:8000/docs 有 Swagger 文档）
python -m scripts.serve
#   POST /retrieve    {query, mode, top_k}   检索
#   POST /chat        {query}               问答
#   POST /chat/stream {query}               SSE 流式问答
```

## 接入 LLM（可选，不接也能演示检索）

复制 `.env.example` 为 `.env` 填入任一 OpenAI 兼容 API：

```bash
LLM_API_BASE=https://api.deepseek.com/v1    # DeepSeek
LLM_API_KEY=sk-xxxx
LLM_MODEL=deepseek-chat
```

也可以指向本地 Ollama（`LLM_API_BASE=http://localhost:11434/v1`，`LLM_API_KEY=EMPTY`）。

## 目录结构

```
rag-knowledge-base/
├── data/
│   ├── docs/            # 原始文档（示例：HRP 医院运营八份语料）
│   └── index/           # 构建产物（FAISS 索引 + BM25 + 分块元数据）
├── eval_data/
│   └── test_questions.json   # 28 题测试集（含期望命中关键词）
├── scripts/
│   ├── build_index.py   # 构建索引
│   ├── query.py         # 命令行检索/问答
│   ├── evaluate.py      # 离线评估
│   └── serve.py         # 启动 Web 服务
└── src/
    ├── config.py        # 集中配置
    ├── ingestion.py     # 文档解析 + 标题感知分块
    ├── embeddings.py    # BGE 向量模型封装
    ├── vector_store.py  # 原生 FAISS 向量库
    ├── hybrid_search.py # BM25 + RRF 融合
    ├── reranker.py      # BGE-Rerank 精排
    ├── retrieval.py     # 检索管线总装
    ├── evaluator.py     # 评估指标与实验
    ├── qa.py            # LangChain LCEL 问答链
    └── api.py           # FastAPI 服务
```

## 替换成你自己的语料

1. 清空 `data/docs/`，放入你的文档（支持 PDF/DOCX/TXT/MD）；
2. 重写 `eval_data/test_questions.json`：每个问题标注 `expected_key`（期望命中的原文片段）
   和 `expected_sources`（期望来源文档）；
3. 重新 `python -m scripts.build_index` 和 `python -m scripts.evaluate`。

## 简历写法（结果论模板，跑完把数字填进去）

> **企业级 RAG 知识库问答系统**（Python / LangChain / FAISS / BM25 / BGE-Rerank / FastAPI）
>
> - 构建完整 RAG 链路：多格式文档解析、标题感知分块、向量化检索、来源引用与拒答机制；
> - 设计向量+BM25 混合检索与 RRF 融合，结合 Cross-Encoder 重排，Hit@5 由 92.9% 提升至 100%，
>   MRR 提升 12.4%、NDCG@5 提升 13.0%（28 题离线测试集复现）；
> - 实现 FastAPI + SSE 流式问答服务并 Docker 化，支持 OpenAI 兼容模型接口。

## 面试追问准备（对应实现点）

- **为什么用 RRF 而不是分数加权？** 两路检索器分数尺度不可比，RRF 只看排名，免调参且稳定；
- **为什么 Rerank 用 Cross-Encoder？** 双塔向量只看整体相似度，Cross-Encoder 让 query 与段落逐 token
  交互打分，精度更高，但慢，所以只对前 N 个候选精排；
- **chunk 怎么切？** 先按 Markdown 标题感知粗切（标题与其内容同 chunk），再按中文标点二次切，
  避免"标题和参数被切到两个块"；
- **怎么防幻觉？** 回答强制带引用编号、资料不足明确拒答、评估集覆盖"无答案"场景。
