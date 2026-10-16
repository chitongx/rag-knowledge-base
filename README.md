# HRP RAG 知识库问答系统

基于 **LangChain + FAISS + BM25 混合检索 + BGE-Rerank** 构建的企业知识库问答系统，完整覆盖
「文档解析 → 语义分块 → 向量化 → 混合检索 → 精排 → 生成 → 离线评估」全链路。
**检索质量可用真实指标量化**（见下文评估结果），适合作为 AI 应用开发工程师的简历项目。

## 核心亮点（面试/简历可直接讲）

1. **混合检索（向量 + BM25 + RRF）**：向量路解决语义相似，BM25 路精确命中专有名词/缩写
   （如 "SPD"、"VMI"、"三单匹配"），RRF（Reciprocal Rank Fusion）按排名融合两路结果，无需调参。
2. **Rerank 精排**：Cross-Encoder（bge-reranker）对候选逐对打分，把最相关的段落提到最前。
3. **离线评估闭环**：不依赖 LLM 的检索评估（Hit@K / Recall@K / MRR / NDCG），
   同一测试集对比三种检索模式，用数字证明优化有效。
4. **来源引用**：检索结果保留文档来源，可展开完整片段；流式生成复用页面展示的同一批资料，避免重复检索及引用错位。生成提示要求引用编号并在资料不足时如实说明。该提示约束不等于已验证的拒答保证。

## 原始历史评估（8 篇资料、28 个问题，top-k=5）

| 模式 | Hit@5 | Recall@5 | MRR | NDCG@5 |
|---|---|---|---|---|
| 纯向量（baseline） | 92.86% | 92.86% | 0.8423 | 0.8558 |
| 混合检索（+BM25+RRF） | **100%** | **100%** | 0.9077（+7.8%） | 0.9226（+7.8%） |
| 混合 + Rerank 精排 | **100%** | **100%** | **0.9464（+12.4%）** | **0.9561（+11.7%）** |

> 结果文件：`eval_data/eval_result.json`。复现历史数字须恢复当时的 8 篇资料与环境；当前实时检索已经扩展，不能直接重跑后覆盖历史结果。

本次记录生成于 2026-09-30，包含逐题结果、模型与环境版本、配置和输入文件 SHA256。相关分块须同时满足期望来源和关键词；NDCG 使用 `1/log2(rank+1)` 折扣，并由全语料相关分块计算理想排序。百分比提升相对纯向量模式；28 题均为有答案的 HRP 样本，结果不能代表真实业务全量表现或无答案拒答能力。

## 扩展评估 V2（12 篇资料、60 个问题）

新增主数据治理、采购退货、月结异常和权限审计 4 篇作者编写的演示规则，均注明非真实医院制度。扩展索引共 52 个片段，题集为 48 个可回答问题、8 个无答案问题、4 个越界问题，未为新题调整检索排序。

2026-10-16 真实离线运行中，48 个可回答问题的纯向量 / 混合 / 精排 Hit@5 为 97.92% / 100% / 100%，MRR 为 0.9306 / 0.9729 / 1.0000。满分限于本次小型同作者示例集，不是业务回答准确率或第三方盲测。另 12 题全部返回检索候选；尚未调用生成模型，拒答准确率没有结果，不能据此推断拒答成功或失败。

完整记录为 [`eval_data/eval_result_v2.json`](eval_data/eval_result_v2.json)，题目、分母和复现方法见[扩展评估说明](docs/evaluation-v2.md)。公网页面的“历史评估回放”可以选择两个版本，原 28 题保留不变。不同题集和语料版本之间不能直接计算算法提升幅度。

## 高 star 项目参考与本次取舍（2026-10-10）

参考 [RAGFlow](https://github.com/infiniflow/ragflow)（91,943 stars）的可追溯引用与混合检索，以及 [Haystack](https://github.com/deepset-ai/haystack)（26,715 stars）的组件数据流和流式输出。star 数为核对当日快照，不代表质量评估。

现有 FAISS + BM25 + RRF + BGE-Rerank 保留。本次落实完整来源展开、展示与生成共享检索结果、真正消费流式 token，并纠正 NDCG 理想排序与相关性判定，保存逐题证据。不引入更重的数据库或整套框架：当时的 8 篇示例文档和免费部署尚不需要这些成本。V2 已加入无答案问题；生成质量、引用支撑关系与真实服务性能仍需独立验收，不能由检索指标推断。

## 上线部署（两条路径，任选其一）

### 路径 A：Docker Compose 一键部署（本机 / 任何云服务器）

```bash
docker compose up -d          # 构建并启动（首次约 5-10 分钟）
# 浏览器打开 http://localhost:8000 即见演示页；/docs 为 API 文档
# 可选：cp .env.example .env 填入 LLM Key 开启完整问答
./deploy.sh                   # 或一键部署 + 健康自检
```

云服务器部署后把 `localhost` 换成服务器公网 IP 即可对外访问（记得在安全组放行 8000 端口）。

### 路径 B：Streamlit Community Cloud 免费公网托管（简历可直接放链接）

> 当前 RAG 演示使用 Streamlit Community Cloud。生成回答需要另行配置模型服务；免费网页托管不等于免费模型推理。Hugging Face 的 ZeroGPU 是另一条有资格与额度限制的免费 GPU 演示路线，见[官方说明](https://huggingface.co/docs/hub/spaces-zerogpu)。

1. 本仓库已含 `streamlit_app.py`（与 FastAPI 共用同一检索管线）与 `requirements.txt`；
2. 打开 [streamlit.io/cloud](https://streamlit.io/cloud) → **用 GitHub 登录** → **Create app**；
3. 选择本仓库（`chitongx/rag-knowledge-base`）→ Branch `main` → Main file `streamlit_app.py` → **Deploy**；
4. 构建完成后，验证页面提问和来源展示再加入简历。现有演示：[HRP 知识库问答](https://rag-knowledge-base01.streamlit.app)。

> 提示：免费实例会休眠，需在页面点击唤醒；启动与首次模型加载耗时由实例资源和下载速度决定。

## 公网安全（部署前必读）

**风险确认**：本项目公网部署的暴露面 = 只读检索接口 + 演示页 + 示例语料（均为公开示例数据，不含真实企业数据），无写操作、无用户可控外呼（LLM 仅指向部署者配置的固定 API），**示例语料下整体风险低**。但公网/简历上线前请执行：

1. **不传真实企业数据**：`data/docs/` 为公开示例语料；替换真实文档前请自行脱敏；
2. **LLM Key 一律放平台 Secrets**（Streamlit：Settings → Secrets；HF Spaces：Settings → Variables），不要写进代码/镜像；
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

# 5. 扩展版离线评估（旧28题历史结果保留）
python -m scripts.evaluate 5 --questions eval_data/test_questions_v2.json --output eval_data/eval_result_v2.json

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

当前公网生成尚未接通。免费云端账号、匹配权重恢复、两个应用的Secrets配置与实际验收顺序见[云端接入清单](https://github.com/chitongx/qwen-lora-finetune/blob/main/docs/cloud-acceptance.md)。配置项存在不等于模型已经可用。

服务实际可用后，可先显式运行少量生成验收，再扩展到完整V2；输出文件名必须是新文件，不能覆盖历史记录：

```bash
python -m scripts.evaluate_generation --run --limit 3 --output eval_data/generation_20261016_run1.json
```

该工具保存实际问题、来源、完整或中断的回答；字面要点覆盖、引用编号格式和拒答短语都只是检查信号，事实及引用支撑关系仍标记待人工复核。缺少模型凭据不会调用服务或创建虚构结果。现阶段尚未运行真实生成评估。

## 目录结构

```
rag-knowledge-base/
├── data/
│   ├── docs/            # 8 篇原始示例资料 + 4 篇演示规则
│   └── index/           # 构建产物（FAISS 索引 + BM25 + 分块元数据）
├── eval_data/
│   ├── test_questions.json   # 原28题历史集
│   └── test_questions_v2.json # 扩展60题，含无答案与越界分类
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
> - 构建 RAG 链路：多格式文档解析、标题感知分块、向量化检索与来源引用；
> - 设计向量+BM25 混合检索与 RRF 融合，结合 Cross-Encoder 重排，Hit@5 由 92.9% 提升至 100%，
>   MRR 提升 12.4%、NDCG@5 提升 11.7%（28 题 HRP 离线评估集）；
> - 实现 FastAPI + SSE 流式问答服务并 Docker 化，支持 OpenAI 兼容模型接口。

## 面试追问准备（对应实现点）

[五分钟面试演示指南](docs/interview-demo.md)：包含推荐问题、三种模式的真实排序对比、指标证据与故障备用流程。公网页面支持实时检索和明确标注的历史评估回放；未配置生成服务时禁用生成选项。

- **为什么用 RRF 而不是分数加权？** 两路检索器分数尺度不可比，RRF 只看排名，免调参且稳定；
- **为什么 Rerank 用 Cross-Encoder？** 双塔向量只看整体相似度，Cross-Encoder 让 query 与段落逐 token
  交互打分，精度更高，但慢，所以只对前 N 个候选精排；
- **chunk 怎么切？** 先按 Markdown 标题感知粗切（标题与其内容同 chunk），再按中文标点二次切，
  避免"标题和参数被切到两个块"；
- **怎么减少幻觉？** 提示要求依据资料并带引用编号；原28题与扩展60题的记录均只评估检索，尚未验证生成事实准确率或无答案拒答能力。达到输出上限或流中断时，页面标记生成未完成。
