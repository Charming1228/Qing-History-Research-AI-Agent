# 清鉴 · 清史研究智能体

> Qing Jian — 一个面向清史研究的 AI 智能体，融合**词典（辞典）**、**编年（实录）**、**档案（奏折/题本）**三类史料，通过「多路混合检索 + LLM 多 Agent 编排」回答历史考据问题，并生成带证据回溯的学术报告。

---

## 特性

- **三库融合检索**：档案（BM25 + BGE 向量 + RRF 融合 + 实体关系重排）、词典（BM25 + BGE-M3 + Milvus）、编年（text2vec + FAISS）
- **LLM 多 Agent 编排**：主控 Agent 通过 XML action 标签自主规划多轮跨库检索
- **史料精炼与交叉互证**：RiD 精炼、实录 vs 档案矛盾分析
- **可追溯的考据报告**：正文逐句标注证据编号，附录自动挂载完整来源
- **开箱即用的 Web 界面**：单文件前端，无需构建步骤

## 系统架构

```
用户查询
   │
   ▼
前端  清鉴-清史研究智能体.html（单文件 Web 应用）
   │  POST /api/chat
   ▼
FastAPI 后端  server.py（端口 8080）
   │
   ▼
MasterReasoner（agent4.py，当前主 Agent）
   ├─ KnowledgeBase —— 三库检索
   │   ├─ ArchiveRetriever           档案（archive_retrieval2/）
   │   ├─ DictFusionSearcher         词典（dict_retrieval/）
   │   └─ SmartHistoricalRetriever   编年（annal_retrieval/）
   ├─ RiDAgent                       史料精炼
   ├─ ConflictAnalyzerAgent          实录 vs 档案交叉互证
   └─ FinalReportAgent               生成「正文 + 证据附录」报告
```

## 目录结构

```
项目/
├── Qinghistory/               # Web 应用 + Agent 编排（运行主体）
│   ├── agent4.py              # 当前主 Agent
│   ├── agent2.py / agent.py   # 早期版本 Agent
│   ├── server.py              # FastAPI 后端
│   ├── config.py              # 全局配置（环境变量）
│   ├── 清鉴-清史研究智能体.html # 单文件前端
│   ├── archive_retrieval2/    # 档案检索（refined 现代文摘要）
│   ├── archive_retrieval/     # 旧版档案检索（filtered 文言文）
│   ├── dict_retrieval/        # 词典检索
│   └── annal_retrieval/       # 编年检索
│
├── 数据结构化处理/             # 编年 Markdown → 结构化 CSV
├── 档案字段过滤.py             # 档案 JSON 字段精简
├── 档案微调数据构建.py          # 调 LLM 生成档案精炼训练数据
├── 编年微调数据构建.py          # 调 LLM 生成编年精炼训练数据
├── archive_retriever.py        # 档案检索 v3（独立版）
├── archive_search.py           # 档案检索 v2（BM25 + LLM 重排）
├── archive_search_vernacular.py # 文言/白话 × BM25/语义对比实验
├── compare_retrievers.py       # 新旧检索器对比
│
├── 数据基础/                   # 原始与中间数据（本地保留，见 .gitignore）
├── 原始文件/                   # 编年原始 MD/PDF
├── 工作汇报/                   # 课程各阶段工作汇报
├── README.md
└── 项目文件说明.md             # 每个文件的逐项说明
```

## 环境依赖

- Python 3.12（开发时使用 Anaconda base 环境）
- 依赖安装：

```bash
pip install jieba rank-bm25 opencc pandas numpy \
    sentence-transformers faiss-cpu \
    fastapi uvicorn pydantic openai \
    FlagEmbedding pymilvus torch tqdm
```

> `FlagEmbedding` 与 `pymilvus` 仅词典检索需要；`faiss-cpu` 用于档案/编年向量检索。

## 配置

所有密钥通过环境变量读取，参考根目录 `.env.example`：

```bash
cp .env.example .env        # 然后填入真实密钥
# Linux/macOS
set -a && source .env && set +a
```

| 环境变量 | 说明 | 默认值 |
|---|---|---|
| `DEEPSEEK_API_KEY` | DeepSeek API 密钥（必填） | — |
| `DEEPSEEK_BASE_URL` | DeepSeek API 地址 | `https://api.deepseek.com/v1` |
| `DEEPSEEK_MODEL` | DeepSeek 模型名 | `deepseek-chat` |
| `MOARK_API_KEY` | Moark 中转 API 密钥 | — |
| `MOARK_BASE_URL` | Moark 中转地址 | `https://api.moark.com/v1` |
| `MOARK_MODEL` | Moark 模型名 | `DeepSeek-V4-Pro` |

> `.env` 已被 `.gitignore` 排除，不会提交到仓库。

## 快速开始

```bash
cd Qinghistory
python server.py
# 浏览器打开 http://localhost:8080
```

启动时 `server.py` 会自动初始化 `MasterReasoner` 并加载三套检索索引，约需 1–2 分钟。

命令行方式（跳过 Web）：

```bash
cd Qinghistory
python agent4.py
```

## 各模块用法

### 档案检索

```bash
cd Qinghistory/archive_retrieval2
python archive_retriever.py               # refined（现代文摘要，默认）
python archive_retriever.py filtered      # filtered（文言文原文）
```

```python
from archive_retriever import ArchiveRetriever
r = ArchiveRetriever(data_format="refined")
r.load_or_build()          # 首次构建（较慢），后续秒级加载缓存
results = r.search("乾隆年间福建水师海防", top_k=10)
```

### 词典检索

```bash
cd Qinghistory/dict_retrieval/dict_search
python example_call_search.py "军机处"
```

```python
from search_dict_fusion_core import DictFusionSearcher
s = DictFusionSearcher()
out = s.search("军机处")
```

### 编年检索

```bash
cd Qinghistory/annal_retrieval
python a.py
```

```python
from a import SmartHistoricalRetriever
r = SmartHistoricalRetriever(file_paths=[
    "data/光绪朝结构化数据摘要.json",
    "data/乾隆朝结构化数据摘要_aligned.json",
    "data/康熙朝结构化数据摘要_aligned.json",
])
results = r.search("光绪年间", max_k=8, min_score=0.60, drop_threshold=0.1)
```

## 数据复现

仓库只提交**代码与文档**，数据/模型/索引在本地按以下流程重建或获取（`数据基础/`、`原始文件/` 已加入 `.gitignore`）。

| 产物 | 生成方式 | 依赖输入 |
|---|---|---|
| 编年结构化 CSV | `数据结构化处理/data.py`、`data_new.py`、`process_third_group.py` | 编年 Markdown |
| 档案字段精简 JSON | `档案字段过滤.py` | `数据基础/档案/*.json` |
| 档案精炼数据（refined） | `档案微调数据构建.py` | `档案_过滤/*.json` + LLM API |
| 档案检索索引缓存 | `archive_retriever.py` 首次运行 `load_or_build()` | 档案 JSON + BGE 模型 |
| 编年检索索引缓存 | `annal_retrieval/a.py` 首次运行 | 编年摘要 JSON + text2vec 模型 |
| 词典 Milvus 库 | `dict_retrieval/dict_search/rebuild_milvus.py` | `dict_chunks.jsonl` + BGE-M3 模型 |

### 模型获取

- **BGE-large-zh-v1.5**：`archive_retriever.py` 会自动从 HuggingFace 下载，或手动放入 `models/bge-large-zh-v1.5/`
- **BGE-M3**：放入 `Qinghistory/dict_retrieval/dict_search/models/bge-m3/`
- **text2vec-base-chinese**：`annal_retrieval/a.py` 会自动检测本地快照或从 HuggingFace 下载

### 已知说明

1. 词典分块数据 `dict_chunks.jsonl` 的生成脚本未包含在本仓库中（离线预处理产物）。
2. 编年摘要 JSON（`annal_retrieval/data/*.json`）的加工脚本未包含在本仓库中。
3. `编年微调数据构建.py` 中的输入 CSV 路径需按本机实际路径调整。

## 相关文档

- `项目文件说明.md` —— 仓库每个文件的逐项说明
- `CLAUDE.md` / `Qinghistory/CLAUDE.md` —— 开发上下文说明
- `工作汇报/` —— 课程各阶段工作汇报
