# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**语言偏好：始终使用中文回复。**

## Project overview

清鉴 (Qing Jian) — a Qing Dynasty history research AI agent. Integrates three types of historical sources: dictionaries (辞典), chronicles (编年), and archives (档案). The system consists of data processing pipelines, a BM25 search engine over ~250K archival records, LLM-based structured extraction for fine-tuning data, and a web frontend.

## Python environment

**必须使用 Anaconda base 环境**（`D:\anaconda\python.exe`，Python 3.12）。Base 环境已预装几乎所有依赖（jieba, rank-bm25, openai, opencc, pandas, tqdm, sentence-transformers, faiss-cpu 等），无需额外安装。

```bash
# 验证环境
python --version  # Python 3.12.x (Anaconda base)
```

如需安装缺失的包：
```bash
pip install <package-name>
```

Run scripts from the project root (`D:\26Spring\数据科学实践\项目`), not from subdirectories.
**始终使用 `python` 命令（不是 `python3`）。**

## Data pipeline

### Chronicle (编年) processing

Markdown files of Qing Dynasty chronicles are parsed into structured CSV with 11 columns (帝号, 年号纪年, 干支纪年, 农历月, 是否闰月, 农历日, 干支纪日, 原始公历日期, 标准公历日期, 事件正文, 注释).

Three related scripts share the same parsing logic, each slightly evolved:

| Script | Input | Output | Notes |
|--------|-------|--------|-------|
| `数据结构化处理/data.py` | 光绪朝.下.md → 光绪朝结构化数据下.csv | Original, hardcoded to 光绪 |
| `数据结构化处理/data_new.py` | 光绪朝上下.md → 光绪/宣统 split CSVs | Adds auto-detection of 宣统 boundary (1909-01-22) |
| `数据结构化处理/process_third_group.py` | 咸丰/同治.md → CSV | Generalized: dynamic emperor name regex, empty-record filtering, GB18030 encoding |

Key parsing behaviors common to all three: Lunar calendar date extraction with fake-date detection (lines ending in hanging punctuation like `，、；` won't trigger a new record), Gregorian year cross-year detection (when Gregorian month flips to Jan-Mar but lunar year hasn't changed), footnote/annotation association via citation markers like `(1)` or `①`.

### Archive (档案) processing

- `档案字段过滤.py` — Strips archive JSON files down to essential fields (年号, 题名, 内容提要, 责任者, 责任时间 variants). Reads from `数据基础/档案/`, writes to `数据基础/档案_过滤/`.
- `archive_search.py` — The main search engine (see below).

### Dictionary (辞典) data

Excel files in `数据基础/辞典/` are reference data. Not yet integrated into the search pipeline.

## Search engine

### v3: Hybrid retrieval (`archive_retriever.py`) ← 当前主检索器

**状态：双索引已完整构建，交互式检索可用。** 支持两种数据格式，默认使用 refined（现代文摘要）。

**架构**: 查询处理 → BM25+向量 双通道并行检索 → RRF融合 → 关系加权重排序 → Top-K结果

```
用户查询 → OpenCC简繁转换 → jieba分词(清史词典) → 实体抽取
         → 并行双通道:
             BM25 关键词检索 (多字段加权) → Top-100
             向量语义检索 (BGE-large-zh-v1.5 + FAISS IndexFlatIP) → Top-100
         → RRF融合(k=60) → Top-50候选集
         → 关系加权重排序(4维):
             实体共现 30% / 机构隶属 30% / 时序邻近 20% / 地理关联 20%
         → Top-K 最终结果
```

#### 两种数据格式

| 格式 | 数据源 | 缓存目录 | BM25 策略 | 向量策略 | 实体策略 |
|------|--------|----------|-----------|----------|----------|
| `refined` (**默认**) | `数据基础/档案refined/` | `cache_refined/` | modern_summary×2 + source_text×1 | modern_summary | 预提取(people/office/geo) + 文本提取(reign/title/event) 融合 |
| `filtered` | `数据基础/档案_过滤/` | `cache/` | 题名×3 + 内容提要×1 + 责任者×1 | 题名+内容提要+责任者 | 正则提取（全部类型） |

核心类: `ArchiveRetriever` — 主 Agent 调用接口
- `ArchiveRetriever(data_format="refined")` — 默认，现代文摘要索引
- `ArchiveRetriever()` 或 `ArchiveRetriever(data_format="filtered")` — 文言文原文索引（旧版）
- `load_or_build()` — 首次构建索引(~1-2h, GPU)，后续秒级加载缓存。**可以脱离 `数据基础/` 源数据运行**，源目录不存在时自动跳过时间戳检查。
- `search(query, top_k=10)` — 混合检索，返回带多维度评分的排序结果。

缓存文件：
- `cache_refined/` (refined 索引, ~1.6GB): 244,338 条记录
  - `bm25_index.pkl` (559MB) — BM25索引 + 分词语料 + 全部记录
  - `faiss_index.bin` (~980MB) — FAISS向量索引 (244,338 × 1024 × float32)
  - `doc_entities.pkl` — 每篇文档的实体标注
- `cache/` (filtered 索引, ~1.2GB): 248,342 条记录（旧版，不动）

实体词典: 年号14个、机构60+个、官职80+个、地名90+个、人名60+个、事件术语30+个，
覆盖清代核心政务、军事、财政、外交、河工等领域。

运行交互式检索（默认 modern）:
```bash
python archive_retriever.py              # refined（现代文摘要，默认）
python archive_retriever.py filtered     # filtered（文言文原文）
```

新旧对比脚本:
```bash
python compare_retrievers.py
```

Embedding 模型: `BAAI/bge-large-zh-v1.5`，本地存放于 `models/bge-large-zh-v1.5/`(1.24GB)。

**调用方式**:
```python
from archive_retriever import ArchiveRetriever

# 现代文摘要（默认，推荐）
r = ArchiveRetriever(data_format="refined")
r.load_or_build()  # 秒级从缓存加载
results = r.search("乾隆年间福建水师海防", top_k=10)

# 文言文原文（旧版）
r_old = ArchiveRetriever(data_format="filtered")
r_old.load_or_build()
```
每条结果 `results[i]` 包含 `record`（refined格式含 `refined_data.modern_summary` 现代文摘要 + `source_text` 文言文原文）、`final_score`（综合分）、`bm25_score`/`vector_score`（双通道分）、`rrf_score`（融合分）、`relation_score`（关系总分）及四维分解（`entity_score`/`inst_score`/`temporal_score`/`geo_score`）、`matched_entities`（匹配实体）。`record` 的键已转为简体中文。

**已知坑点**:
- faiss C++ 底层 `fopen` 不支持中文路径，读写 `faiss_index.bin` 时需先 `os.chdir(cache_dir)` 用纯英文相对文件名
- `np.save` 若文件名不以 `.npy` 结尾会自动追加 `.npy`
- Windows 上的 bash（Git Bash 等）对中文路径支持差，建议用 Python 直接操作文件系统，不要用 `find`/`ls`/`dir`

**交付物**: `档案检索模块.zip` (1.63GB) 和 `档案检索模块_交接说明.md` 位于项目根目录。

### v2: Keyword search + LLM rerank (`archive_search.py`)

Core classes:

- `ArchiveSearchEngine` — Loads JSON from `数据基础/档案_过滤/`, builds BM25 index over jieba-tokenized text. Caches index to `.archive_bm25_cache.pkl` (~300MB). Auto-rebuilds when JSON files are newer than cache.
- `LLMAnswerGenerator` — Wraps an OpenAI-compatible API to generate cited answers from search results.

The jieba tokenizer is pre-loaded with ~40 Qing Dynasty specialized terms (官职, 机构, 制度, 人名) to prevent incorrect word segmentation. Uses OpenCC for traditional-to-simplified Chinese conversion.

Run interactively:
```bash
python archive_search.py
```
Set `USE_LLM = True` in the script to enable LLM-generated answers with citations.

## Fine-tuning data construction

Two scripts call LLM APIs to structure historical texts into training data:

- `编年微调数据构建.py` — Processes chronicle CSV rows, outputs `hardcore_history_training_data.json`
- `档案微调数据构建.py` — Processes archive JSON records with cross-dynasty proportional sampling. Modes controlled by `TEST_MODE` and `SAMPLE_TOTAL` flags. Outputs `archive_training_data_sample_{N}.json` or `archive_training_data_test_{dynasty}.json`

Both use `AsyncOpenAI` with semaphore-based concurrency control. The archive script includes: 3-retry with exponential backoff, checkpoint saves every 10 records, and a three-layer JSON extraction strategy (direct parse → code block extraction → brace matching).

## Frontend

`清鉴-清史研究智能体.html` is a standalone single-file web app. Left-right dual-panel layout: chat on the left, citation reference cards on the right. Pure CSS/JS with no build step. The JS includes a mock data fallback when the backend `/api/chat` endpoint is unavailable.

## Data directory layout

```
数据基础/
├── 档案/           # Raw archive JSON (10 dynasties, ~248K records, traditional Chinese)
├── 档案_过滤/      # Field-filtered archive JSON (文言文原文检索用)
├── 档案refined/    # 微调后的现代文摘要档案 JSON (10 dynasties, ~244K records, 含 modern_summary)
├── 编年/           # Structured chronicle CSVs (康熙/光绪/宣统)
├── 辞典/           # Reference Excel files (历史大辞典, 文学家大辞典, 近代史词典, 故宫辞典)
原始文件/            # Original markdown + PDF chronicle source files
第三组/              # Third group work: 咸丰/同治 MD sources + output CSVs
cache/              # filtered 索引缓存 (文言文, ~1.2GB)
cache_refined/      # refined 索引缓存 (现代文摘要, ~1.6GB)
models/             # 本地 embedding 模型 (bge-large-zh-v1.5)
compare_retrievers.py  # 新旧检索器对比脚本
```

CSV encoding note: chronicle CSVs use `gb18030` (for compatibility with rare characters in classical texts). Archive JSON uses UTF-8.
