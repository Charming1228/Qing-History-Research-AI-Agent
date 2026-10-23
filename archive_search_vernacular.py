"""
清史档案检索 — 文言 vs 白话 & BM25 vs 语义 检索效果对比工具
=============================================================
在原有 248K 文言档案索引（archive_search.py）基础上，
新建 3000 条白话档案索引，支持三种对比维度：
  1. 文言 BM25 vs 白话 BM25（跨语料对比）
  2. BM25 稀疏检索 vs 语义稠密检索（同语料方法对比）

用法：
  python archive_search_vernacular.py                              # 交互式（文言vs白话）
  python archive_search_vernacular.py --mode bm25-vs-semantic      # 交互式（BM25 vs 语义）
  python archive_search_vernacular.py --batch                      # 批量测试（文言vs白话）
  python archive_search_vernacular.py --batch --mode bm25-vs-semantic  # 批量测试（BM25 vs 语义）
"""

import json
import os
import sys
import jieba
import numpy as np
from rank_bm25 import BM25Okapi
from opencc import OpenCC

# 语义检索依赖（sentence-transformers），缺失时降级提示
_SEMANTIC_AVAILABLE = True
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    _SEMANTIC_AVAILABLE = False
    SentenceTransformer = None

_cc = OpenCC('t2s')

# ============================================================
# 清代专有名词词典（与 archive_search.py 保持一致）
# ============================================================
_QING_TERMS = [
    "掌印郎中", "员外郎", "笔帖式", "给事中", "协办大学士", "军机大臣",
    "总督", "巡抚", "布政使", "按察使", "提督", "总兵", "参将", "游击",
    "都统", "副都统", "参领", "佐领", "领催", "内阁学士", "侍读学士",
    "监察御史", "道员", "知府", "知州", "知县", "同知", "通判",
    "钦差大臣", "办事大臣", "领队大臣", "参赞大臣",
    "军机处", "内务府", "理藩院", "都察院", "大理寺", "太常寺",
    "光禄寺", "太仆寺", "鸿胪寺", "国子监", "钦天监", "宗人府",
    "吏部", "户部", "礼部", "兵部", "刑部", "工部",
    "总理衙门", "总理各国事务衙门", "外务部",
    "稽察房", "典籍厅", "档房",
    "八旗", "满洲八旗", "蒙古八旗", "汉军八旗", "绿营",
    "科举", "乡试", "会试", "殿试", "引见", "议叙", "议处",
    "丁忧", "起复", "休致", "原品休致",
    "伊犁", "乌里雅苏台", "盛京", "乌鲁木齐", "喀什噶尔",
    "福康安", "和珅", "曾国藩", "左宗棠", "李鸿章", "张之洞",
    "刘铭传", "沈葆桢", "林则徐", "琦善", "僧格林沁", "袁世凯",
    "太平天国", "义和团", "甲午战争", "鸦片战争", "洋务运动",
    "百日维新", "戊戌变法", "辛亥革命", "捻军",
    "光绪皇帝", "光绪帝", "慈禧太后", "慈禧",
    "咸丰皇帝", "同治皇帝", "道光皇帝", "嘉庆皇帝",
    "乾隆皇帝", "雍正皇帝", "康熙皇帝", "顺治皇帝",
    "北洋水师", "北洋海军", "南洋水师",
    "虎门销烟", "安庆之战", "马关条约", "南京条约", "北京条约",
    "瀛台", "圆明园", "颐和园",
]
for _term in _QING_TERMS:
    jieba.add_word(_term)

_BM25_STOP_WORDS = {
    "的", "是", "还是", "了", "吗", "呢", "吧", "啊", "呀",
    "什么", "怎么", "怎样", "如何", "为什么", "为何",
    "多少", "几个", "哪", "哪个", "哪些", "谁",
    "不", "没", "很", "都", "就", "也", "会", "要", "能", "可以",
    "在", "和", "与", "或", "及", "之", "等", "其", "这", "那",
    "有", "被", "把", "从", "到", "对", "向", "将", "以",
    "一个", "一种", "这个", "那个", "中", "后", "前", "上", "下",
    "是否", "有无", "可否", "怎么办", "怎样办",
}

# ============================================================
# 语义检索引擎 — 基于 sentence-transformers 的稠密向量检索
# ============================================================
DEFAULT_SEMANTIC_MODEL = "BAAI/bge-small-zh-v1.5"


class SemanticRetriever:
    """语义检索器：将文档编码为稠密向量，通过余弦相似度检索。

    使用 BGE 中文模型，对白话文和文言文均有较好的语义理解能力。
    支持对 3000~10000 条文档的暴力检索（无需向量数据库）。
    """

    def __init__(self, model_name=DEFAULT_SEMANTIC_MODEL):
        if not _SEMANTIC_AVAILABLE:
            raise ImportError(
                "语义检索需要 sentence-transformers 库。请运行:\n"
                "  pip install sentence-transformers\n"
                "若从 HuggingFace 下载模型较慢，可设置镜像:\n"
                "  export HF_ENDPOINT=https://hf-mirror.com"
            )
        print(f"[语义引擎] 加载模型 {model_name} ...")
        self.model = SentenceTransformer(model_name)
        self.doc_embeddings = None
        self.records = []

    def encode(self, texts, show_progress=True):
        """将文本列表编码为归一化向量"""
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
            batch_size=64,
        )

    def build_index(self, records, doc_texts):
        """构建语义索引：编码所有文档"""
        self.records = list(records)
        print(f"[语义引擎] 正在编码 {len(doc_texts)} 条文档...")
        self.doc_embeddings = self.encode(doc_texts)
        print(f"[语义引擎] 索引构建完成，向量维度 {self.doc_embeddings.shape[1]}。")

    def search(self, query, top_k=10):
        """语义检索：query → embedding → cosine similarity → top_k"""
        if self.doc_embeddings is None:
            raise RuntimeError("语义索引尚未构建，请先调用 build_index()")
        query_vec = self.encode([query], show_progress=False)
        scores = (query_vec @ self.doc_embeddings.T).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            results.append({
                'index': int(idx),
                'score': round(float(scores[idx]), 4),
            })
        return results


# ============================================================
# 批量测试问题（聚焦 3000 条数据覆盖的主题）
# ============================================================
TEST_QUERIES = [
    "乾隆年间和珅在朝中担任过哪些官职？",
    "清代地方官员如何向朝廷奏报钱粮征收的完成情况？",
    "乾隆年间直隶地区发生水灾后朝廷如何赈济灾民？",
    "清朝刑部是怎样审理和判决人命案件的？",
    "清代官员丁忧守制结束后如何重新起复任用？",
    "清朝江南地区的河道水利工程是如何修建和核销银两的？",
    "清代台湾地区有哪些军事防务部署和驻防情况？",
    "清朝户部日常处理哪些财政事务和职能？",
    "乾隆年间博学鸿词科是如何选拔和举荐人才的？",
    "道光年间各省驻军的军饷和红白赏恤银两如何筹措？",
    "清朝吏部如何进行官员的考核升迁和调补？",
    "清代如何处理官员贪污渎职等案件？",
    "乾隆年间新疆地区的军事换防是怎样安排的？",
    "清代礼部如何筹备朝廷的筵宴和祭祀典礼？",
    "清朝如何处理涉外通商口岸和关税事务？",
]


class VernacularSearchEngine:
    """白话文档案检索引擎 — 基于 3000 条 LLM 改写数据"""

    def __init__(self, json_path):
        self.json_path = json_path
        self.records = []
        self.tokenized_corpus = []
        self.bm25 = None
        self.semantic = None  # SemanticRetriever, 需调用 build_semantic_index() 初始化

    def load_data(self):
        with open(self.json_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)

        for item in raw:
            gpt_val = item['conversations'][1]['value']
            try:
                parsed = json.loads(gpt_val)
            except (json.JSONDecodeError, KeyError):
                continue

            summary = parsed.get('modern_summary', '')
            if not summary:
                continue

            self.records.append({
                'modern_summary': summary,
                'entities_people': parsed.get('entities_people', []),
                'entities_office': parsed.get('entities_office', []),
                'entities_geo': parsed.get('entities_geo', []),
                'document_type': parsed.get('document_type', ''),
                'event_type': parsed.get('event_type', ''),
            })

        print(f"[白话引擎] 已加载 {len(self.records)} 条白话档案记录")
        return self

    def _build_doc_text(self, record):
        """构建文档文本：摘要 + 实体拼接"""
        parts = [record['modern_summary']]
        if record['entities_people']:
            parts.append(' '.join(record['entities_people']))
        if record['entities_office']:
            parts.append(' '.join(record['entities_office']))
        if record['entities_geo']:
            parts.append(' '.join(record['entities_geo']))
        return ' '.join(parts)

    def build_index(self):
        print("[白话引擎] 正在构建 BM25 索引...")
        self.tokenized_corpus = []
        for i, rec in enumerate(self.records):
            tokens = list(jieba.cut(self._build_doc_text(rec)))
            self.tokenized_corpus.append(tokens)
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        print(f"[白话引擎] 索引构建完成，共 {len(self.records)} 条记录。")
        return self

    def search(self, query, top_k=10):
        """BM25 检索白话档案"""
        if self.bm25 is None:
            raise RuntimeError("索引尚未构建")

        query_simp = _cc.convert(query)
        query_tokens = [t for t in jieba.cut(query_simp)
                        if t.strip() and t not in _BM25_STOP_WORDS]
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            rec = self.records[idx]
            results.append({
                'index': int(idx),
                'score': round(float(scores[idx]), 2),
                '现代摘要': rec['modern_summary'],
                '人物': rec['entities_people'],
                '机构': rec['entities_office'],
                '地名': rec['entities_geo'],
                '文档类型': rec['document_type'],
                '事件类型': rec['event_type'],
            })
        return results

    def build_semantic_index(self, model_name=DEFAULT_SEMANTIC_MODEL):
        """构建语义索引：将所有白话文档编码为稠密向量"""
        print("[白话引擎] 正在构建语义索引...")
        self.semantic = SemanticRetriever(model_name)
        doc_texts = [self._build_doc_text(rec) for rec in self.records]
        self.semantic.build_index(self.records, doc_texts)
        return self

    def semantic_search(self, query, top_k=10):
        """语义检索白话档案（稠密向量 + 余弦相似度）"""
        if self.semantic is None:
            raise RuntimeError("语义索引尚未构建，请先调用 build_semantic_index()")
        raw_results = self.semantic.search(query, top_k=top_k)
        results = []
        for r in raw_results:
            rec = self.records[r['index']]
            results.append({
                'index': r['index'],
                'score': r['score'],
                '现代摘要': rec['modern_summary'],
                '人物': rec['entities_people'],
                '机构': rec['entities_office'],
                '地名': rec['entities_geo'],
                '文档类型': rec['document_type'],
                '事件类型': rec['event_type'],
            })
        return results


# ============================================================
# 对比器 — 同时查询两个引擎并排展示
# ============================================================
class SearchComparator:
    """文言 / 白话 检索对比器"""

    def __init__(self, classical_engine, vernacular_engine):
        self.classical = classical_engine
        self.vernacular = vernacular_engine

    def compare(self, query, top_k=5):
        """对同一查询分别检索两个索引，返回并排结果"""
        print()
        print("=" * 70)
        print(f"  查询: {query}")
        print("=" * 70)

        # 白话检索
        v_results = self.vernacular.search(query, top_k=top_k)
        # 文言检索（关闭查询扩展，纯 BM25 对比更公平）
        c_results = self.classical.search(query, top_k=top_k, use_expansion=False)

        # ── 白话结果 ──
        print()
        print("┌" + "─" * 68 + "┐")
        print("│" + "  📖 白话文索引 (3000条 LLM改写) ".ljust(69) + "│")
        print("└" + "─" * 68 + "┘")
        self._print_vernacular_results(v_results)

        # ── 文言结果 ──
        print()
        print("┌" + "─" * 68 + "┐")
        print("│" + "  📜 文言文索引 (248K条 原始档案) ".ljust(69) + "│")
        print("└" + "─" * 68 + "┘")
        self._print_classical_results(c_results)

        return v_results, c_results

    def _print_vernacular_results(self, results):
        if not results:
            print("  (无结果)")
            return
        for i, r in enumerate(results):
            print(f"  [{i+1}] 得分={r['score']:.2f} | {r['事件类型']} | {r['文档类型']}")
            print(f"      {r['现代摘要'][:150]}")
            if r['人物']:
                print(f"      人物: {' · '.join(r['人物'][:5])}")
            if r['地名']:
                print(f"      地名: {' · '.join(r['地名'][:5])}")
            print()

    def _print_classical_results(self, results):
        if not results:
            print("  (无结果)")
            return
        for i, r in enumerate(results):
            print(f"  [{i+1}] 得分={r['score']:.2f} | {r['来源朝代']}朝 | {r.get('年号','')}")
            print(f"      题名: {r['题名'][:120]}")
            print(f"      提要: {r['内容提要'][:150]}")
            print(f"      责任者: {r['责任者']} | {r['责任时间(中历)']}")
            print()

    # ── BM25 vs 语义检索 对比 ──
    def compare_methods(self, query, corpus="vernacular", top_k=5):
        """同一语料库上 BM25 与语义检索并排对比

        corpus: "vernacular" (3000条白话) 或 "classical" (文言采样)
        """
        print()
        print("=" * 70)
        print(f"  查询: {query}")
        print(f"  对比: BM25(稀疏) vs Semantic(稠密) — 语料: {corpus}")
        print("=" * 70)

        if corpus == "vernacular":
            bm25_results = self.vernacular.search(query, top_k=top_k)
            semantic_results = self.vernacular.semantic_search(query, top_k=top_k)
            print_func = self._print_vernacular_results
            bm25_label = "📖 BM25 稀疏检索 (白话 3000条)"
            sem_label = "🧠 语义稠密检索 (白话 3000条)"
        else:
            bm25_results = self.classical.search(query, top_k=top_k, use_expansion=False)
            semantic_results = self.classical.semantic_search(query, top_k=top_k)
            print_func = self._print_classical_results
            bm25_label = "📖 BM25 稀疏检索 (文言 采样)"
            sem_label = "🧠 语义稠密检索 (文言 采样)"

        # ── BM25 结果 ──
        print()
        print("┌" + "─" * 68 + "┐")
        print("│" + f"  {bm25_label} ".ljust(69) + "│")
        print("└" + "─" * 68 + "┘")
        print_func(bm25_results)

        # ── 语义结果 ──
        print()
        print("┌" + "─" * 68 + "┐")
        print("│" + f"  {sem_label} ".ljust(69) + "│")
        print("└" + "─" * 68 + "┘")
        self._print_semantic_results(semantic_results, corpus)

        return bm25_results, semantic_results

    def _print_semantic_results(self, results, corpus="vernacular"):
        """打印语义检索结果（余弦相似度分数）"""
        if not results:
            print("  (无结果)")
            return
        if corpus == "vernacular":
            for i, r in enumerate(results):
                print(f"  [{i+1}] 相似度={r['score']:.4f} | {r['事件类型']} | {r['文档类型']}")
                print(f"      {r['现代摘要'][:150]}")
                if r['人物']:
                    print(f"      人物: {' · '.join(r['人物'][:5])}")
                if r['地名']:
                    print(f"      地名: {' · '.join(r['地名'][:5])}")
                print()
        else:
            for i, r in enumerate(results):
                print(f"  [{i+1}] 相似度={r['score']:.4f} | "
                      f"{r.get('来源朝代','')}朝 | {r.get('年号','')}")
                print(f"      题名: {r.get('题名','')[:120]}")
                print(f"      提要: {r.get('内容提要','')[:150]}")
                print(f"      责任者: {r.get('责任者','')} | {r.get('责任时间(中历)','')}")
                print()


# ============================================================
# 辅助：为文言引擎构建语义索引（随机采样）
# ============================================================
def _build_classical_semantic(engine, sample_size=5000, model_name=DEFAULT_SEMANTIC_MODEL):
    """从文言全量数据中随机采样，构建语义稠密索引并注入到 engine 上"""
    n_total = len(engine.records)
    n_sample = min(sample_size, n_total)
    print(f"[文言引擎] 正在从 {n_total} 条中采样 {n_sample} 条构建语义索引...")

    rng = np.random.RandomState(42)
    indices = rng.choice(n_total, size=n_sample, replace=False)

    sampled_records = [engine.records[i] for i in indices]
    doc_texts = [engine._build_doc_text(rec) for rec in sampled_records]

    retriever = SemanticRetriever(model_name)
    retriever.build_index(sampled_records, doc_texts)

    engine._semantic = retriever
    engine._semantic_records = sampled_records

    def semantic_search(query, top_k=10):
        results = []
        for r in retriever.search(query, top_k=top_k):
            rec = sampled_records[r['index']]
            results.append({
                'index': r['index'],
                'score': r['score'],
                '题名': _cc.convert(rec.get('題名', '') or ''),
                '内容提要': _cc.convert(rec.get('內容提要', '') or ''),
                '年号': _cc.convert(rec.get('年号', '') or ''),
                '责任者': _cc.convert(rec.get('責任者', '') or ''),
                '责任时间(中历)': _cc.convert(rec.get('責任時間(中曆)', '') or ''),
                '来源朝代': _cc.convert(rec.get('来源朝代', '') or ''),
            })
        return results

    engine.semantic_search = semantic_search
    print(f"[文言引擎] 语义索引就绪 ({n_sample}条)。")


# ============================================================
# 主程序
# ============================================================
def main():
    sys.stdout.reconfigure(encoding='utf-8')

    # ── 路径配置 ──
    BASE = os.path.dirname(os.path.abspath(__file__))
    CLASSICAL_DATA_DIR = os.path.join(BASE, "数据基础", "档案_过滤")
    VERNACULAR_JSON = os.path.join(BASE, "archive_training_data_sample_3000.json")

    # ── 模式解析 ──
    # --mode classical-vs-vernacular (默认): 文言BM25 vs 白话BM25
    # --mode bm25-vs-semantic: BM25 vs 语义检索 对比
    # --batch: 批量测试模式
    MODE = "classical-vs-vernacular"
    if "--mode" in sys.argv:
        idx = sys.argv.index("--mode")
        if idx + 1 < len(sys.argv):
            MODE = sys.argv[idx + 1]
    IS_BATCH = "--batch" in sys.argv

    # ── 标题 ──
    print("=" * 70)
    if MODE == "bm25-vs-semantic":
        print("  清史档案检索 — BM25(稀疏) vs Semantic(稠密) 对比工具")
    else:
        print("  清史档案检索 — 文言 vs 白话 对比工具")
    print("=" * 70)
    print()

    # ── 初始化文言引擎 ──
    from archive_search import ArchiveSearchEngine
    print(">>> 加载文言文索引 (全量 248K)...")
    classical_engine = ArchiveSearchEngine(CLASSICAL_DATA_DIR)
    classical_engine.load_data()
    classical_engine.build_index()

    # ── 初始化白话引擎 ──
    print()
    print(">>> 加载白话文索引 (3000条)...")
    vernacular_engine = VernacularSearchEngine(VERNACULAR_JSON)
    vernacular_engine.load_data()
    vernacular_engine.build_index()

    # ── 构建语义索引 ──
    if MODE == "bm25-vs-semantic":
        if not _SEMANTIC_AVAILABLE:
            print("\n" + "!" * 70)
            print("  语义检索需要 sentence-transformers 库。请运行:")
            print("    pip install sentence-transformers")
            print("  若从 HuggingFace 下载模型较慢，可设置镜像:")
            print("    set HF_ENDPOINT=https://hf-mirror.com    (Windows)")
            print("    export HF_ENDPOINT=https://hf-mirror.com  (Linux/Mac)")
            print("!" * 70)
            sys.exit(1)

        print()
        print(">>> 构建白话文语义索引 (3000条)...")
        vernacular_engine.build_semantic_index()

        print()
        print(">>> 构建文言文语义索引 (采样 5000条)...")
        _build_classical_semantic(classical_engine, sample_size=5000)

    comparator = SearchComparator(classical_engine, vernacular_engine)

    # ── 启动 ──
    if IS_BATCH:
        run_batch(comparator, mode=MODE)
    else:
        run_interactive(comparator, mode=MODE)


def run_batch(comparator, mode="classical-vs-vernacular"):
    """批量测试模式

    mode:
      classical-vs-vernacular — 文言BM25 vs 白话BM25
      bm25-vs-semantic       — BM25 vs 语义检索 (白话+文言论流)
    """
    print()
    print("=" * 70)
    print(f"  批量测试 — 共 {len(TEST_QUERIES)} 个查询")
    if mode == "bm25-vs-semantic":
        print("  对比模式: BM25(稀疏) vs Semantic(稠密)")
        print("  对每个查询分别在 [白话3000] 和 [文言采样5000] 上对比")
    else:
        print("  对比模式: 文言(248K) vs 白话(3000) BM25")
    print("=" * 70)

    if mode == "bm25-vs-semantic":
        _run_batch_semantic(comparator)
    else:
        _run_batch_classical_vs_vernacular(comparator)


def _run_batch_classical_vs_vernacular(comparator):
    """批量测试：文言 vs 白话 BM25"""
    summary = []
    for i, q in enumerate(TEST_QUERIES):
        v_res, c_res = comparator.compare(q, top_k=5)
        summary.append({
            'query': q,
            'vernacular_hits': len(v_res),
            'vernacular_top_score': v_res[0]['score'] if v_res else 0,
            'classical_hits': len(c_res),
            'classical_top_score': c_res[0]['score'] if c_res else 0,
        })

    _print_summary_table_cv(summary)
    print()
    print("  注意：文言索引 248K 条，白话索引仅 3000 条，直接比较分数无意义。")
    print("  请重点关注：返回结果的主题相关性、信息的完整性和准确性。")


def _run_batch_semantic(comparator):
    """批量测试：BM25 vs 语义检索（白话 + 文言论流各测一次）"""
    for corpus_name in ["vernacular", "classical"]:
        print()
        print(f"  {'─' * 60}")
        print(f"  语料: {corpus_name} ({'3000条白话' if corpus_name == 'vernacular' else '5000条文言论样'})")
        print(f"  {'─' * 60}")

        summary = []
        for i, q in enumerate(TEST_QUERIES):
            bm25_res, sem_res = comparator.compare_methods(
                q, corpus=corpus_name, top_k=5
            )
            summary.append({
                'query': q,
                'bm25_hits': len(bm25_res),
                'bm25_top_score': bm25_res[0]['score'] if bm25_res else 0,
                'semantic_hits': len(sem_res),
                'semantic_top_score': sem_res[0]['score'] if sem_res else 0,
            })

        _print_summary_table_bs(summary)

    # ── 综合说明 ──
    print()
    print("=" * 70)
    print("  关于分数的说明")
    print("=" * 70)
    print("  BM25 分数: 无上界，受文档长度和词频影响，不同语料间不可比。")
    print("  语义相似度: [0, 1] 范围，余弦相似度，可跨语料/跨查询比较。")
    print("  白话 (3000条): 现代汉语，语义模型理解更好，相似度通常较高。")
    print("  文言 (采样5000): 古文表达，语义模型可能不如白话准确。")
    print()
    print("  评价建议:")
    print("  - 关注每列分数在同查询下的相对排序")
    print("  - 对比 BM25 和 Semantic 返回结果的差异")
    print("  - 判断哪种方法的 Top-5 结果更贴合查询意图")


def _print_summary_table_cv(summary):
    """打印「文言 vs 白话」批量测试汇总表格"""
    if not summary:
        return
    print()
    print(f"  {'':<2} {'查询':<46} {'白话命中':>6} {'白话Top分':>8} {'文言命中':>6} {'文言Top分':>8}")
    print(f"  {'':->2} {'':->46} {'':->6} {'':->8} {'':->6} {'':->8}")
    for i, s in enumerate(summary):
        print(f"  {i+1:<2} {s['query']:<46} {s['vernacular_hits']:>6} "
              f"{s['vernacular_top_score']:>8.2f} {s['classical_hits']:>6} "
              f"{s['classical_top_score']:>8.2f}")

    v_avg_h = sum(s['vernacular_hits'] for s in summary) / len(summary)
    v_avg_s = sum(s['vernacular_top_score'] for s in summary) / len(summary)
    c_avg_h = sum(s['classical_hits'] for s in summary) / len(summary)
    c_avg_s = sum(s['classical_top_score'] for s in summary) / len(summary)

    print(f"  {'':->2} {'':->46} {'':->6} {'':->8} {'':->6} {'':->8}")
    print(f"  {'平均':<2} {'':46} {v_avg_h:>6.1f} {v_avg_s:>8.2f} "
          f"{c_avg_h:>6.1f} {c_avg_s:>8.2f}")


def _print_summary_table_bs(summary):
    """打印「BM25 vs Semantic」批量测试汇总表格"""
    if not summary:
        return
    print()
    print(f"  {'':<2} {'查询':<46} {'BM25命中':>6} {'BM25分':>8} {'语义命中':>6} {'语义相似度':>8}")
    print(f"  {'':->2} {'':->46} {'':->6} {'':->8} {'':->6} {'':->8}")
    for i, s in enumerate(summary):
        print(f"  {i+1:<2} {s['query']:<46} {s['bm25_hits']:>6} "
              f"{s['bm25_top_score']:>8.2f} {s['semantic_hits']:>6} "
              f"{s['semantic_top_score']:>8.4f}")

    b_avg_h = sum(s['bm25_hits'] for s in summary) / len(summary)
    b_avg_s = sum(s['bm25_top_score'] for s in summary) / len(summary)
    s_avg_h = sum(s['semantic_hits'] for s in summary) / len(summary)
    s_avg_s = sum(s['semantic_top_score'] for s in summary) / len(summary)

    print(f"  {'':->2} {'':->46} {'':->6} {'':->8} {'':->6} {'':->8}")
    print(f"  {'平均':<2} {'':46} {b_avg_h:>6.1f} {b_avg_s:>8.2f} "
          f"{s_avg_h:>6.1f} {s_avg_s:>8.4f}")


def run_interactive(comparator, mode="classical-vs-vernacular"):
    """交互式查询模式

    mode:
      classical-vs-vernacular — 文言BM25 vs 白话BM25
      bm25-vs-semantic       — BM25 vs 语义检索 对比
    """
    # 检测语义索引是否可用
    _semantic_ready = (
        comparator.vernacular.semantic is not None
        and hasattr(comparator.classical, 'semantic_search')
    )

    print()
    print("=" * 70)
    if mode == "bm25-vs-semantic":
        print("  交互式对比查询 — BM25(稀疏) vs Semantic(稠密)")
        print("  默认在「白话3000条」上对比")
        print("  命令: 'v' 切到白话 | 'c' 切到文言采样 | 'b' 批量测试")
    else:
        print("  交互式对比查询 — 文言 vs 白话 BM25")
        print("  输入自然语言问题，同时查看文言/白话两边检索结果")
        print("  命令: 'b' 批量测试")
    if _semantic_ready:
        print("  命令: 'm' 切换对比模式 | 'q' 退出 | '?' 预设问题列表")
    else:
        print("  命令: 'q' 退出 | '?' 预设问题列表")
        if mode != "bm25-vs-semantic":
            print("  提示: 用 --mode bm25-vs-semantic 启动可启用语义检索对比")
    print("=" * 70)

    current_corpus = "vernacular"

    while True:
        try:
            query = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not query:
            continue
        if query.lower() in ('q', 'quit', 'exit'):
            print("再见！")
            break
        if query == 'm':
            if not _semantic_ready:
                print("  语义索引未构建。请用 --mode bm25-vs-semantic 重新启动。")
                continue
            if mode == "classical-vs-vernacular":
                mode = "bm25-vs-semantic"
                print("  已切换到 BM25 vs 语义检索 模式 (白话3000条)")
            else:
                mode = "classical-vs-vernacular"
                print("  已切换到 文言 vs 白话 BM25 模式")
            continue
        if query == 'b':
            run_batch(comparator, mode=mode)
            continue
        if query == '?':
            print("\n预设测试问题：")
            for i, q in enumerate(TEST_QUERIES):
                print(f"  [{i+1}] {q}")
            continue
        if mode == "bm25-vs-semantic":
            if query == 'v':
                current_corpus = "vernacular"
                print("  当前语料: 白话 (3000条)")
                continue
            if query == 'c':
                current_corpus = "classical"
                print("  当前语料: 文言 (采样5000条)")
                continue

            comparator.compare_methods(query, corpus=current_corpus, top_k=5)
        else:
            comparator.compare(query, top_k=5)


if __name__ == "__main__":
    main()
