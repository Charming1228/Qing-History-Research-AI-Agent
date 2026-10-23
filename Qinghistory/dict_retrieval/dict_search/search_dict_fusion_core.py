"""
函数式辞典融合检索核心模块。

默认资源目录：本文件所在目录。
请将以下资源放在与本文件相同的目录结构下：
- qing_dict_milvus.db
- dict_chunks.jsonl
- alias_map.json
- models/bge-m3/

最简单调用：
    from search_dict_fusion_core import search_dict
    output = search_dict("军机处")

推荐重复调用方式，避免每次重复加载模型和索引：
    from search_dict_fusion_core import DictFusionSearcher
    searcher = DictFusionSearcher()
    output = searcher.search("军机处")
"""

import contextlib
import heapq
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from FlagEmbedding import BGEM3FlagModel
from pymilvus import MilvusClient


# =========================
# Relative default paths
# =========================

DEFAULT_BASE_DIR = Path(__file__).parent
DEFAULT_MILVUS_URI = "qing_dict_milvus.db"
DEFAULT_COLLECTION_NAME = "qing_dict"
DEFAULT_MODEL_PATH = "models/bge-m3"
DEFAULT_CHUNK_PATH = "dict_chunks.jsonl"
DEFAULT_ALIAS_PATH = "alias_map.json"

DEFAULT_BM25_TOP_K = 20
DEFAULT_VECTOR_TOP_K = 20
DEFAULT_FINAL_TOP_K = 10
DEFAULT_MAX_EXACT_HITS = 8

OUTPUT_FIELDS = [
    "entry_id",
    "chunk_id",
    "row_id",
    "entry",
    "definition",
    "source",
    "type",
    "period",
    "author",
    "source_file",
    "data_type",
]


# =========================
# Basic utils
# =========================

def path_from_base(base_dir: Path, path_value: str | Path) -> Path:
    """把相对路径解释为相对 base_dir 的路径。"""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return base_dir / path


def normalize_text(text: Any) -> str:
    if text is None:
        return ""

    text = str(text).strip()
    text = re.sub(r"\s+", "", text)
    text = text.replace("？", "?")
    text = text.replace("，", ",")
    text = text.replace("。", ".")
    return text


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def make_entity_from_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "entry_id": row.get("entry_id", ""),
        "chunk_id": row.get("chunk_id", ""),
        "row_id": row.get("row_id", ""),
        "entry": row.get("entry", ""),
        "definition": row.get("definition", ""),
        "source": row.get("source", ""),
        "type": row.get("type", ""),
        "period": row.get("period", ""),
        "author": row.get("author", ""),
        "source_file": row.get("source_file", ""),
        "data_type": row.get("data_type", ""),
    }


# =========================
# Load data and alias
# =========================

def load_alias_map(alias_path, verbose=True):

    # =====================================
    # 文件不存在
    # =====================================
    if not os.path.exists(alias_path):

        if verbose:
            print(f"[Warning] Alias file not found: {alias_path}")

        return {}

    # =====================================
    # 空文件
    # =====================================
    if os.path.getsize(alias_path) == 0:

        if verbose:
            print(f"[Warning] Alias file is empty: {alias_path}")

        return {}

    # =====================================
    # 正常读取
    # =====================================
    try:

        with open(alias_path, "r", encoding="utf-8") as f:
            alias_map = json.load(f)

        if verbose:
            print(f"Loaded alias map: {len(alias_map)} aliases")

        return alias_map

    except Exception as e:

        print(f"[Warning] Failed to load alias map: {e}")

        return {}


def expand_query_by_alias(query: str, alias_map: Dict[str, Any]) -> Dict[str, Any]:
    q_norm = normalize_text(query)

    expanded_terms: List[str] = []
    matched_aliases: List[str] = []

    for alias, targets in alias_map.items():
        alias_norm = normalize_text(alias)

        if not alias_norm:
            continue

        if alias_norm in q_norm:
            matched_aliases.append(alias)

            if isinstance(targets, str):
                expanded_terms.append(targets)
            elif isinstance(targets, list):
                expanded_terms.extend(targets)

    seen = set()
    unique_terms = []

    for term in expanded_terms:
        term = str(term).strip()
        if term and term not in seen:
            unique_terms.append(term)
            seen.add(term)

    return {
        "original_query": query,
        "expanded_terms": unique_terms,
        "matched_aliases": matched_aliases,
    }


def load_dict_rows(path: str | Path, verbose: bool = True) -> List[Dict[str, Any]]:
    path = Path(path)
    rows: List[Dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            if row.get("entry") and row.get("definition"):
                rows.append(row)

    if verbose:
        print(f"Loaded dictionary chunks: {len(rows)}")
    return rows


# =========================
# Exact match
# =========================

def is_good_contain_match(entry_norm: str, query_norm: str) -> bool:
    if not entry_norm:
        return False

    # 单字条目只允许完全相等，防止“清”“官”“图”等乱命中
    if len(entry_norm) <= 1:
        return query_norm == entry_norm

    # 两字条目谨慎处理
    if len(entry_norm) == 2:
        if query_norm == entry_norm:
            return True

        if entry_norm in query_norm and len(query_norm) <= 12:
            return True

        return False

    # 三字及以上比较可靠
    return entry_norm in query_norm


def exact_match_search_one_query(
    query: str,
    dict_rows: List[Dict[str, Any]],
    source_label: str = "exact",
    max_hits: int = DEFAULT_MAX_EXACT_HITS,
) -> List[Dict[str, Any]]:
    q_norm = normalize_text(query)
    candidates: List[Dict[str, Any]] = []

    for row in dict_rows:
        entry = row.get("entry", "")
        e_norm = normalize_text(entry)

        if not e_norm:
            continue

        match_type = None
        exact_score = 0.0

        if q_norm == e_norm:
            match_type = "entry_exact"
            exact_score = 100.0

        elif is_good_contain_match(e_norm, q_norm):
            match_type = "entry_in_query"
            exact_score = 90.0 + min(len(e_norm), 10) / 10

        elif len(q_norm) >= 3 and q_norm in e_norm:
            match_type = "query_in_entry"
            exact_score = 75.0 + min(len(q_norm), 10) / 10

        if match_type:
            candidates.append(
                {
                    "id": row.get("id", row.get("chunk_id", "")),
                    "entity": make_entity_from_row(row),
                    "exact_score": exact_score,
                    "bm25_score": None,
                    "bm25_rank_score": 0.0,
                    "vector_score": None,
                    "vector_rank_score": 0.0,
                    "final_score": exact_score,
                    "match_source": source_label,
                    "match_type": match_type,
                }
            )

    candidates.sort(
        key=lambda x: (
            x["exact_score"],
            len(x["entity"].get("entry", "")),
            -int(x["entity"].get("row_id", 0) or 0),
        ),
        reverse=True,
    )

    return candidates[:max_hits]


def exact_match_search(
    query: str,
    expanded_terms: List[str],
    dict_rows: List[Dict[str, Any]],
    max_hits: int = DEFAULT_MAX_EXACT_HITS,
) -> List[Dict[str, Any]]:
    all_hits: List[Dict[str, Any]] = []

    original_hits = exact_match_search_one_query(
        query=query,
        dict_rows=dict_rows,
        source_label="exact",
        max_hits=max_hits,
    )

    all_hits.extend(original_hits)

    for term in expanded_terms:
        alias_hits = exact_match_search_one_query(
            query=term,
            dict_rows=dict_rows,
            source_label="alias_exact",
            max_hits=max_hits,
        )

        for hit in alias_hits:
            hit["exact_score"] = max(hit["exact_score"], 95.0)
            hit["final_score"] = hit["exact_score"]

        all_hits.extend(alias_hits)

    return all_hits


# =========================
# BM25 implementation
# =========================

def tokenize_chinese_like(text: Any) -> List[str]:
    """
    不依赖 jieba，直接做适合中文辞典检索的轻量 tokenization。

    生成：
    - 英文/数字词
    - 中文单字
    - 中文 bigram
    - 中文 trigram
    - 较短中文片段整词
    """
    text = safe_str(text).lower()
    tokens: List[str] = []

    # 英文/数字
    tokens.extend(re.findall(r"[a-z0-9]+", text))

    # 连续中文片段
    cjk_segments = re.findall(r"[\u4e00-\u9fff]+", text)

    for seg in cjk_segments:
        n = len(seg)

        if n == 0:
            continue

        # 单字
        for ch in seg:
            tokens.append(ch)

        # bigram
        for i in range(n - 1):
            tokens.append(seg[i : i + 2])

        # trigram
        for i in range(n - 2):
            tokens.append(seg[i : i + 3])

        # 短术语整词
        if 2 <= n <= 10:
            tokens.append(seg)

    return tokens


def build_bm25_text(row: Dict[str, Any]) -> str:
    """
    给 BM25 用的文本。
    entry 多重复几次，是为了让“条目名命中”权重更高。
    """
    entry = safe_str(row.get("entry", ""))
    type_ = safe_str(row.get("type", ""))
    period = safe_str(row.get("period", ""))
    source = safe_str(row.get("source", ""))
    definition = safe_str(row.get("definition", ""))

    parts: List[str] = []

    # 条目字段加权
    if entry:
        parts.extend([entry] * 6)

    if type_:
        parts.extend([type_] * 2)

    if period:
        parts.append(period)

    if source:
        parts.append(source)

    if definition:
        parts.append(definition)

    return " ".join(parts)


class BM25Index:
    def __init__(self, rows: List[Dict[str, Any]], k1: float = 1.5, b: float = 0.75, verbose: bool = True):
        self.rows = rows
        self.k1 = k1
        self.b = b
        self.N = len(rows)
        self.doc_lens: List[int] = []
        self.avgdl = 0.0
        self.postings: Dict[str, Dict[int, int]] = defaultdict(dict)
        self.idf: Dict[str, float] = {}
        self.verbose = verbose

        self._build()

    def _build(self) -> None:
        total_len = 0
        df: Counter[str] = Counter()

        for doc_id, row in enumerate(self.rows):
            text = build_bm25_text(row)
            tokens = tokenize_chinese_like(text)
            counts = Counter(tokens)

            dl = sum(counts.values())
            self.doc_lens.append(dl)
            total_len += dl

            for token, tf in counts.items():
                self.postings[token][doc_id] = tf
                df[token] += 1

        self.avgdl = total_len / max(self.N, 1)

        for token, freq in df.items():
            self.idf[token] = math.log(1 + (self.N - freq + 0.5) / (freq + 0.5))

        if self.verbose:
            print(f"Built BM25 index: docs={self.N}, avgdl={self.avgdl:.2f}, vocab={len(self.idf)}")

    def search(self, query: str, top_k: int = DEFAULT_BM25_TOP_K) -> List[Tuple[int, float]]:
        query_tokens = tokenize_chinese_like(query)

        if not query_tokens:
            return []

        # 查询 token 去重，避免 query 里重复词过度加权
        query_terms = list(dict.fromkeys(query_tokens))

        scores: Dict[int, float] = defaultdict(float)

        for term in query_terms:
            if term not in self.postings:
                continue

            idf = self.idf.get(term, 0.0)
            posting = self.postings[term]

            for doc_id, tf in posting.items():
                dl = self.doc_lens[doc_id]
                denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                score = idf * (tf * (self.k1 + 1)) / denom
                scores[doc_id] += score

        if not scores:
            return []

        return heapq.nlargest(top_k, scores.items(), key=lambda x: x[1])


def bm25_search(
    bm25_index: BM25Index,
    query: str,
    expanded_terms: List[str],
    top_k: int = DEFAULT_BM25_TOP_K,
) -> List[Dict[str, Any]]:
    bm25_query = query

    if expanded_terms:
        bm25_query += " " + " ".join(expanded_terms)

    raw_hits = bm25_index.search(bm25_query, top_k=top_k)

    hits: List[Dict[str, Any]] = []

    for rank, (doc_id, bm25_score) in enumerate(raw_hits, start=1):
        row = bm25_index.rows[doc_id]

        # rank score 用于融合排序，避免 BM25 原始分数尺度过大或过小
        bm25_rank_score = 45.0 / rank

        hits.append(
            {
                "id": row.get("id", row.get("chunk_id", "")),
                "entity": make_entity_from_row(row),
                "exact_score": 0.0,
                "bm25_score": float(bm25_score),
                "bm25_rank_score": bm25_rank_score,
                "vector_score": None,
                "vector_rank_score": 0.0,
                "final_score": bm25_rank_score,
                "match_source": "bm25",
                "match_type": "bm25_lexical",
            }
        )

    return hits


# =========================
# Vector search
# =========================

def load_model(model_path: str | Path, verbose: bool = True) -> BGEM3FlagModel:
    use_fp16 = torch.cuda.is_available()

    if verbose:
        print("Loading BGE-M3 from local path...")
        print(f"Model path: {model_path}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"use_fp16: {use_fp16}")

    model = BGEM3FlagModel(
        str(model_path),
        use_fp16=False,
    )

    return model


def build_query_text(query: str, expanded_terms: Optional[List[str]] = None) -> str:
    if expanded_terms:
        terms_text = "、".join(expanded_terms)
        return f"为这个问题检索相关辞典条目：{query}\n重点参考标准术语：{terms_text}"

    return f"为这个问题检索相关辞典条目：{query}"


def encode_query(model: BGEM3FlagModel, query: str, expanded_terms: Optional[List[str]] = None) -> List[float]:
    query_text = build_query_text(query, expanded_terms)

    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            try:
                outputs = model.encode(
                    [query_text],
                    batch_size=1,
                    max_length=8192,
                    show_progress_bar=False,
                )
            except TypeError:
                outputs = model.encode(
                    [query_text],
                    batch_size=1,
                    max_length=8192,
                )

    return outputs["dense_vecs"][0].tolist()


def vector_search(
    model: BGEM3FlagModel,
    client: MilvusClient,
    query: str,
    expanded_terms: List[str],
    collection_name: str = DEFAULT_COLLECTION_NAME,
    output_fields: Optional[List[str]] = None,
    top_k: int = DEFAULT_VECTOR_TOP_K,
) -> List[Dict[str, Any]]:
    if output_fields is None:
        output_fields = OUTPUT_FIELDS

    query_vector = encode_query(model, query, expanded_terms)

    results = client.search(
        collection_name=collection_name,
        data=[query_vector],
        anns_field="vector",
        limit=top_k,
        output_fields=output_fields,
    )

    vector_hits: List[Dict[str, Any]] = []

    for rank, hit in enumerate(results[0], start=1):
        entity = hit.get("entity", {})
        score = hit.get("distance", 0.0)

        # rank score 用于融合排序
        vector_rank_score = 30.0 / rank

        vector_hits.append(
            {
                "id": hit.get("id", entity.get("chunk_id", "")),
                "entity": entity,
                "exact_score": 0.0,
                "bm25_score": None,
                "bm25_rank_score": 0.0,
                "vector_score": float(score) if score is not None else 0.0,
                "vector_rank_score": vector_rank_score,
                "final_score": vector_rank_score,
                "match_source": "vector",
                "match_type": "vector_semantic",
            }
        )

    return vector_hits


# =========================
# Fusion
# =========================

def dedupe_key(hit: Dict[str, Any]) -> str:
    entity = hit.get("entity", {})

    # 按 entry_id 去重，不按 chunk_id 去重
    entry_id = entity.get("entry_id")
    if entry_id:
        return f"entry::{entry_id}"

    entry = entity.get("entry", "")
    source = entity.get("source", "")
    return f"entry_text::{entry}::{source}"


def merge_source(old_source: str, new_source: str) -> str:
    old_parts = old_source.split("+") if old_source else []
    new_parts = new_source.split("+") if new_source else []

    merged: List[str] = []

    for part in old_parts + new_parts:
        if part and part not in merged:
            merged.append(part)

    return "+".join(merged)


def merge_type(old_type: str, new_type: str) -> str:
    old_parts = old_type.split("+") if old_type else []
    new_parts = new_type.split("+") if new_type else []

    merged: List[str] = []

    for part in old_parts + new_parts:
        if part and part not in merged:
            merged.append(part)

    return "+".join(merged)


def recompute_final_score(hit: Dict[str, Any]) -> float:
    """
    融合分数设计：
    - exact_score 最高，保证条目精确命中靠前
    - bm25_rank_score 次之，强化字面专名匹配
    - vector_rank_score 补充语义相关性
    """
    exact_score = hit.get("exact_score", 0) or 0
    bm25_rank_score = hit.get("bm25_rank_score", 0) or 0
    vector_rank_score = hit.get("vector_rank_score", 0) or 0

    # 多路命中奖励
    source = hit.get("match_source", "")
    route_count = len([x for x in source.split("+") if x])
    multi_route_bonus = max(0, route_count - 1) * 5

    return float(exact_score + bm25_rank_score + vector_rank_score + multi_route_bonus)


def fuse_results(
    exact_hits: List[Dict[str, Any]],
    bm25_hits: List[Dict[str, Any]],
    vector_hits: List[Dict[str, Any]],
    final_top_k: int = DEFAULT_FINAL_TOP_K,
) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}

    for hit in exact_hits + bm25_hits + vector_hits:
        key = dedupe_key(hit)

        if key not in merged:
            hit["final_score"] = recompute_final_score(hit)
            merged[key] = hit
            continue

        old = merged[key]

        old["exact_score"] = max(old.get("exact_score", 0) or 0, hit.get("exact_score", 0) or 0)

        if hit.get("bm25_score") is not None:
            old["bm25_score"] = hit.get("bm25_score")
            old["bm25_rank_score"] = max(
                old.get("bm25_rank_score", 0) or 0,
                hit.get("bm25_rank_score", 0) or 0,
            )

        if hit.get("vector_score") is not None:
            old["vector_score"] = hit.get("vector_score")
            old["vector_rank_score"] = max(
                old.get("vector_rank_score", 0) or 0,
                hit.get("vector_rank_score", 0) or 0,
            )

        old["match_source"] = merge_source(
            old.get("match_source", ""),
            hit.get("match_source", ""),
        )

        old["match_type"] = merge_type(
            old.get("match_type", ""),
            hit.get("match_type", ""),
        )

        old["final_score"] = recompute_final_score(old)

    fused = list(merged.values())

    fused.sort(
        key=lambda x: (
            x.get("final_score", 0),
            x.get("exact_score", 0) or 0,
            x.get("bm25_rank_score", 0) or 0,
            x.get("vector_rank_score", 0) or 0,
        ),
        reverse=True,
    )

    return fused[:final_top_k]


def confidence_label(hit: Dict[str, Any]) -> str:
    source = hit.get("match_source", "")

    if "alias_exact" in source and "bm25" in source and "vector" in source:
        return "别名映射、BM25、向量三路命中，高置信度"

    if "exact" in source and "bm25" in source and "vector" in source:
        return "条目精确、BM25、向量三路命中，高置信度"

    if "alias_exact" in source:
        return "别名映射命中，优先参考"

    if "exact" in source:
        return "条目精确命中，优先参考"

    if "bm25" in source and "vector" in source:
        return "BM25 与向量共同命中，较高置信度"

    if "bm25" in source:
        return "BM25 字面命中，适合专名/术语参考"

    if "vector" in source:
        return "向量语义相关"

    return "相关结果"


def add_confidence_labels(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for hit in results:
        hit["confidence"] = confidence_label(hit)
    return results


# =========================
# Functional search API
# =========================

class DictFusionSearcher:
    """加载一次资源后，可重复执行 search(query) 的检索器。"""

    def __init__(
        self,
        base_dir: str | Path = DEFAULT_BASE_DIR,
        milvus_uri: str | Path = DEFAULT_MILVUS_URI,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        chunk_path: str | Path = DEFAULT_CHUNK_PATH,
        alias_path: str | Path = DEFAULT_ALIAS_PATH,
        bm25_top_k: int = DEFAULT_BM25_TOP_K,
        vector_top_k: int = DEFAULT_VECTOR_TOP_K,
        final_top_k: int = DEFAULT_FINAL_TOP_K,
        max_exact_hits: int = DEFAULT_MAX_EXACT_HITS,
        verbose: bool = True,
    ):
        self.base_dir = Path(base_dir)
        self.milvus_uri = path_from_base(self.base_dir, milvus_uri)
        self.collection_name = collection_name
        self.model_path = path_from_base(self.base_dir, model_path)
        self.chunk_path = path_from_base(self.base_dir, chunk_path)
        self.alias_path = path_from_base(self.base_dir, alias_path)

        self.bm25_top_k = bm25_top_k
        self.vector_top_k = vector_top_k
        self.final_top_k = final_top_k
        self.max_exact_hits = max_exact_hits
        self.verbose = verbose

        self._check_required_paths()
        self.client = self._load_milvus_client()
        self.alias_map = load_alias_map(self.alias_path, verbose=self.verbose)
        self.dict_rows = load_dict_rows(self.chunk_path, verbose=self.verbose)

        if self.verbose:
            print("Building BM25 index...")
        self.bm25_index = BM25Index(self.dict_rows, verbose=self.verbose)
        self.model = load_model(self.model_path, verbose=self.verbose)

    def _check_required_paths(self) -> None:
        if not self.chunk_path.exists():
            raise FileNotFoundError(
                f"Cannot find {self.chunk_path}. Please put dict_chunks.jsonl under {self.base_dir}."
            )

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Cannot find local model: {self.model_path}. Please put models/bge-m3 under {self.base_dir}."
            )

        if not self.milvus_uri.exists():
            raise FileNotFoundError(
                f"Cannot find Milvus database: {self.milvus_uri}. Please put qing_dict_milvus.db under {self.base_dir}."
            )

    def _load_milvus_client(self) -> MilvusClient:
        client = MilvusClient(str(self.milvus_uri))
        collections = client.list_collections()
        if self.collection_name not in collections:
            raise RuntimeError(
                f"Collection {self.collection_name} not found. Existing collections: {collections}"
            )
        # milvus-lite 3.0 需要显式 load_collection
        client.load_collection(self.collection_name)
        return client

    def search(self, query: str) -> Dict[str, Any]:
        """
        传入 query，返回 alias_info 和融合检索 results。
        返回结果是可 JSON 序列化的 dict。
        """
        query = safe_str(query).strip()
        if not query:
            return {
                "query": query,
                "alias_info": {
                    "original_query": query,
                    "expanded_terms": [],
                    "matched_aliases": [],
                },
                "results": [],
            }

        alias_info = expand_query_by_alias(query, self.alias_map)
        expanded_terms = alias_info["expanded_terms"]

        exact_hits = exact_match_search(
            query=query,
            expanded_terms=expanded_terms,
            dict_rows=self.dict_rows,
            max_hits=self.max_exact_hits,
        )

        bm25_hits = bm25_search(
            bm25_index=self.bm25_index,
            query=query,
            expanded_terms=expanded_terms,
            top_k=self.bm25_top_k,
        )

        vector_hits = vector_search(
            model=self.model,
            client=self.client,
            query=query,
            expanded_terms=expanded_terms,
            collection_name=self.collection_name,
            output_fields=OUTPUT_FIELDS,
            top_k=self.vector_top_k,
        )

        fused_results = fuse_results(
            exact_hits=exact_hits,
            bm25_hits=bm25_hits,
            vector_hits=vector_hits,
            final_top_k=self.final_top_k,
        )

        return {
            "query": query,
            "alias_info": alias_info,
            "results": add_confidence_labels(fused_results),
        }


_DEFAULT_SEARCHER: Optional[DictFusionSearcher] = None


def get_default_searcher(**kwargs: Any) -> DictFusionSearcher:
    """获取默认全局检索器；首次调用时加载资源，后续复用。"""
    global _DEFAULT_SEARCHER
    if _DEFAULT_SEARCHER is None:
        _DEFAULT_SEARCHER = DictFusionSearcher(**kwargs)
    return _DEFAULT_SEARCHER


def search_dict(query: str, searcher: Optional[DictFusionSearcher] = None, **init_kwargs: Any) -> Dict[str, Any]:
    """
    函数式调用入口：传入 query，返回检索结果。

    参数：
    - query: 查询文本
    - searcher: 可选，传入已经初始化好的 DictFusionSearcher 以复用模型和索引
    - init_kwargs: 可选，首次初始化默认检索器时传给 DictFusionSearcher 的参数
    """
    if searcher is None:
        searcher = get_default_searcher(**init_kwargs)
    return searcher.search(query)
