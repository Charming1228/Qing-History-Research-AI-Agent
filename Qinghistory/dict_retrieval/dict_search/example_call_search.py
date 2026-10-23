"""
调用示例。

运行方式：
    python example_call_search.py "军机处"

目录结构示例：
    ./example_call_search.py
    ./search_dict_fusion_core.py
    ./qing_dict_milvus.db
    ./dict_chunks.jsonl
    ./alias_map.json
    ./models/bge-m3/
"""

import json
import sys

from search_dict_fusion_core import DictFusionSearcher, search_dict


def call_once(query: str) -> dict:
    """最简单的函数式调用：直接给 query，返回结果。"""
    return search_dict(query)


def call_reuse_searcher(query: str) -> dict:
    """
    推荐的重复调用方式：先初始化检索器，再多次调用 search(query)。
    这样不会每次都重新加载模型、Milvus、BM25 索引。
    """
    searcher = DictFusionSearcher()
    return searcher.search(query)


def main() -> None:
    query = " ".join(sys.argv[1:]).strip() or "军机处"

    # 单次调用用 call_once 即可。
    result = call_once(query)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
