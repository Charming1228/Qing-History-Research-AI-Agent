r"""
重建词典 Milvus 数据库（milvus-lite 3.0 目录格式）

用法：
    cd dict_retrieval/dict_search
    python rebuild_milvus.py

功能：
    1. 备份旧的 qing_dict_milvus.db (2.x SQLite 文件) -> qing_dict_milvus.db.old
    2. 用 BGE-M3 对 dict_chunks.jsonl 中全部 chunk 编码
    3. 创建 milvus-lite 3.0 集合并插入数据
"""

import json
import os
import shutil
import sys
import time
from pathlib import Path

import torch
from FlagEmbedding import BGEM3FlagModel
from pymilvus import DataType, MilvusClient

# =========================
# 配置
# =========================

BASE_DIR = Path(__file__).parent
MILVUS_URI = str(BASE_DIR / "qing_dict_milvus.db")  # 必须以 .db 结尾，milvus-lite 3.0 创建目录
COLLECTION_NAME = "qing_dict"
MODEL_PATH = str(BASE_DIR / "models" / "bge-m3")
CHUNK_PATH = str(BASE_DIR / "dict_chunks.jsonl")
VECTORS_CACHE = str(BASE_DIR / "vectors_cache.npy")  # 编码缓存，避免重跑时重新编码

# 向量维度 (BGE-M3 dense: 1024)
VECTOR_DIM = 1024
BATCH_SIZE = 200  # 每批插入的条数
ENCODE_BATCH_SIZE = 32  # 编码时的 batch size

# 需要存入 Milvus 的标量字段
SCALAR_FIELDS = [
    ("entry_id", DataType.VARCHAR, 64),
    ("chunk_id", DataType.VARCHAR, 64),
    ("chunk_index", DataType.INT64),
    ("row_id", DataType.VARCHAR, 64),
    ("entry", DataType.VARCHAR, 512),
    ("definition", DataType.VARCHAR, 8192),
    ("source", DataType.VARCHAR, 512),
    ("type", DataType.VARCHAR, 128),
    ("period", DataType.VARCHAR, 128),
    ("author", DataType.VARCHAR, 256),
    ("source_file", DataType.VARCHAR, 256),
    ("data_type", DataType.VARCHAR, 64),
    ("embedding_text", DataType.VARCHAR, 8192),
]


def load_chunks(path: str) -> list:
    """加载 dict_chunks.jsonl"""
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if row.get("entry") and row.get("definition"):
                    chunks.append(row)
            except json.JSONDecodeError:
                continue
    print(f"[OK] 加载了 {len(chunks)} 条词典 chunks")
    return chunks


def backup_old_db():
    """备份旧的 milvus-lite 2.x SQLite 文件"""
    db_path = Path(MILVUS_URI)
    if not db_path.exists():
        print("[INFO]  旧数据库文件不存在，无需备份")
        return

    if db_path.is_dir():
        print(f"[INFO]  {MILVUS_URI} 已经是目录，跳过备份")
        return

    # 是旧版 SQLite 文件，重命名为 .old
    backup_path = Path(str(db_path) + ".old")
    if backup_path.exists():
        if backup_path.is_dir():
            shutil.rmtree(backup_path)
        else:
            backup_path.unlink()

    db_path.rename(backup_path)
    print(f"[OK] 旧数据库已备份为: {backup_path}")


def load_model(model_path: str) -> BGEM3FlagModel:
    """加载 BGE-M3 模型"""
    use_fp16 = torch.cuda.is_available()
    print(f"加载 BGE-M3 模型...")
    print(f"  模型路径: {model_path}")
    print(f"  CUDA 可用: {torch.cuda.is_available()}")
    print(f"  use_fp16: {use_fp16}")

    model = BGEM3FlagModel(model_path, use_fp16=use_fp16)
    print("[OK] 模型加载完毕")
    return model


def encode_chunks(model: BGEM3FlagModel, chunks: list, cache_path: str = None) -> list:
    """用 BGE-M3 对全部 chunk 的 embedding_text 进行编码，返回 dense vectors"""
    import numpy as np

    # 尝试从缓存加载
    if cache_path and os.path.exists(cache_path):
        print(f"[INFO] 从缓存加载向量: {cache_path}")
        cached = np.load(cache_path)
        if len(cached) == len(chunks):
            print(f"[OK] 缓存向量加载成功，共 {len(cached)} 条")
            return cached.tolist()
        else:
            print(f"[WARN] 缓存的向量数量 ({len(cached)}) 与当前数据 ({len(chunks)}) 不匹配，重新编码")

    texts = [c.get("embedding_text", c.get("definition", "")) for c in chunks]
    vectors = []

    total = len(texts)
    print(f"开始编码 {total} 条文本...")

    for i in range(0, total, ENCODE_BATCH_SIZE):
        batch = texts[i : i + ENCODE_BATCH_SIZE]
        # BGE-M3 encode returns dense_vecs and lexical_weights (sparse)
        output = model.encode(
            batch,
            batch_size=ENCODE_BATCH_SIZE,
            max_length=8192,
        )
        dense = output["dense_vecs"].tolist()
        vectors.extend(dense)

        if (i + ENCODE_BATCH_SIZE) % 500 == 0 or i + ENCODE_BATCH_SIZE >= total:
            pct = min(i + ENCODE_BATCH_SIZE, total) / total * 100
            print(f"  进度: {min(i+ENCODE_BATCH_SIZE, total)}/{total} ({pct:.1f}%)")

    print(f"[OK] 编码完成，共 {len(vectors)} 条向量")

    # 保存缓存
    if cache_path:
        np.save(cache_path, np.array(vectors, dtype=np.float32))
        print(f"[OK] 向量已缓存到: {cache_path}")

    return vectors


def create_milvus_collection():
    """创建 milvus-lite 3.0 数据库和集合"""
    # 如果旧目录存在，先删除
    db_path = Path(MILVUS_URI)
    if db_path.exists():
        if db_path.is_dir():
            shutil.rmtree(db_path)
        else:
            db_path.unlink()

    print(f"创建 Milvus 数据库: {MILVUS_URI}")
    client = MilvusClient(MILVUS_URI)

    # 创建 schema
    schema = client.create_schema(
        auto_id=False,
        enable_dynamic_field=False,
    )

    # 主键: id (chunk_id)
    schema.add_field(
        field_name="id",
        datatype=DataType.VARCHAR,
        is_primary=True,
        max_length=64,
    )

    # 向量字段
    schema.add_field(
        field_name="vector",
        datatype=DataType.FLOAT_VECTOR,
        dim=VECTOR_DIM,
    )

    # 标量字段
    for item in SCALAR_FIELDS:
        if len(item) == 3:
            field_name, datatype, max_len = item
        else:
            field_name, datatype = item
            max_len = None
        if datatype == DataType.VARCHAR and max_len is not None:
            schema.add_field(
                field_name=field_name,
                datatype=datatype,
                max_length=max_len,
            )
        else:
            schema.add_field(
                field_name=field_name,
                datatype=datatype,
            )

    # 创建集合
    print(f"创建集合: {COLLECTION_NAME}")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
    )

    # 创建向量索引 (IP = inner product / cosine for normalized vectors)
    print("创建向量索引...")
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="FLAT",  # 精确搜索
        metric_type="IP",  # inner product (BGE 向量已归一化)
    )
    client.create_index(
        collection_name=COLLECTION_NAME,
        index_params=index_params,
    )

    print("[OK] Milvus 集合和索引创建完毕")
    return client


def insert_data(client: MilvusClient, chunks: list, vectors: list):
    """批量插入数据"""
    assert len(chunks) == len(vectors), f"数据长度不匹配: {len(chunks)} vs {len(vectors)}"

    total = len(chunks)
    print(f"开始插入 {total} 条数据...")

    for i in range(0, total, BATCH_SIZE):
        batch_chunks = chunks[i : i + BATCH_SIZE]
        batch_vectors = vectors[i : i + BATCH_SIZE]

        data = []
        for chunk, vec in zip(batch_chunks, batch_vectors):
            record = {
                "id": chunk.get("chunk_id", chunk.get("id", "")),
                "vector": vec,
                "entry_id": str(chunk.get("entry_id", "")),
                "chunk_id": str(chunk.get("chunk_id", chunk.get("id", ""))),
                "chunk_index": int(chunk.get("chunk_index", 0)),
                "row_id": str(chunk.get("row_id", "")),
                "entry": str(chunk.get("entry", "")),
                "definition": str(chunk.get("definition", "")),
                "source": str(chunk.get("source", "")),
                "type": str(chunk.get("type", "")),
                "period": str(chunk.get("period", "")),
                "author": str(chunk.get("author", "")),
                "source_file": str(chunk.get("source_file", "")),
                "data_type": str(chunk.get("data_type", "")),
                "embedding_text": str(chunk.get("embedding_text", "")),
            }
            data.append(record)

        try:
            client.insert(
                collection_name=COLLECTION_NAME,
                data=data,
            )
        except Exception as e:
            print(f"  [WARN] 批次 {i // BATCH_SIZE} 插入失败: {e}")
            # 尝试逐条插入
            for j, record in enumerate(data):
                try:
                    client.insert(
                        collection_name=COLLECTION_NAME,
                        data=[record],
                    )
                except Exception as e2:
                    print(f"    记录 {i + j} 插入失败: {e2}")
            continue

        if (i + BATCH_SIZE) % 2000 == 0 or i + BATCH_SIZE >= total:
            pct = min(i + BATCH_SIZE, total) / total * 100
            print(f"  进度: {min(i+BATCH_SIZE, total)}/{total} ({pct:.1f}%)")

    # flush 确保数据持久化 (milvus-lite 3.0 Windows 上 os.rename 可能失败, 数据实际已写入)
    try:
        client.flush(COLLECTION_NAME)
    except Exception as e:
        print(f"[WARN] flush 失败 (Windows 已知问题, 数据通常已持久化): {e}")
    print("[OK] 数据插入完毕")


def main():
    start_time = time.time()

    print("=" * 60)
    print("[BOOK] 词典 Milvus 数据库重建工具")
    print("=" * 60)

    # 1. 备份旧数据库
    print("\n[1/5] 备份旧数据库...")
    backup_old_db()

    # 2. 加载数据
    print("\n[2/5] 加载词典 chunks...")
    chunks = load_chunks(CHUNK_PATH)
    if not chunks:
        print("[ERROR] 没有数据，退出")
        sys.exit(1)

    # 3. 加载模型
    print("\n[3/5] 加载 BGE-M3 模型...")
    model = load_model(MODEL_PATH)

    # 4. 编码
    print("\n[4/5] 编码所有 chunks...")
    vectors = encode_chunks(model, chunks, cache_path=VECTORS_CACHE)

    # 5. 创建数据库并插入
    print("\n[5/5] 创建 Milvus 数据库并插入数据...")
    client = create_milvus_collection()
    insert_data(client, chunks, vectors)

    # 验证
    print("\n" + "=" * 60)
    stats = client.get_collection_stats(COLLECTION_NAME)
    print(f"集合统计: {stats}")
    print(f"集合列表: {client.list_collections()}")
    client.close()

    elapsed = time.time() - start_time
    print(f"\n[OK] 全部完成！耗时: {elapsed / 60:.1f} 分钟")
    print(f"数据库路径: {MILVUS_URI}")
    print(f"记录数: {len(chunks)}")
    print("\n可以运行 python server.py 启动服务了。")


if __name__ == "__main__":
    main()
