"""
基于 Milvus Lite 的 RAG 知识库实现

使用 Milvus Lite 作为本地向量数据库，提供更好的检索性能和扩展性。
"""

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

try:
    from pymilvus import (
        MilvusException,
        connections,
        utility,
        Collection,
        CollectionSchema,
        FieldSchema,
        DataType,
    )
    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False

from .bailian_client import BailianClient
from .extractors import extract_text_from_file

ARTICLE_PATTERN = re.compile(r"^(第[一二三四五六七八九十百零〇0-9]+条)")


@dataclass
class RuleChunk:
    source_file: str
    clause_id: str
    clause_text: str


@dataclass
class MilvusKnowledgeBase:
    collection: Collection
    chunks: List[RuleChunk]


def split_into_rule_chunks(
    source_file: str,
    text: str,
    max_chunk_chars: int = 700,
) -> List[RuleChunk]:
    """与原版相同的分块逻辑"""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    chunks: List[RuleChunk] = []

    current_clause = "未标注条文"
    buffer: List[str] = []

    def flush() -> None:
        nonlocal buffer
        if not buffer:
            return
        merged = "\n".join(buffer).strip()
        if len(merged) < 15:
            buffer = []
            return

        if len(merged) <= max_chunk_chars:
            chunks.append(
                RuleChunk(
                    source_file=source_file,
                    clause_id=current_clause,
                    clause_text=merged,
                )
            )
        else:
            for i in range(0, len(merged), max_chunk_chars):
                part = merged[i : i + max_chunk_chars]
                chunks.append(
                    RuleChunk(
                        source_file=source_file,
                        clause_id=f"{current_clause}-片段{i // max_chunk_chars + 1}",
                        clause_text=part,
                    )
                )
        buffer = []

    for line in lines:
        match = ARTICLE_PATTERN.match(line)
        if match:
            flush()
            current_clause = match.group(1)
            buffer.append(line)
        else:
            buffer.append(line)

    flush()

    if not chunks:
        text_flat = "\n".join(lines)
        for i in range(0, len(text_flat), max_chunk_chars):
            part = text_flat[i : i + max_chunk_chars]
            if part.strip():
                chunks.append(
                    RuleChunk(
                        source_file=source_file,
                        clause_id=f"段落{i // max_chunk_chars + 1}",
                        clause_text=part,
                    )
                )
    return chunks


def get_embedding_dim(model: str = "text-embedding-v3") -> int:
    """获取不同模型的向量维度"""
    dim_map = {
        "text-embedding-v3": 1024,
        "text-embedding-v2": 1536,
        "text-embedding-v1": 768,
    }
    return dim_map.get(model, 1024)


def create_collection(collection_name: str, embedding_dim: int, overwrite: bool = False) -> Collection:
    """创建 Milvus 集合"""
    if not MILVUS_AVAILABLE:
        raise RuntimeError("Milvus not available. Install with: pip install pymilvus")

    # 连接到 Milvus Lite
    connections.connect("default", uri="milvus_demo.db")

    # 如果已存在且需要覆盖
    if utility.has_collection(collection_name):
        if overwrite:
            utility.drop_collection(collection_name)
        else:
            return Collection(collection_name)

    # 定义 Schema
    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100, auto_id=False),
        FieldSchema(name="source_file", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="clause_id", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim),
    ]

    schema = CollectionSchema(fields, f"Insurance regulations knowledge base: {collection_name}")
    collection = Collection(collection_name, schema)

    # 创建索引
    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "COSINE",
        "params": {"nlist": 128},
    }
    collection.create_index(field_name="vector", index_params=index_params)

    return collection


def build_knowledge_base(
    doc_paths: List[str],
    output_dir: str,
    client: BailianClient,
    collection_name: str = "insurance_knowledge",
    pdf_mode: str = "vl",
    max_pages: Optional[int] = None,
    overwrite: bool = True,
) -> MilvusKnowledgeBase:
    """构建 Milvus 知识库"""
    if not MILVUS_AVAILABLE:
        raise RuntimeError("Milvus not available. Install with: pip install pymilvus")

    # 收集所有 chunks
    all_chunks: List[RuleChunk] = []

    for file_path in doc_paths:
        path = Path(file_path)
        text = extract_text_from_file(
            file_path=str(path),
            client=client,
            pdf_mode=pdf_mode,
            max_pages=max_pages,
        )
        chunks = split_into_rule_chunks(source_file=path.name, text=text)
        all_chunks.extend(chunks)

    if not all_chunks:
        raise RuntimeError("No knowledge chunks were generated from source documents.")

    # 生成向量
    embeddings = client.embed_texts([chunk.clause_text for chunk in all_chunks])

    # 创建集合
    embedding_dim = get_embedding_dim(client.settings.embedding_model)
    collection = create_collection(collection_name, embedding_dim, overwrite=overwrite)

    # 准备数据
    data = [
        [str(i) for i in range(len(all_chunks))],  # IDs
        [chunk.source_file for chunk in all_chunks],  # source_file
        [chunk.clause_id for chunk in all_chunks],  # clause_id
        embeddings.tolist(),  # vectors
    ]

    # 插入数据
    collection.insert(data)
    collection.flush()

    # 加载集合到内存
    collection.load()

    # 保存元数据
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    with (output / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")

    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "num_chunks": len(all_chunks),
        "embedding_model": client.settings.embedding_model,
        "pdf_mode": pdf_mode,
        "vector_db": "milvus",
        "collection_name": collection_name,
    }
    with (output / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return MilvusKnowledgeBase(collection=collection, chunks=all_chunks)


def load_knowledge_base(
    collection_name: str = "insurance_knowledge",
    meta_dir: str = "kb",
) -> MilvusKnowledgeBase:
    """加载已有知识库"""
    if not MILVUS_AVAILABLE:
        raise RuntimeError("Milvus not available. Install with: pip install pymilvus")

    # 连接到 Milvus
    connections.connect("default", uri="milvus_demo.db")

    # 加载集合
    collection = Collection(collection_name)
    collection.load()

    # 从文件加载 chunks
    chunks_path = Path(meta_dir) / "chunks.jsonl"
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")

    chunks: List[RuleChunk] = []
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            chunks.append(RuleChunk(**row))

    return MilvusKnowledgeBase(collection=collection, chunks=chunks)


def retrieve_relevant_rules(
    query: str,
    kb: MilvusKnowledgeBase,
    client: BailianClient,
    top_k: int = 6,
) -> List[dict]:
    """使用 Milvus 向量检索"""
    # 查询向量化
    query_vec = client.embed_texts([query])[0]

    # Milvus 搜索参数
    search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}

    # 执行搜索
    results = kb.collection.search(
        data=[query_vec.tolist()],
        anns_field="vector",
        param=search_params,
        limit=top_k,
        output_fields=["source_file", "clause_id"],
    )

    # 格式化结果
    result_list: List[dict] = []
    for hit in results[0]:
        result_list.append(
            {
                "score": float(hit.score),
                "source_file": hit.entity.get("source_file"),
                "clause_id": hit.entity.get("clause_id"),
                "clause_text": kb.chunks[int(hit.id)].clause_text,
            }
        )

    return result_list
