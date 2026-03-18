"""
基于 Milvus Lite 的 RAG 知识库实现

使用 Milvus Lite 作为本地向量数据库，提供更好的检索性能和扩展性。
支持稠密向量（Dense）+ 稀疏向量（Sparse/BM25）混合检索。
"""

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

# BM25 分词器
try:
    import jieba
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False

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


# ==================== BM25 稀疏向量 ====================

class BM25Model:
    """BM25 模型，用于生成稀疏向量"""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Args:
            k1: 控制词频饱和度的参数 (默认1.5)
            b: 控制文档长度归一化的参数 (默认0.75)
        """
        self.k1 = k1
        self.b = b
        self.corpus: List[List[str]] = []
        self.doc_freqs: Dict[str, int] = {}  # 文档频率
        self.idf: Dict[str, float] = {}  # 逆文档频率
        self.doc_len: List[int] = []  # 文档长度
        self.avgdl: float = 0.0  # 平均文档长度
        self.fitted = False

    def tokenize(self, text: str) -> List[str]:
        """中文分词"""
        if JIEBA_AVAILABLE:
            return list(jieba.cut(text))
        else:
            # 简单按字符分词
            return list(text)

    def fit(self, corpus: List[str]) -> "BM25Model":
        """训练BM25模型"""
        self.corpus = [self.tokenize(doc) for doc in corpus]
        self.doc_len = [len(doc) for doc in self.corpus]
        self.avgdl = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 0

        # 计算文档频率
        self.doc_freqs = Counter()
        for doc in self.corpus:
            unique_words = set(doc)
            self.doc_freqs.update(unique_words)

        # 计算IDF
        n_docs = len(self.corpus)
        for word, freq in self.doc_freqs.items():
            self.idf[word] = np.log((n_docs - freq + 0.5) / (freq + 0.5) + 1)

        self.fitted = True
        return self

    def encode(self, text: str) -> Dict[int, float]:
        """
        将文本编码为稀疏向量格式 {token_id: score}

        使用与Milvus兼容的格式: {index: weight}
        """
        if not self.fitted:
            raise RuntimeError("BM25 model must be fitted before encoding")

        tokens = self.tokenize(text)
        doc_len = len(tokens)

        # 计算词频
        term_freqs = Counter(tokens)

        # 计算BM25分数
        sparse_vector = {}
        for term, freq in term_freqs.items():
            if term not in self.idf:
                continue

            # BM25公式
            numerator = freq * (self.k1 + 1)
            denominator = freq + self.k1 * (1 - self.b + self.b * (doc_len / self.avgdl))
            score = self.idf[term] * (numerator / denominator)

            # 使用词的hash作为索引（实际应用中应该用词表）
            # 这里用简单的hash取模方式，实际应该用词表映射
            token_id = hash(term) % 1000000  # 限制ID范围
            sparse_vector[token_id] = float(score)

        return sparse_vector

    def encode_corpus(self, corpus: List[str]) -> List[Dict[int, float]]:
        """批量编码语料"""
        return [self.encode(doc) for doc in corpus]


def create_collection(collection_name: str, embedding_dim: int, overwrite: bool = False, enable_sparse: bool = True) -> Collection:
    """创建 Milvus 集合（支持稠密+稀疏向量）"""
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

    # 定义 Schema - 稠密向量 + 稀疏向量
    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100, auto_id=False),
        FieldSchema(name="source_file", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="clause_id", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim),
    ]

    # 添加稀疏向量字段
    if enable_sparse:
        fields.append(
            FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR)
        )

    schema = CollectionSchema(fields, f"Insurance regulations knowledge base: {collection_name} (Hybrid)")
    collection = Collection(collection_name, schema)

    # 创建稠密向量索引
    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "COSINE",
        "params": {"nlist": 128},
    }
    collection.create_index(field_name="vector", index_params=index_params)

    # 创建稀疏向量索引
    if enable_sparse:
        sparse_index_params = {
            "index_type": "SPARSE_INVERTED_INDEX",
            "metric_type": "IP",
            "params": {"drop_ratio_build": 0.1},
        }
        collection.create_index(field_name="sparse_vector", index_params=sparse_index_params)

    return collection


def build_knowledge_base(
    doc_paths: List[str],
    output_dir: str,
    client: BailianClient,
    collection_name: str = "insurance_knowledge",
    pdf_mode: str = "vl",
    max_pages: Optional[int] = None,
    overwrite: bool = True,
    enable_sparse: bool = True,
) -> MilvusKnowledgeBase:
    """构建 Milvus 知识库（支持稠密+稀疏向量）"""
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

    print(f"[*] 生成稠密向量 (Dense)...")
    # 生成稠密向量
    embeddings = client.embed_texts([chunk.clause_text for chunk in all_chunks])

    # 生成稀疏向量 (BM25)
    sparse_vectors = None
    if enable_sparse:
        print(f"[*] 生成稀疏向量 (Sparse/BM25)...")
        bm25 = BM25Model()
        corpus = [chunk.clause_text for chunk in all_chunks]
        bm25.fit(corpus)
        sparse_vectors = bm25.encode_corpus(corpus)
        print(f"[*] BM25模型训练完成")

    # 创建集合
    embedding_dim = get_embedding_dim(client.settings.embedding_model)
    collection = create_collection(collection_name, embedding_dim, overwrite=overwrite, enable_sparse=enable_sparse)

    # 准备数据
    ids = [str(i) for i in range(len(all_chunks))]
    source_files = [chunk.source_file for chunk in all_chunks]
    clause_ids = [chunk.clause_id for chunk in all_chunks]
    vectors = embeddings.tolist()

    # 插入数据
    print(f"[*] 插入数据到 Milvus...")
    if enable_sparse and sparse_vectors:
        data = [ids, source_files, clause_ids, vectors, sparse_vectors]
    else:
        data = [ids, source_files, clause_ids, vectors]

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
        "enable_sparse": enable_sparse,
        "hybrid_search": enable_sparse,
    }
    with (output / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[✓] 知识库构建完成: {len(all_chunks)} chunks")

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


def _has_sparse_vector(collection: Collection) -> bool:
    """检测集合是否有稀疏向量字段"""
    schema = collection.schema
    for field in schema.fields:
        if field.name == "sparse_vector":
            return True
    return False


def _encode_query_sparse(query: str, kb: MilvusKnowledgeBase) -> Dict[int, float]:
    """为查询生成稀疏向量（使用BM25模型）"""
    # 注意：这里简化处理，实际应该保存BM25模型
    # 为了简化，我们使用关键词匹配的方式
    if not JIEBA_AVAILABLE:
        return {}

    bm25 = BM25Model()
    # 使用知识库中的文本作为corpus训练
    corpus = [chunk.clause_text for chunk in kb.chunks]
    bm25.fit(corpus)

    return bm25.encode(query)


def retrieve_dense(
    query: str,
    kb: MilvusKnowledgeBase,
    client: BailianClient,
    top_k: int = 6,
) -> List[dict]:
    """稠密向量检索（Dense Retrieval）"""
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
                "retrieval_type": "dense",
            }
        )

    return result_list


def retrieve_sparse(
    query: str,
    kb: MilvusKnowledgeBase,
    client: BailianClient,
    top_k: int = 6,
) -> List[dict]:
    """稀疏向量检索（Sparse Retrieval / BM25）"""
    if not _has_sparse_vector(kb.collection):
        return []

    # 生成稀疏向量
    sparse_vec = _encode_query_sparse(query, kb)
    if not sparse_vec:
        return []

    # Milvus 搜索参数
    search_params = {"metric_type": "IP", "params": {"drop_ratio_build": 0.1}}

    # 执行搜索
    try:
        results = kb.collection.search(
            data=[sparse_vec],
            anns_field="sparse_vector",
            param=search_params,
            limit=top_k,
            output_fields=["source_file", "clause_id"],
        )
    except Exception as e:
        # 如果稀疏搜索失败，返回空结果
        return []

    # 格式化结果
    result_list: List[dict] = []
    for hit in results[0]:
        result_list.append(
            {
                "score": float(hit.score),
                "source_file": hit.entity.get("source_file"),
                "clause_id": hit.entity.get("clause_id"),
                "clause_text": kb.chunks[int(hit.id)].clause_text,
                "retrieval_type": "sparse",
            }
        )

    return result_list


def retrieve_relevant_rules(
    query: str,
    kb: MilvusKnowledgeBase,
    client: BailianClient,
    top_k: int = 6,
) -> List[dict]:
    """使用 Milvus 向量检索（兼容旧接口，使用稠密检索）"""
    return retrieve_dense(query, kb, client, top_k)


def hybrid_retrieve(
    queries: List[str],
    kb: MilvusKnowledgeBase,
    client: BailianClient,
    top_k: int = 6,
    dense_weight: float = 0.7,
    sparse_weight: float = 0.3,
) -> List[dict]:
    """
    混合检索（6路召回）= 3路稠密 + 3路稀疏

    Args:
        queries: 查询列表（通常是问题重写后的多个查询）
        kb: 知识库
        client: Bailian客户端
        top_k: 每路返回的数量
        dense_weight: 稠密检索权重
        sparse_weight: 稀疏检索权重

    Returns:
        融合后的结果列表
    """
    if not _has_sparse_vector(kb.collection):
        # 如果不支持稀疏向量，只使用稠密检索
        all_results = []
        for query in queries:
            results = retrieve_dense(query, kb, client, top_k)
            all_results.extend(results)
        return _deduplicate_and_rerank(all_results, top_k * 2)

    # 6路召回
    all_results: List[dict] = []

    print(f"[*] 开始6路召回 (稠密x3 + 稀疏x3)...")
    for i, query in enumerate(queries[:3], 1):
        print(f"  [{i}/3] 稠密检索: {query[:30]}...")
        dense_results = retrieve_dense(query, kb, client, top_k)
        all_results.extend(dense_results)

    for i, query in enumerate(queries[:3], 1):
        print(f"  [{i+3}/6] 稀疏检索: {query[:30]}...")
        sparse_results = retrieve_sparse(query, kb, client, top_k)
        all_results.extend(sparse_results)

    print(f"[*] 召回完成，共 {len(all_results)} 条结果")

    # 融合结果
    return _fuse_results(all_results, dense_weight, sparse_weight, top_k * 2)


def _fuse_results(
    results: List[dict],
    dense_weight: float,
    sparse_weight: float,
    top_k: int,
) -> List[dict]:
    """融合稠密和稀疏检索结果（RRF算法）"""
    # 按类型分组
    dense_results = [r for r in results if r.get("retrieval_type") == "dense"]
    sparse_results = [r for r in results if r.get("retrieval_type") == "sparse"]

    # RRF (Reciprocal Rank Fusion)
    k = 60  # RRF常数
    fused_scores = {}

    for i, result in enumerate(dense_results):
        key = (result["clause_id"], result["source_file"])
        rrf_score = dense_weight * (1 / (k + i + 1))
        fused_scores[key] = fused_scores.get(key, 0) + rrf_score

    for i, result in enumerate(sparse_results):
        key = (result["clause_id"], result["source_file"])
        rrf_score = sparse_weight * (1 / (k + i + 1))
        fused_scores[key] = fused_scores.get(key, 0) + rrf_score

    # 按融合分数排序
    sorted_keys = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)[:top_k]

    # 构建最终结果
    final_results = []
    seen_chunks = {}
    for key in sorted_keys:
        clause_id, source_file = key
        # 在原结果中找到对应的chunk
        for result in results:
            if result["clause_id"] == clause_id and result["source_file"] == source_file:
                if key not in seen_chunks:
                    final_results.append({
                        "score": fused_scores[key],
                        "source_file": result["source_file"],
                        "clause_id": result["clause_id"],
                        "clause_text": result["clause_text"],
                        "retrieval_type": "hybrid",
                    })
                    seen_chunks[key] = True
                break

    return final_results


def _deduplicate_and_rerank(results: List[dict], top_k: int) -> List[dict]:
    """去重并重新排序"""
    # 去重
    seen = set()
    unique_results = []
    for result in results:
        key = (result["clause_id"], result["source_file"])
        if key not in seen:
            seen.add(key)
            unique_results.append(result)

    # 按分数排序
    unique_results.sort(key=lambda x: x["score"], reverse=True)
    return unique_results[:top_k]
