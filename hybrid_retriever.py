"""
混合检索实现 - BM25 + 稠密向量检索

结合稀疏检索（BM25）和稠密检索（向量）的优势，提高检索精度。
"""

import json
import re
from typing import Any, Dict, List, TYPE_CHECKING
from pathlib import Path

import numpy as np

try:
    from rank_bm25 import BM25Okapi
    import jieba
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

from .bailian_client import BailianClient
from .knowledge_base import RuleChunk, KnowledgeBase

if TYPE_CHECKING:
    from . import knowledge_base_milvus


class HybridRetriever:
    """混合检索器 - 结合 BM25 和向量检索"""

    def __init__(
        self,
        chunks: List[RuleChunk],
        dense_embeddings: np.ndarray,
        bm25_weight: float = 0.3,
        dense_weight: float = 0.7,
    ):
        """
        初始化混合检索器

        Args:
            chunks: 法规条文chunks
            dense_embeddings: 稠密向量嵌入
            bm25_weight: BM25检索权重（默认0.3）
            dense_weight: 稠密检索权重（默认0.7）
        """
        self.chunks = chunks
        self.embeddings = dense_embeddings
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight

        # 初始化 BM25 索引
        if BM25_AVAILABLE:
            self._init_bm25_index()
        else:
            self.bm25 = None
            print("⚠️  BM25 not available. Install with: pip install rank-bm25 jieba")

    def _init_bm25_index(self):
        """初始化 BM25 索引"""
        # 使用 jieba 分词
        tokenized_chunks = []
        for chunk in self.chunks:
            # 移除标点符号，分词
            text = re.sub(r'[^\w\s]', '', chunk.clause_text)
            tokens = list(jieba.cut(text))
            tokenized_chunks.append(tokens)

        # 构建 BM25 索引
        self.bm25 = BM25Okapi(tokenized_chunks)
        self.tokenized_chunks = tokenized_chunks

    def retrieve(
        self,
        query: str,
        client: BailianClient,
        top_k: int = 6,
        rerank_top_k: int = 20,
    ) -> List[Dict]:
        """
        混合检索 - 结合 BM25 和向量检索

        Args:
            query: 查询文本
            client: 百炼客户端
            top_k: 返回结果数量
            rerank_top_k: 重排序前保留的数量

        Returns:
            检索结果列表
        """
        # 1. 稠密向量检索
        dense_results = self._dense_retrieve(query, client, top_k=rerank_top_k)

        # 2. BM25 稀疏检索（如果可用）
        if self.bm25 is not None:
            sparse_results = self._sparse_retrieve(query, top_k=rerank_top_k)
            # 3. 加权融合
            fused_results = self._fuse_results(dense_results, sparse_results)
        else:
            fused_results = dense_results

        # 4. 返回 Top-K
        return fused_results[:top_k]

    def _dense_retrieve(
        self,
        query: str,
        client: BailianClient,
        top_k: int = 20,
    ) -> List[Dict]:
        """稠密向量检索"""
        # 查询向量化
        query_vec = client.embed_texts([query])[0]
        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-12)

        # 归一化嵌入
        emb_norm = self.embeddings / (np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-12)

        # 余弦相似度
        scores = emb_norm @ query_norm

        # Top-K
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            chunk = self.chunks[int(idx)]
            results.append({
                'score': float(scores[idx]),
                'source_file': chunk.source_file,
                'clause_id': chunk.clause_id,
                'clause_text': chunk.clause_text,
                'retrieval_method': 'dense',
            })

        return results

    def _sparse_retrieve(
        self,
        query: str,
        top_k: int = 20,
    ) -> List[Dict]:
        """BM25 稀疏检索"""
        if self.bm25 is None:
            return []

        # 查询分词
        query_clean = re.sub(r'[^\w\s]', '', query)
        tokenized_query = list(jieba.cut(query_clean))

        # BM25 评分
        scores = self.bm25.get_scores(tokenized_query)

        # Top-K
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if idx >= 0 and idx < len(self.chunks):
                chunk = self.chunks[idx]
                results.append({
                    'score': float(scores[idx]),
                    'source_file': chunk.source_file,
                    'clause_id': chunk.clause_id,
                    'clause_text': chunk.clause_text,
                    'retrieval_method': 'sparse',
                })

        return results

    def _fuse_results(
        self,
        dense_results: List[Dict],
        sparse_results: List[Dict],
    ) -> List[Dict]:
        """
        加权融合两种检索结果

        使用 RRF (Reciprocal Rank Fusion) 算法
        """
        # 归一化分数
        dense_scores = np.array([r['score'] for r in dense_results])
        if len(dense_scores) > 0:
            dense_scores = (dense_scores - dense_scores.min()) / (dense_scores.max() - dense_scores.min() + 1e-12)

        sparse_scores = np.array([r['score'] for r in sparse_results])
        if len(sparse_scores) > 0:
            sparse_scores = (sparse_scores - sparse_scores.min()) / (sparse_scores.max() - sparse_scores.min() + 1e-12)

        # 为每个chunk计算融合分数
        chunk_scores = {}

        # 稠密检索分数
        for i, result in enumerate(dense_results):
            key = (result['clause_id'], result['source_file'])
            chunk_scores[key] = chunk_scores.get(key, 0) + self.dense_weight * dense_scores[i]

        # 稀疏检索分数
        for i, result in enumerate(sparse_results):
            key = (result['clause_id'], result['source_file'])
            chunk_scores[key] = chunk_scores.get(key, 0) + self.bm25_weight * sparse_scores[i]

        # 构建结果
        fused_results = []
        seen_chunks = set()

        # 保留原始检索信息
        all_results = dense_results + sparse_results
        result_map = {(r['clause_id'], r['source_file']): r for r in all_results}

        # 按融合分数排序
        sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)

        for (clause_id, source_file), fused_score in sorted_chunks:
            if (clause_id, source_file) in seen_chunks:
                continue

            result = result_map.get((clause_id, source_file))
            if result:
                result = result.copy()
                result['score'] = fused_score
                result['retrieval_method'] = 'hybrid'
                fused_results.append(result)
                seen_chunks.add((clause_id, source_file))

        return fused_results


def create_hybrid_retriever(
    kb_dir: str = "kb",
    bm25_weight: float = 0.3,
    dense_weight: float = 0.7,
) -> HybridRetriever:
    """
    创建混合检索器

    Args:
        kb_dir: 知识库目录
        bm25_weight: BM25检索权重
        dense_weight: 稠密检索权重

    Returns:
        混合检索器实例
    """
    # 加载chunks和embeddings
    kb_path = Path(kb_dir)
    chunks_path = kb_path / "chunks.jsonl"
    emb_path = kb_path / "embeddings.npy"

    if not chunks_path.exists() or not emb_path.exists():
        raise FileNotFoundError(f"Knowledge base files not found in {kb_dir}")

    # 加载chunks
    chunks: List[RuleChunk] = []
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            chunks.append(RuleChunk(**row))

    # 加载embeddings
    embeddings = np.load(emb_path)

    return HybridRetriever(
        chunks=chunks,
        dense_embeddings=embeddings,
        bm25_weight=bm25_weight,
        dense_weight=dense_weight,
    )


def hybrid_retrieve_rules(
    query: str,
    kb: KnowledgeBase,
    client: BailianClient,
    retriever: HybridRetriever = None,
    top_k: int = 6,
) -> List[Dict]:
    """
    使用混合检索获取相关法规

    Args:
        query: 查询文本
        kb: 知识库（用于向后兼容）
        client: 百炼客户端
        retriever: 混合检索器（可选，如果不提供则使用向量检索）
        top_k: 返回结果数量

    Returns:
        检索结果列表
    """
    if retriever is not None:
        # 使用混合检索
        return retriever.retrieve(query, client, top_k=top_k)
    else:
        # 回退到向量检索
        from .knowledge_base import retrieve_relevant_rules
        return retrieve_relevant_rules(query, kb, client, top_k=top_k)
