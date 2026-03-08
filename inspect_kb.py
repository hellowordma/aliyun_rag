#!/usr/bin/env python3
"""
RAG知识库检查脚本 - 查看Chunk和向量详情

使用方法:
    PYTHONPATH=/mnt/workspace:$PYTHONPATH python inspect_kb.py
"""

import json
import sys
from pathlib import Path

import numpy as np

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))


def inspect_knowledge_base(kb_dir: str = "kb"):
    """检查知识库详情"""

    kb_path = Path(kb_dir)

    if not kb_path.exists():
        print(f"❌ 知识库目录不存在: {kb_dir}")
        return

    print("=" * 70)
    print("📚 RAG 知识库详情")
    print("=" * 70)
    print()

    # 1. 读取元数据
    meta_file = kb_path / "meta.json"
    if meta_file.exists():
        with meta_file.open("r", encoding="utf-8") as f:
            meta = json.load(f)

        print("📋 元数据:")
        print(f"  创建时间: {meta.get('created_at', 'N/A')}")
        print(f"  总Chunk数: {meta.get('num_chunks', 'N/A')}")
        print(f"  嵌入模型: {meta.get('embedding_model', 'N/A')}")
        print(f"  PDF模式: {meta.get('pdf_mode', 'N/A')}")
        print()

    # 2. 分析Chunks
    chunks_file = kb_path / "chunks.jsonl"
    if chunks_file.exists():
        chunks = []
        with chunks_file.open("r", encoding="utf-8") as f:
            for line in f:
                chunks.append(json.loads(line))

        print("📦 Chunk 统计:")
        print(f"  总数量: {len(chunks)}")

        # 文本长度统计
        lengths = [len(c["clause_text"]) for c in chunks]
        print(f"  平均长度: {sum(lengths) / len(lengths):.0f} 字符")
        print(f"  最大长度: {max(lengths)} 字符")
        print(f"  最小长度: {min(lengths)} 字符")

        # 按条文类型统计
        clause_types = {}
        for chunk in chunks:
            clause_id = chunk["clause_id"]
            if "条" in clause_id:
                # 提取条文编号
                base_clause = clause_id.split("-")[0]
                clause_types[base_clause] = clause_types.get(base_clause, 0) + 1

        print(f"  条文种类: {len(clause_types)} 种")
        print()

        # 分段统计
        segmented = [c for c in chunks if "-" in c["clause_id"]]
        if segmented:
            print("🔪 分段情况:")
            print(f"  需要分段的条文: {len(set(c['clause_id'].split('-')[0] for c in segmented))} 条")
            print(f"  分段后Chunk数: {len(segmented)} 个")
            print()

        # 来源文件统计
        source_files = {}
        for chunk in chunks:
            source = chunk["source_file"]
            source_files[source] = source_files.get(source, 0) + 1

        print("📄 来源文件:")
        for source, count in source_files.items():
            print(f"  {source}: {count} chunks")
        print()

        # 显示示例Chunks
        print("📝 Chunk 示例 (前3个):")
        for i, chunk in enumerate(chunks[:3], 1):
            preview = chunk["clause_text"][:100] + "..." if len(chunk["clause_text"]) > 100 else chunk["clause_text"]
            print(f"\n  [{i}] 文件: {chunk['source_file']}")
            print(f"      条文: {chunk['clause_id']}")
            print(f"      内容: {preview}")
        print()

    # 3. 分析向量
    embeddings_file = kb_path / "embeddings.npy"
    if embeddings_file.exists():
        embeddings = np.load(embeddings_file)

        print("🔢 向量矩阵:")
        print(f"  形状: {embeddings.shape}")
        print(f"  维度: {embeddings.shape[1]} (text-embedding-v3)")
        print(f"  数据类型: {embeddings.dtype}")
        print(f"  文件大小: {embeddings.nbytes / 1024:.1f} KB")
        print(f"  内存占用: {embeddings.nbytes / (1024 * 1024):.2f} MB")

        # 向量统计
        print(f"\n  向量统计:")
        print(f"    最大值: {np.max(embeddings):.4f}")
        print(f"    最小值: {np.min(embeddings):.4f}")
        print(f"    平均值: {np.mean(embeddings):.4f}")
        print(f"    标准差: {np.std(embeddings):.4f}")

        # 归一化检查
        norms = np.linalg.norm(embeddings, axis=1)
        print(f"\n  L2范数:")
        print(f"    最大范数: {np.max(norms):.4f}")
        print(f"    最小范数: {np.min(norms):.4f}")
        print(f"    平均范数: {np.mean(norms):.4f}")
        print(f"    是否已归一化: {np.allclose(norms, 1.0, atol=1e-3)}")
        print()

    # 4. 存储效率分析
    if chunks_file.exists() and embeddings_file.exists():
        chunks_size = chunks_file.stat().st_size
        embeddings_size = embeddings_file.stat().st_size
        total_size = chunks_size + embeddings_size

        print("💾 存储分析:")
        print(f"  chunks.jsonl: {chunks_size / 1024:.1f} KB")
        print(f"  embeddings.npy: {embeddings_size / 1024:.1f} KB")
        print(f"  总大小: {total_size / 1024:.1f} KB")
        print(f"  平均每Chunk: {total_size / len(chunks) / 1024:.2f} KB")
        print()

    # 5. 相似度矩阵示例（可选，如果chunks不多）
    if embeddings_file.exists() and len(chunks) <= 200:
        print("🔍 相似度分析:")
        embeddings = np.load(embeddings_file)

        # 归一化
        emb_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)

        # 计算相似度矩阵
        similarity_matrix = emb_norm @ emb_norm.T

        # 找出最相似的对（排除自身）
        np.fill_diagonal(similarity_matrix, -1)
        max_sim_idx = np.unravel_index(np.argmax(similarity_matrix), similarity_matrix.shape)

        print(f"  最相似Chunk对:")
        idx1, idx2 = max_sim_idx
        print(f"    Chunk {idx1}: {chunks[idx1]['clause_id']}")
        print(f"    Chunk {idx2}: {chunks[idx2]['clause_id']}")
        print(f"    相似度: {similarity_matrix[idx1, idx2]:.4f}")

        # 相似度分布
        upper_tri = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
        print(f"\n  相似度分布:")
        print(f"    最高: {np.max(upper_tri):.4f}")
        print(f"    最低: {np.min(upper_tri):.4f}")
        print(f"    平均: {np.mean(upper_tri):.4f}")
        print(f"    中位数: {np.median(upper_tri):.4f}")
        print()

    print("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="检查RAG知识库详情")
    parser.add_argument("--kb-dir", default="kb", help="知识库目录")
    args = parser.parse_args()

    inspect_knowledge_base(args.kb_dir)
