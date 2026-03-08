import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from .bailian_client import BailianClient
from .extractors import extract_text_from_file

ARTICLE_PATTERN = re.compile(r"^(第[一二三四五六七八九十百零〇0-9]+条)")


@dataclass
class RuleChunk:
    source_file: str
    clause_id: str
    clause_text: str


@dataclass
class KnowledgeBase:
    chunks: list[RuleChunk]
    embeddings: np.ndarray


def split_into_rule_chunks(
    source_file: str,
    text: str,
    max_chunk_chars: int = 700,
) -> list[RuleChunk]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    chunks: list[RuleChunk] = []

    current_clause = "未标注条文"
    buffer: list[str] = []

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


def build_knowledge_base(
    doc_paths: list[str],
    output_dir: str,
    client: BailianClient,
    pdf_mode: str = "vl",
    max_pages: int | None = None,
) -> KnowledgeBase:
    all_chunks: list[RuleChunk] = []

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

    embeddings = client.embed_texts([chunk.clause_text for chunk in all_chunks])

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    with (output / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")

    np.save(output / "embeddings.npy", embeddings)

    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "num_chunks": len(all_chunks),
        "embedding_model": client.settings.embedding_model,
        "pdf_mode": pdf_mode,
    }
    with (output / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return KnowledgeBase(chunks=all_chunks, embeddings=embeddings)


def load_knowledge_base(output_dir: str) -> KnowledgeBase:
    output = Path(output_dir)
    chunks_path = output / "chunks.jsonl"
    emb_path = output / "embeddings.npy"

    if not chunks_path.exists() or not emb_path.exists():
        raise FileNotFoundError(
            f"Knowledge base files not found in {output}. Run build-kb first."
        )

    chunks: list[RuleChunk] = []
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            chunks.append(RuleChunk(**row))

    embeddings = np.load(emb_path)
    return KnowledgeBase(chunks=chunks, embeddings=embeddings)


def retrieve_relevant_rules(
    query: str,
    kb: KnowledgeBase,
    client: BailianClient,
    top_k: int = 6,
) -> list[dict]:
    query_vec = client.embed_texts([query])[0]
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-12)

    emb = kb.embeddings
    emb_norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)

    scores = emb_norm @ query_norm
    top_indices = np.argsort(scores)[::-1][:top_k]

    result: list[dict] = []
    for idx in top_indices:
        chunk = kb.chunks[int(idx)]
        result.append(
            {
                "score": float(scores[idx]),
                "source_file": chunk.source_file,
                "clause_id": chunk.clause_id,
                "clause_text": chunk.clause_text,
            }
        )
    return result
