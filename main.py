import argparse
import json
from pathlib import Path

from .auditor import audit_marketing_text
from .bailian_client import BailianClient
from .config import Settings
from .evaluate import evaluate_dataset
from .extractors import extract_text_from_file
from .knowledge_base import build_knowledge_base, load_knowledge_base, KnowledgeBase, retrieve_relevant_rules
from . import knowledge_base_milvus


DEFAULT_DOCS = [
    "互联网保险业务监管办法.docx",
    "保险销售行为管理办法.pdf",
    "金融产品网络营销管理办法（征求意见稿）.doc",
]


def _default_doc_paths(cwd: Path) -> list[str]:
    paths: list[str] = []
    for name in DEFAULT_DOCS:
        p = cwd / name
        if p.exists():
            paths.append(str(p))
    return paths


def cmd_build_kb(args: argparse.Namespace) -> None:
    settings = Settings.from_env()
    settings.validate()
    settings.vector_db_type = args.vector_db  # 设置向量数据库类型
    client = BailianClient(settings)

    if args.docs:
        docs = [str(Path(p).resolve()) for p in args.docs]
    else:
        docs = _default_doc_paths(Path.cwd())

    if not docs:
        raise RuntimeError(
            "No source docs found. Pass --docs explicitly or run in aliyun_project directory."
        )

    # 根据向量数据库类型选择实现
    if args.vector_db == "milvus":
        kb = knowledge_base_milvus.build_knowledge_base(
            doc_paths=docs,
            output_dir=args.kb_dir,
            client=client,
            collection_name=args.collection_name,
            pdf_mode=args.pdf_mode,
            max_pages=args.max_pages,
            overwrite=args.overwrite,
        )
    else:  # numpy
        kb = build_knowledge_base(
            doc_paths=docs,
            output_dir=args.kb_dir,
            client=client,
            pdf_mode=args.pdf_mode,
            max_pages=args.max_pages,
        )

    print(f"KB built: {args.kb_dir}")
    print(f"Total chunks: {len(kb.chunks)}")
    print(f"Vector DB: {args.vector_db}")


def cmd_audit_text(args: argparse.Namespace) -> None:
    settings = Settings.from_env()
    settings.validate()
    settings.vector_db_type = args.vector_db
    client = BailianClient(settings)

    # 根据向量数据库类型加载知识库
    if args.vector_db == "milvus":
        kb = knowledge_base_milvus.load_knowledge_base(
            collection_name=args.collection_name,
            meta_dir=args.kb_dir,
        )
    else:  # numpy
        kb = load_knowledge_base(args.kb_dir)

    result = audit_marketing_text(
        marketing_text=args.text,
        kb=kb,
        client=client,
        top_k=args.top_k,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def cmd_audit_file(args: argparse.Namespace) -> None:
    settings = Settings.from_env()
    settings.validate()
    settings.vector_db_type = args.vector_db
    client = BailianClient(settings)

    # 根据向量数据库类型加载知识库
    if args.vector_db == "milvus":
        kb = knowledge_base_milvus.load_knowledge_base(
            collection_name=args.collection_name,
            meta_dir=args.kb_dir,
        )
    else:  # numpy
        kb = load_knowledge_base(args.kb_dir)

    extracted_text = extract_text_from_file(
        file_path=args.file,
        client=client,
        pdf_mode=args.pdf_mode,
        max_pages=args.max_pages,
    )

    result = audit_marketing_text(
        marketing_text=extracted_text,
        kb=kb,
        client=client,
        top_k=args.top_k,
    )
    result["input_file"] = args.file
    result["extracted_text_preview"] = extracted_text[:1000]

    print(json.dumps(result, ensure_ascii=False, indent=2))


def cmd_evaluate(args: argparse.Namespace) -> None:
    settings = Settings.from_env()
    settings.validate()
    client = BailianClient(settings)

    metrics = evaluate_dataset(
        dataset_path=args.dataset,
        kb_dir=args.kb_dir,
        client=client,
        limit=args.limit,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aliyun_rag",
        description="Quick runnable insurance compliance demo with Bailian + Qwen.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build-kb", help="Extract regulations and build vector KB.")
    p_build.add_argument("--docs", nargs="+", help="Regulation file paths.")
    p_build.add_argument("--kb-dir", default="kb", help="KB output directory.")
    p_build.add_argument(
        "--pdf-mode",
        choices=["vl", "native"],
        default="vl",
        help="PDF extraction mode: vl = Qwen-VL OCR, native = direct PDF text parser.",
    )
    p_build.add_argument("--max-pages", type=int, default=40, help="Max PDF pages for VL OCR.")
    p_build.add_argument(
        "--vector-db",
        choices=["numpy", "milvus"],
        default="numpy",
        help="Vector database type: numpy (local files) or milvus (Milvus Lite)",
    )
    p_build.add_argument(
        "--collection-name",
        default="insurance_knowledge",
        help="Milvus collection name (only used when --vector-db=milvus)",
    )
    p_build.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing collection (Milvus only)",
    )
    p_build.set_defaults(func=cmd_build_kb)

    p_text = sub.add_parser("audit-text", help="Audit a marketing text.")
    p_text.add_argument("--text", required=True, help="Marketing text to audit.")
    p_text.add_argument("--kb-dir", default="kb")
    p_text.add_argument("--top-k", type=int, default=6)
    p_text.add_argument(
        "--vector-db",
        choices=["numpy", "milvus"],
        default="numpy",
        help="Vector database type: numpy (local files) or milvus (Milvus Lite)",
    )
    p_text.add_argument(
        "--collection-name",
        default="insurance_knowledge",
        help="Milvus collection name (only used when --vector-db=milvus)",
    )
    p_text.set_defaults(func=cmd_audit_text)

    p_file = sub.add_parser("audit-file", help="Audit text extracted from a file.")
    p_file.add_argument("--file", required=True, help="Input file (.pdf/.docx/.doc/image)")
    p_file.add_argument("--kb-dir", default="kb")
    p_file.add_argument("--top-k", type=int, default=6)
    p_file.add_argument(
        "--pdf-mode",
        choices=["vl", "native"],
        default="vl",
        help="PDF extraction mode used for the input file.",
    )
    p_file.add_argument("--max-pages", type=int, default=20)
    p_file.add_argument(
        "--vector-db",
        choices=["numpy", "milvus"],
        default="numpy",
        help="Vector database type: numpy (local files) or milvus (Milvus Lite)",
    )
    p_file.add_argument(
        "--collection-name",
        default="insurance_knowledge",
        help="Milvus collection name (only used when --vector-db=milvus)",
    )
    p_file.set_defaults(func=cmd_audit_file)

    p_eval = sub.add_parser("evaluate", help="Run quick evaluation by demo dataset.")
    p_eval.add_argument("--dataset", default="demo_cases.jsonl")
    p_eval.add_argument("--kb-dir", default="kb")
    p_eval.add_argument("--limit", type=int, default=None)
    p_eval.set_defaults(func=cmd_evaluate)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

