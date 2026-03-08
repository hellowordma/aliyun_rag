import json
from pathlib import Path

from .auditor import audit_marketing_text
from .bailian_client import BailianClient
from .knowledge_base import load_knowledge_base


def evaluate_dataset(
    dataset_path: str,
    kb_dir: str,
    client: BailianClient,
    limit: int | None = None,
) -> dict:
    kb = load_knowledge_base(kb_dir)
    rows = [
        json.loads(line)
        for line in Path(dataset_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if limit:
        rows = rows[:limit]

    if not rows:
        raise RuntimeError("Dataset is empty.")

    total = 0
    correct = 0
    details: list[dict] = []

    for row in rows:
        text = row["text"]
        expected = row["is_compliant"].strip().lower()
        result = audit_marketing_text(text, kb=kb, client=client)
        pred = str(result.get("is_compliant", "")).strip().lower()
        is_hit = pred == expected

        total += 1
        if is_hit:
            correct += 1

        details.append(
            {
                "text": text,
                "expected": expected,
                "predicted": pred,
                "hit": is_hit,
                "summary": result.get("summary", ""),
            }
        )

    return {
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 4),
        "details": details,
    }
