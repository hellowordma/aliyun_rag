import json
import re
from typing import Any

from .bailian_client import BailianClient
from .knowledge_base import KnowledgeBase, retrieve_relevant_rules


def _extract_json_block(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("Model output does not contain a valid JSON object.")
    return json.loads(match.group(0))


def _build_rule_context(rules: list[dict]) -> str:
    rows: list[str] = []
    for i, rule in enumerate(rules, start=1):
        rows.append(
            f"[{i}] 来源: {rule['source_file']} | 条文: {rule['clause_id']}\n"
            f"条文原文: {rule['clause_text']}"
        )
    return "\n\n".join(rows)


def audit_marketing_text(
    marketing_text: str,
    kb: KnowledgeBase,
    client: BailianClient,
    top_k: int = 6,
) -> dict[str, Any]:
    rules = retrieve_relevant_rules(
        query=marketing_text,
        kb=kb,
        client=client,
        top_k=top_k,
    )

    rule_context = _build_rule_context(rules)

    system_prompt = (
        "你是金融保险营销合规审核助手。"
        "你只能依据给定监管条文进行判断，不得编造条文。"
        "输出必须是严格JSON，不要输出Markdown。"
    )

    user_prompt = f"""
请审核以下营销内容是否合规。

营销内容：
{marketing_text}

可参考监管条文：
{rule_context}

请输出严格JSON，字段必须完整：
{{
  "is_compliant": "yes 或 no",
  "violations": [
    {{
      "type": "违规类型",
      "clause_id": "条文编号",
      "clause_text": "条文原文",
      "reason": "违规原因",
      "confidence": 0.0
    }}
  ],
  "overall_confidence": 0.0,
  "summary": "一句话总结"
}}

要求：
1. 如果合规，violations 返回空数组。
2. confidence 与 overall_confidence 取值范围 [0,1]。
3. 必须引用上方出现的条文编号与原文。
""".strip()

    raw = client.chat(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
    )

    parsed = _extract_json_block(raw)
    parsed["retrieved_rules"] = rules
    parsed["raw_model_output"] = raw
    return parsed
