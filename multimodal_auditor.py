"""
多模态营销内容审核模块

支持图片、图文混合内容的合规审核
"""

import json
import re
from typing import Any

from .bailian_client import BailianClient
from .knowledge_base import KnowledgeBase, retrieve_relevant_rules


def _extract_json_block(text: str) -> dict[str, Any]:
    """从文本中提取JSON块"""
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
    """构建规则上下文字符串"""
    rows: list[str] = []
    for i, rule in enumerate(rules, start=1):
        rows.append(
            f"[{i}] 来源: {rule['source_file']} | 条文: {rule['clause_id']}\n"
            f"条文原文: {rule['clause_text']}"
        )
    return "\n\n".join(rows)


def audit_marketing_image(
    image_bytes: bytes,
    kb: KnowledgeBase,
    client: BailianClient,
    image_mime: str = "image/png",
    text_context: str | None = None,
    top_k: int = 6,
) -> dict[str, Any]:
    """审核营销图片

    Args:
        image_bytes: 图片字节数据
        kb: 知识库
        client: Bailian客户端
        image_mime: 图片MIME类型
        text_context: 可选的文字说明
        top_k: 检索的规则数量

    Returns:
        审核结果字典，包含:
        - is_compliant: 是否合规 (yes/no)
        - violations: 违规项列表
        - overall_confidence: 整体置信度
        - summary: 总结
        - retrieved_rules: 检索到的规则
        - image_analysis: 图片分析结果
    """

    # 第一步：使用多模态模型分析图片内容
    image_analysis_raw = client.analyze_marketing_image(
        image_bytes=image_bytes,
        image_mime=image_mime,
        text_context=text_context,
    )

    try:
        image_analysis = _extract_json_block(image_analysis_raw)
        marketing_content = image_analysis.get("marketing_content", "")
        extracted_text = image_analysis.get("extracted_text", "")
    except (ValueError, json.JSONDecodeError):
        # 如果JSON解析失败，使用原始文本作为营销内容
        marketing_content = image_analysis_raw
        extracted_text = image_analysis_raw
        image_analysis = {
            "extracted_text": image_analysis_raw,
            "visual_elements": [],
            "marketing_content": image_analysis_raw,
            "detected_issues": [],
        }

    # 如果没有提取到有效内容，返回默认结果
    if not marketing_content or len(marketing_content.strip()) < 10:
        return {
            "is_compliant": "unknown",
            "violations": [],
            "overall_confidence": 0.0,
            "summary": "无法从图片中提取有效的营销内容",
            "retrieved_rules": [],
            "image_analysis": image_analysis,
            "raw_model_output": image_analysis_raw,
        }

    # 第二步：基于提取的内容进行向量检索
    # 组合提取的文字和文字说明
    search_query = marketing_content
    if text_context:
        search_query = f"{text_context}\n{marketing_content}"

    rules = retrieve_relevant_rules(
        query=search_query,
        kb=kb,
        client=client,
        top_k=top_k,
    )

    # 第三步：使用多模态模型进行合规分析
    rule_context = _build_rule_context(rules)

    system_prompt = (
        "你是金融保险营销合规审核助手。"
        "你只能依据给定监管条文进行判断，不得编造条文。"
        "输出必须是严格JSON，不要输出Markdown。"
    )

    # 构建多模态分析提示词
    if text_context:
        user_prompt = f"""请审核以下保险营销内容（图文混合）是否合规。

【文字说明】
{text_context}

【图片内容】
提取的文字：{extracted_text}
营销内容描述：{marketing_content}
视觉元素：{', '.join(image_analysis.get('visual_elements', []))}

【可参考监管条文】
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
4. 重点关注图片中的夸大宣传、保证收益、明星代言等违规内容。"""
    else:
        user_prompt = f"""请审核以下保险营销图片内容是否合规。

【图片分析】
提取的文字：{extracted_text}
营销内容描述：{marketing_content}
视觉元素：{', '.join(image_analysis.get('visual_elements', []))}

【可参考监管条文】
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
4. 重点关注图片中的夸大宣传、保证收益、明星代言等违规内容。"""

    # 使用多模态模型进行最终分析
    image_base64 = image_bytes  # 已经是bytes，不需要再编码
    # 在实际调用时需要重新编码为base64 URL
    import base64
    image_base64_b64 = base64.b64encode(image_bytes).decode("utf-8")
    image_url = f"data:{image_mime};base64,{image_base64_b64}"

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        },
    ]

    raw_result = client.chat(
        messages=messages,
        model=client.settings.vl_model,
        temperature=0.0,
        max_tokens=client.settings.max_tokens,
    )

    try:
        parsed = _extract_json_block(raw_result)
    except ValueError:
        # 如果JSON解析失败，返回默认结果
        parsed = {
            "is_compliant": "unknown",
            "violations": [],
            "overall_confidence": 0.0,
            "summary": "审核结果解析失败",
        }

    parsed["retrieved_rules"] = rules
    parsed["image_analysis"] = image_analysis
    parsed["extracted_text"] = extracted_text
    parsed["raw_model_output"] = raw_result

    return parsed


def audit_marketing_multimodal(
    content: str | bytes | tuple[str, bytes],
    kb: KnowledgeBase,
    client: BailianClient,
    content_type: str = "text",  # "text", "image", "multimodal"
    top_k: int = 6,
) -> dict[str, Any]:
    """统一的多模态审核入口

    Args:
        content: 内容
            - str: 纯文本
            - bytes: 纯图片
            - tuple[str, bytes]: (文字说明, 图片字节) 图文混合
        kb: 知识库
        client: Bailian客户端
        content_type: 内容类型 ("text", "image", "multimodal")
        top_k: 检索规则数量

    Returns:
        审核结果字典
    """
    if content_type == "text":
        # 纯文本审核，调用原有审核器
        from .auditor import audit_marketing_text
        return audit_marketing_text(content, kb, client, top_k)

    elif content_type == "image":
        # 纯图片审核
        return audit_marketing_image(
            image_bytes=content,
            kb=kb,
            client=client,
            top_k=top_k,
        )

    elif content_type == "multimodal":
        # 图文混合审核
        text_context, image_bytes = content
        return audit_marketing_image(
            image_bytes=image_bytes,
            kb=kb,
            client=client,
            text_context=text_context,
            top_k=top_k,
        )

    else:
        raise ValueError(f"Unknown content_type: {content_type}")
