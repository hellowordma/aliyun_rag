"""
增强型审核器 - 多阶段推理

实现：
1. 意图识别 - 识别营销文案的类型和风险点
2. 问题重写 - 将营销文案转化为更精确的检索查询
3. 多阶段推理 - 逐步深入分析，处理隐含逻辑和上下文依赖
"""

import json
import re
from typing import Any, List, Dict, TYPE_CHECKING

from .bailian_client import BailianClient
from .confidence_calculator import ConfidenceCalculator

if TYPE_CHECKING:
    from . import knowledge_base_milvus


class EnhancedAuditor:
    """增强型合规审核器"""

    # 违规类型映射（意图识别关键词）
    VIOLATION_PATTERNS = {
        "承诺保证收益": [
            "保证", "承诺", "保本", "零风险", "稳赚不赔",
            "年化收益", "固定收益", "必定", "100%"
        ],
        "夸大宣传": [
            "最高", "第一", "最佳", "唯一", "顶级",
            "最专业", "最好", "无双", "之最"
        ],
        "误导性宣传": [
            "相当于", "等同于", "类似于", "类似于存款",
            "比银行好", "替代银行", "理财产品"
        ],
        "无证代言": [
            "代言", "推荐", "明星", "专家",
            "认证", "授权", "官方合作"
        ],
        "风险提示不足": [
            "无风险", "低风险", "安全可靠",
            "必定盈利", "只赚不赔"
        ],
        "资质问题": [
            "无证", "未备案", "黑户", "地下",
            "非法", "违规销售"
        ],
        "销售诱导": [
            "限时", "最后机会", "错过不再",
            "立即下单", "马上购买", "赶紧抢"
        ],
    }

    @classmethod
    def identify_intent(cls, marketing_text: str) -> Dict[str, Any]:
        """
        意图识别 - 分析营销文案的风险点和类型

        返回：
        {
            "primary_intent": "主要意图",
            "detected_risks": ["检测到的风险1", "风险2"],
            "risk_level": "high/medium/low",
            "keyword_matches": {"关键词": 出现次数}
        }
        """
        detected_risks = []
        keyword_matches = {}
        max_score = 0
        primary_intent = "其他"

        # 检测每种违规类型
        for violation_type, keywords in cls.VIOLATION_PATTERNS.items():
            matched_keywords = []
            for kw in keywords:
                count = marketing_text.count(kw)
                if count > 0:
                    matched_keywords.append((kw, count))

            if matched_keywords:
                score = sum(count for _, count in matched_keywords)
                keyword_matches[violation_type] = dict(matched_keywords)
                detected_risks.append(violation_type)

                if score > max_score:
                    max_score = score
                    primary_intent = violation_type

        # 评估风险等级
        risk_level = cls._assess_risk_level(detected_risks, keyword_matches)

        return {
            "primary_intent": primary_intent,
            "detected_risks": detected_risks,
            "risk_level": risk_level,
            "keyword_matches": keyword_matches,
        }

    @staticmethod
    def _assess_risk_level(detected_risks: List[str], keyword_matches: Dict) -> str:
        """评估风险等级"""
        # 高风险关键词
        high_risk_keywords = ["保证", "承诺", "稳赚不赔", "零风险", "无证"]

        # 检查是否包含高风险关键词
        for risk_type, matches in keyword_matches.items():
            for kw in matches:
                if kw in high_risk_keywords:
                    return "high"

        # 中等风险：有2个以上风险类型
        if len(detected_risks) >= 2:
            return "medium"

        # 低风险：0-1个风险类型
        return "low" if detected_risks else "none"

    @classmethod
    def rewrite_query(
        cls,
        marketing_text: str,
        intent_result: Dict[str, Any],
    ) -> List[str]:
        """
        问题重写 - 将营销文案转化为多个精确的检索查询

        生成多个查询以捕获不同维度的违规风险
        """
        queries = []

        # 原始查询
        queries.append(marketing_text)

        # 基于意图的查询重写
        primary_intent = intent_result["primary_intent"]

        if primary_intent == "承诺保证收益":
            queries.extend([
                "保险产品收益保证承诺 固定收益 保本保收益",
                "保险产品风险提示 收益不确定性说明",
            ])
        elif primary_intent == "夸大宣传":
            queries.extend([
                "保险营销绝对化用语 夸大宣传 最高最好",
                "保险营销合规用语 宣传规范",
            ])
        elif primary_intent == "误导性宣传":
            queries.extend([
                "保险产品性质说明 误导混淆 理财产品",
                "保险销售真实性要求 信息披露",
            ])
        elif primary_intent == "无证代言":
            queries.extend([
                "保险销售资质要求 代言推荐",
                "保险销售人员管理 执业登记",
            ])
        elif primary_intent == "风险提示不足":
            queries.extend([
                "保险产品风险提示义务 信息披露",
                "保险条款说明 责任免除 犹豫期",
            ])

        # 添加通用合规查询
        if intent_result["risk_level"] in ["high", "medium"]:
            queries.extend([
                "保险销售合规要求 禁止性规定",
                "互联网保险业务监管 营销宣传规范",
            ])

        return list(set(queries))  # 去重

    @classmethod
    def multi_stage_audit(
        cls,
        marketing_text: str,
        kb: Any,  # KnowledgeBase or MilvusKnowledgeBase
        client: BailianClient,
        top_k: int = 6,
        enable_math_confidence: bool = True,
    ) -> Dict[str, Any]:
        """
        多阶段审核流程

        阶段1: 意图识别
        阶段2: 问题重写与多轮检索
        阶段3: 综合分析与违规判定
        阶段4: 置信度计算
        """

        # ========== 阶段1: 意图识别 ==========
        intent_result = cls.identify_intent(marketing_text)

        # ========== 阶段2: 问题重写与多轮检索 ==========
        rewritten_queries = cls.rewrite_query(marketing_text, intent_result)

        # 执行多轮检索，收集所有相关条文
        all_retrieved_rules = []

        if hasattr(kb, 'collection'):  # MilvusKnowledgeBase
            from . import knowledge_base_milvus
            for query in rewritten_queries[:3]:  # 限制查询次数
                rules = knowledge_base_milvus.retrieve_relevant_rules(
                    query=query,
                    kb=kb,
                    client=client,
                    top_k=top_k,
                )
                all_retrieved_rules.extend(rules)
        else:  # NumPy KnowledgeBase
            from .knowledge_base import retrieve_relevant_rules
            for query in rewritten_queries[:3]:
                rules = retrieve_relevant_rules(
                    query=query,
                    kb=kb,
                    client=client,
                    top_k=top_k,
                )
                all_retrieved_rules.extend(rules)

        # 去重并排序
        unique_rules = cls._deduplicate_rules(all_retrieved_rules)
        unique_rules = sorted(unique_rules, key=lambda x: x['score'], reverse=True)[:top_k * 2]

        # ========== 阶段3: 综合分析与违规判定 ==========
        # 构建增强的 Prompt
        system_prompt = cls._build_enhanced_system_prompt(intent_result)

        user_prompt = cls._build_enhanced_user_prompt(
            marketing_text=marketing_text,
            retrieved_rules=unique_rules,
            intent_result=intent_result,
        )

        # 调用 LLM
        raw = client.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )

        # 解析结果
        parsed = cls._extract_json(raw)

        # ========== 阶段4: 置信度计算 ==========
        if enable_math_confidence:
            confidence_result = ConfidenceCalculator.calculate_enhanced_confidence(
                retrieved_rules=unique_rules,
                llm_output=parsed,
                query_text=marketing_text,
            )
            parsed.update(confidence_result)

        # 添加元数据
        parsed["intent_analysis"] = intent_result
        parsed["rewritten_queries"] = rewritten_queries[:3]  # 只保留前3个
        parsed["retrieved_rules"] = unique_rules
        parsed["raw_model_output"] = raw

        return parsed

    @staticmethod
    def _build_enhanced_system_prompt(intent_result: Dict[str, Any]) -> str:
        """构建增强的系统提示词"""

        primary_intent = intent_result["primary_intent"]
        risk_level = intent_result["risk_level"]

        return f"""你是专业的金融保险营销合规审核专家。

当前审核的营销文案主要涉及：{primary_intent}（风险等级：{risk_level}）

审核原则：
1. 全面分析：结合显性违规和隐性违规
2. 上下文理解：考虑条款之间的关联性和依赖关系
3. 严格判定：宁可误判为违规，不可漏判
4. 条文引用：必须引用具体的条文编号和原文

注意事项：
- 关注"隐含逻辑"和"上下文依赖"，如条款间的相互引用、补充说明等
- 考虑"禁止用语"、"资质要求"、"风险提示"等多个维度
- 对于模糊表述，应从严判断

输出格式：严格JSON，不要输出Markdown。"""

    @staticmethod
    def _build_enhanced_user_prompt(
        marketing_text: str,
        retrieved_rules: List[Dict],
        intent_result: Dict[str, Any],
    ) -> str:
        """构建增强的用户提示词"""

        # 构建条文上下文（添加条文间的关系）
        rule_context_parts = []
        for i, rule in enumerate(retrieved_rules[:10], 1):  # 展示更多条文
            context = f"[{i}] 来源: {rule['source_file']} | 条文: {rule['clause_id']}\n"
            context += f"相似度: {rule['score']:.4f}\n"
            context += f"条文原文: {rule['clause_text']}"
            rule_context_parts.append(context)

        rule_context = "\n\n".join(rule_context_parts)

        # 意图分析结果
        intent_desc = f"""
主要意图：{intent_result['primary_intent']}
检测到的风险：{', '.join(intent_result['detected_risks'])}
风险等级：{intent_result['risk_level']}
"""

        return f"""请审核以下营销内容是否合规。

营销内容：
{marketing_text}

意图分析：
{intent_desc}

可参考监管条文（共{len(retrieved_rules)}条，按相关度排序）：
{rule_context}

请输出严格JSON，字段必须完整：
{{
  "is_compliant": "yes 或 no",
  "violations": [
    {{
      "type": "违规类型",
      "clause_id": "条文编号",
      "clause_text": "条文原文",
      "reason": "违规原因（详细分析，包括显性和隐性违规）",
      "confidence": 0.0,
      "implicit_violations": ["可能存在的隐含违规1", "隐含违规2"]
    }}
  ],
  "overall_confidence": 0.0,
  "summary": "一句话总结（包括主要违规点和风险等级）",
  "context_analysis": "上下文依赖和隐含逻辑分析"
}}

要求：
1. 如果合规，violations 返回空数组。
2. confidence 与 overall_confidence 取值范围 [0,1]。
3. 必须引用上方出现的条文编号与原文。
4. implicit_violations 字段列出可能的隐含违规（如条款间的关联违规）。
5. context_analysis 字段分析上下文依赖和隐含逻辑。
6. 考虑多个维度：文字表述、禁止用语、资质要求、风险提示。"""

    @staticmethod
    def _extract_json(text: str) -> Dict[str, Any]:
        """从文本中提取 JSON"""
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError("Model output does not contain a valid JSON object.")
        return json.loads(match.group(0))

    @staticmethod
    def _deduplicate_rules(rules: List[Dict]) -> List[Dict]:
        """去重条文（基于条文编号）"""
        seen = set()
        unique_rules = []

        for rule in rules:
            # 使用条文编号 + 文件名作为唯一标识
            key = (rule['clause_id'], rule['source_file'])
            if key not in seen:
                seen.add(key)
                unique_rules.append(rule)

        return unique_rules


# 导出便捷函数
def enhanced_audit_marketing_text(
    marketing_text: str,
    kb: Any,
    client: BailianClient,
    top_k: int = 6,
    enable_math_confidence: bool = True,
) -> Dict[str, Any]:
    """
    增强型营销文案审核

    使用多阶段推理流程，包括意图识别、问题重写、综合分析和置信度计算
    """
    return EnhancedAuditor.multi_stage_audit(
        marketing_text=marketing_text,
        kb=kb,
        client=client,
        top_k=top_k,
        enable_math_confidence=enable_math_confidence,
    )
