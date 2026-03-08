"""
置信度计算模块

使用数学公式计算置信度，而非仅依赖大模型输出。
"""

import numpy as np
from typing import List, Dict, Any


class ConfidenceCalculator:
    """置信度计算器"""

    @staticmethod
    def calculate_overall_confidence(
        retrieved_rules: List[Dict],
        violations: List[Dict],
        query_embedding: np.ndarray = None,
        kb_embeddings: np.ndarray = None,
    ) -> float:
        """
        计算整体置信度

        综合考虑：
        1. 检索相关度（向量相似度）
        2. 违规项数量和置信度
        3. 条文匹配度

        公式：
        overall_confidence = w1 * retrieval_score + w2 * violation_score + w3 * clause_match_score

        其中：
        - retrieval_score: 检索相关度分数
        - violation_score: 违规项置信度分数
        - clause_match_score: 条文匹配度分数
        - w1, w2, w3: 权重系数（默认 0.3, 0.5, 0.2）
        """

        # 1. 检索相关度分数（基于相似度）
        retrieval_score = ConfidenceCalculator._calculate_retrieval_score(retrieved_rules)

        # 2. 违规项置信度分数
        violation_score = ConfidenceCalculator._calculate_violation_score(violations)

        # 3. 条文匹配度分数（检查是否引用了检索到的条文）
        clause_match_score = ConfidenceCalculator._calculate_clause_match_score(
            retrieved_rules, violations
        )

        # 权重配置
        w1, w2, w3 = 0.3, 0.5, 0.2

        # 加权平均
        overall_confidence = w1 * retrieval_score + w2 * violation_score + w3 * clause_match_score

        # 确保在 [0, 1] 范围内
        return max(0.0, min(1.0, overall_confidence))

    @staticmethod
    def _calculate_retrieval_score(retrieved_rules: List[Dict]) -> float:
        """
        计算检索相关度分数

        公式：
        retrieval_score = (1/n) * Σ similarity_score_i

        其中 similarity_score_i = sigmoid(5 * (score_i - 0.5))
        """
        if not retrieved_rules:
            return 0.0

        scores = [rule.get('score', 0.0) for rule in retrieved_rules]

        # 使用 sigmoid 将相似度映射到 [0, 1]
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        # 对每个分数进行 sigmoid 变换，放大差异
        transformed_scores = [sigmoid(5 * (score - 0.5)) for score in scores]

        # 返回平均值
        return float(np.mean(transformed_scores))

    @staticmethod
    def _calculate_violation_score(violations: List[Dict]) -> float:
        """
        计算违规项置信度分数

        如果没有违规，返回 1.0（高度确信合规）
        如果有违规，返回违规项置信度的加权平均

        公式：
        - 无违规: violation_score = 1.0
        - 有违规: violation_score = (1/n) * Σ confidence_i
        """
        if not violations:
            return 1.0  # 没有违规，高度确信合规

        confidences = [v.get('confidence', 0.0) for v in violations]
        return float(np.mean(confidences))

    @staticmethod
    def _calculate_clause_match_score(
        retrieved_rules: List[Dict],
        violations: List[Dict],
    ) -> float:
        """
        计算条文匹配度分数

        检查违规项引用的条文是否在检索到的条文中

        公式：
        clause_match_score = matched_clauses / total_violations
        """
        if not violations:
            return 1.0

        # 获取检索到的条文ID集合
        retrieved_clause_ids = {rule.get('clause_id', '') for rule in retrieved_rules}

        # 统计匹配数量
        matched_count = 0
        for violation in violations:
            clause_id = violation.get('clause_id', '')
            if clause_id in retrieved_clause_ids:
                matched_count += 1

        return matched_count / len(violations)

    @staticmethod
    def calculate_violation_confidence(
        violation_type: str,
        clause_text: str,
        retrieved_rules: List[Dict],
        rule_relevance: float,
    ) -> float:
        """
        计算单个违规项的置信度

        综合考虑：
        1. 条文相关性（检索相似度）
        2. 违规类型明确性
        3. 条文长度（条文越长，越可能相关）

        公式：
        confidence = w1 * relevance + w2 * type_clarity + w3 * length_factor

        其中：
        - relevance: 检索相关度
        - type_clarity: 违规类型明确性（基于关键词匹配）
        - length_factor: 条文长度因子（条文越长，权重越高）
        """

        # 1. 检索相关度
        relevance = rule_relevance

        # 2. 违规类型明确性（基于关键词匹配）
        type_clarity = ConfidenceCalculator._calculate_type_clarity(
            violation_type, clause_text
        )

        # 3. 条文长度因子
        length_factor = ConfidenceCalculator._calculate_length_factor(clause_text)

        # 权重配置
        w1, w2, w3 = 0.5, 0.3, 0.2

        confidence = w1 * relevance + w2 * type_clarity + w3 * length_factor

        return max(0.0, min(1.0, confidence))

    @staticmethod
    def _calculate_type_clarity(violation_type: str, clause_text: str) -> float:
        """
        计算违规类型明确性

        基于关键词匹配度
        """
        # 违规类型关键词映射
        type_keywords = {
            "承诺保证收益": ["保证", "承诺", "保本", "收益", "稳赚"],
            "夸大宣传": ["夸大", "最高", "第一", "最佳", "唯一"],
            "误导性宣传": ["误导", "混淆", "相当于", "等同"],
            "无证代言": ["代言", "推荐", "认证", "授权"],
            "风险提示不足": ["风险", "提示", "告知", "说明"],
        }

        # 获取该类型的关键词
        keywords = type_keywords.get(violation_type, [])

        if not keywords:
            return 0.5  # 无关键词，返回中等明确性

        # 计算关键词匹配度
        matched_count = sum(1 for kw in keywords if kw in clause_text)
        match_ratio = matched_count / len(keywords)

        return float(match_ratio)

    @staticmethod
    def _calculate_length_factor(text: str) -> float:
        """
        计算条文长度因子

        条文越长，包含的信息越丰富，但也不是越长越好
        使用分段函数：
        - < 50 字符: 0.3 (太短，信息不足)
        - 50-200 字符: 0.5 + (len - 50) / 150 * 0.3
        - 200-500 字符: 0.8 + (len - 200) / 300 * 0.2
        - > 500 字符: 1.0 (足够长)
        """
        length = len(text)

        if length < 50:
            return 0.3
        elif length < 200:
            return 0.5 + (length - 50) / 150 * 0.3
        elif length < 500:
            return 0.8 + (length - 200) / 300 * 0.2
        else:
            return 1.0

    @staticmethod
    def calculate_enhanced_confidence(
        retrieved_rules: List[Dict],
        llm_output: Dict[str, Any],
        query_text: str,
        kb_embeddings: np.ndarray = None,
    ) -> Dict[str, Any]:
        """
        计算增强的置信度（综合数学公式和LLM输出）

        结合数学公式计算的置信度和LLM输出的置信度
        """
        # 提取 LLM 输出
        llm_overall_confidence = llm_output.get('overall_confidence', 0.5)
        violations = llm_output.get('violations', [])

        # 使用数学公式计算
        calculated_confidence = ConfidenceCalculator.calculate_overall_confidence(
            retrieved_rules=retrieved_rules,
            violations=violations,
        )

        # 加权融合（LLM 和数学公式）
        final_confidence = 0.4 * llm_overall_confidence + 0.6 * calculated_confidence

        # 为每个违规项重新计算置信度
        enhanced_violations = []
        for i, violation in enumerate(violations):
            # 找到对应的相关条文
            rule_relevance = retrieved_rules[i]['score'] if i < len(retrieved_rules) else 0.5

            # 计算置信度
            calculated_violation_conf = ConfidenceCalculator.calculate_violation_confidence(
                violation_type=violation.get('type', ''),
                clause_text=violation.get('clause_text', ''),
                retrieved_rules=retrieved_rules,
                rule_relevance=rule_relevance,
            )

            # 融合 LLM 输出和计算结果
            llm_violation_conf = violation.get('confidence', 0.5)
            final_violation_conf = 0.4 * llm_violation_conf + 0.6 * calculated_violation_conf

            enhanced_violations.append({
                **violation,
                'confidence': round(final_violation_conf, 2),
                'calculated_confidence': round(calculated_violation_conf, 2),
                'llm_confidence': round(llm_violation_conf, 2),
            })

        return {
            'overall_confidence': round(final_confidence, 2),
            'llm_confidence': round(llm_overall_confidence, 2),
            'calculated_confidence': round(calculated_confidence, 2),
            'violations': enhanced_violations,
        }
