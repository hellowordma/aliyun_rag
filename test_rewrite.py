#!/usr/bin/env python3
"""
问题重写功能测试脚本

演示从原始营销文案到问题重写，再到最终审核结果的完整流程
"""

import sys
import os
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
os.chdir(project_root)
sys.path.insert(0, str(project_root.parent))

from aliyun_rag.bailian_client import BailianClient
from aliyun_rag.config import Settings
from aliyun_rag.knowledge_base import load_knowledge_base
from aliyun_rag.enhanced_auditor import EnhancedAuditor, enhanced_audit_marketing_text


def print_header(text):
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def print_section(text):
    print(f"\n{'─' * 70}")
    print(f"  {text}")
    print('─' * 70)


def test_rewrite():
    """测试问题重写功能"""

    print_header("问题重写功能测试")

    # 加载资源
    print("[*] 正在加载资源...")
    try:
        settings = Settings.from_env()
        settings.validate()
        client = BailianClient(settings)
        kb = load_knowledge_base('kb')
        print(f"[OK] 资源加载成功！")
        print(f"[KB] 知识库: {len(kb.chunks)} 个条文块")
    except Exception as e:
        print(f"[!] 加载失败: {e}")
        print("\n请确保：")
        print("  1. 已设置 DASHSCOPE_API_KEY 环境变量")
        print("  2. 已构建知识库")
        return

    # 测试案例
    test_cases = [
        "本保险产品保本保收益，年化收益率保证8%，零风险，稳赚不赔！",
        "本保险产品行业排名第一，最专业保障，最佳服务！",
        "这款保险产品等同于银行存款，收益比银行高！",
        "知名明星强力推荐，官方认证产品！",
    ]

    print(f"\n[*] 共 {len(test_cases)} 个测试案例\n")

    for idx, marketing_text in enumerate(test_cases, 1):
        print_header(f"案例 {idx}/{len(test_cases)}")

        # 显示原始文案
        print(f"\n【原始营销文案】")
        print(f"  {marketing_text}")

        # 步骤1：意图识别
        print_section("步骤1：意图识别")
        intent_result = EnhancedAuditor.identify_intent(marketing_text)

        print(f"  主要意图: {intent_result['primary_intent']}")
        print(f"  检测到的风险: {', '.join(intent_result['detected_risks']) or '无'}")
        print(f"  风险等级: {intent_result['risk_level']}")

        if intent_result['keyword_matches']:
            print(f"\n  关键词匹配:")
            for vtype, kws in intent_result['keyword_matches'].items():
                print(f"    {vtype}: {list(kws.keys())}")

        # 步骤2：问题重写
        print_section("步骤2：问题重写")
        rewritten_queries = EnhancedAuditor.rewrite_query(marketing_text, intent_result)

        print(f"  生成 {len(rewritten_queries)} 个检索查询:")
        for i, query in enumerate(rewritten_queries, 1):
            print(f"    {i}. {query}")

        # 步骤3：使用重写后的查询进行检索
        print_section("步骤3：多轮检索（基于重写查询）")

        all_rules = []
        from aliyun_rag.knowledge_base import retrieve_relevant_rules

        for i, query in enumerate(rewritten_queries[:3], 1):
            print(f"  查询 {i}: {query[:40]}...")
            rules = retrieve_relevant_rules(query, kb, client, top_k=3)
            all_rules.extend(rules)

        # 去重
        unique_rules = EnhancedAuditor._deduplicate_rules(all_rules)
        unique_rules = sorted(unique_rules, key=lambda x: x['score'], reverse=True)[:6]

        print(f"\n  检索到 {len(unique_rules)} 个相关条文（去重后Top-6）:")
        for i, rule in enumerate(unique_rules, 1):
            print(f"    [{i}] {rule['clause_id']} | 相似度: {rule['score']:.4f}")
            print(f"        {rule['clause_text'][:60]}...")

        # 步骤4：完整审核（使用增强审核器）
        print_section("步骤4：完整审核结果")

        input("\n按 Enter 查看完整审核结果...")

        result = enhanced_audit_marketing_text(
            marketing_text=marketing_text,
            kb=kb,
            client=client,
            top_k=6,
            enable_math_confidence=True,
        )

        # 显示结果
        is_compliant = result.get('is_compliant', 'unknown')
        status_icon = "✅" if is_compliant == "yes" else "❌"
        print(f"\n  {status_icon} 是否合规: {is_compliant.upper()}")

        if 'calculated_confidence' in result:
            print(f"\n  置信度分析:")
            print(f"    LLM置信度: {result['llm_confidence']:.2%}")
            print(f"    计算置信度: {result['calculated_confidence']:.2%}")
            print(f"    融合置信度: {result['overall_confidence']:.2%}")

        print(f"\n  总结: {result.get('summary', 'N/A')}")

        violations = result.get('violations', [])
        if violations:
            print(f"\n  违规项 ({len(violations)}):")
            for i, v in enumerate(violations, 1):
                print(f"\n    [{i}] {v.get('type', 'N/A')}")
                print(f"        条文: {v.get('clause_id', 'N/A')}")
                print(f"        原因: {v.get('reason', 'N/A')[:80]}...")
                if v.get('implicit_violations'):
                    print(f"        隐含违规: {', '.join(v['implicit_violations'])}")

        if result.get('context_analysis'):
            print(f"\n  上下文分析:")
            print(f"    {result['context_analysis'][:100]}...")

        # 案例间隔
        if idx < len(test_cases):
            input(f"\n{'='*70}\n按 Enter 继续下一个案例...")

    print_header("测试完成")


if __name__ == "__main__":
    try:
        test_rewrite()
    except KeyboardInterrupt:
        print("\n\n测试已中断")
    except Exception as e:
        print(f"\n[!] 发生错误: {e}")
        import traceback
        traceback.print_exc()
