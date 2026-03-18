#!/usr/bin/env python3
"""
意图识别与问题重写 - 可视化追踪脚本

逐行展示代码执行过程，帮助你理解每一步在做什么
"""

import sys
import os
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
os.chdir(project_root)
sys.path.insert(0, str(project_root.parent))

from aliyun_rag.enhanced_auditor import EnhancedAuditor


def print_step(step_num, title):
    print("\n" + "▶" * 35)
    print(f"  步骤 {step_num}: {title}")
    print("▶" * 35)


def print_code(code):
    print(f"\n  📝 代码:")
    for line in code.split('\n'):
        print(f"     {line}")


def print_result(label, data):
    print(f"\n  📊 {label}:")
    if isinstance(data, str):
        print(f'     "{data}"')
    elif isinstance(data, list):
        for i, item in enumerate(data, 1):
            print(f"     [{i}] {item}")
    elif isinstance(data, dict):
        for key, value in data.items():
            print(f"     {key}: {value}")
    else:
        print(f"     {data}")


def main():
    print("=" * 70)
    print("  意图识别与问题重写 - 可视化追踪")
    print("=" * 70)

    # 测试文案
    marketing_text = "本保险产品保本保收益，年化收益率保证8%，零风险，稳赚不赔！"

    print_result("输入营销文案", marketing_text)

    # ========================================================================
    # 步骤1: 意图识别
    # ========================================================================
    print_step(1, "意图识别 (identify_intent)")

    print_code("""
    # 1. 定义违规类型与关键词映射
    VIOLATION_PATTERNS = {
        "承诺保证收益": ["保证", "承诺", "保本", "零风险", "稳赚不赔"],
        "夸大宣传": ["最高", "第一", "最佳"],
        # ... 更多类型
    }

    # 2. 遍历每种违规类型，统计关键词出现次数
    for violation_type, keywords in VIOLATION_PATTERNS.items():
        for kw in keywords:
            count = marketing_text.count(kw)
            if count > 0:
                记录匹配
    """)

    # 执行意图识别
    intent_result = EnhancedAuditor.identify_intent(marketing_text)

    print_result("关键词匹配详情", intent_result['keyword_matches'])
    print_result("主要意图", intent_result['primary_intent'])
    print_result("检测到的风险", intent_result['detected_risks'])
    print_result("风险等级", intent_result['risk_level'])

    # ========================================================================
    # 步骤2: 问题重写
    # ========================================================================
    print_step(2, "问题重写 (rewrite_query)")

    print_code("""
    # 1. 添加原始查询
    queries = [marketing_text]

    # 2. 根据主要意图生成专业查询
    if primary_intent == "承诺保证收益":
        queries.extend([
            "保险产品收益保证承诺 固定收益 保本保收益",
            "保险产品风险提示 收益不确定性说明",
        ])

    # 3. 高风险时添加通用合规查询
    if risk_level in ["high", "medium"]:
        queries.extend([
            "保险销售合规要求 禁止性规定",
            "互联网保险业务监管 营销宣传规范",
        ])

    # 4. 去重返回
    return list(set(queries))
    """)

    # 执行问题重写
    rewritten_queries = EnhancedAuditor.rewrite_query(marketing_text, intent_result)

    print_result("生成的检索查询", rewritten_queries)

    # ========================================================================
    # 步骤3: 解释为什么要这样重写
    # ========================================================================
    print_step(3, "重写策略解释")

    print("""
    📌 重写策略说明:

    1️⃣  原始查询保留
        → 保留用户输入的完整语义
        → "本保险产品保本保收益，年化收益率保证8%..."

    2️⃣  专业术语转换
        → 将口语化表达转为监管标准术语
        → "稳赚不赔" → "保险产品收益保证承诺"
        → 目的: 召回监管条文中的专业表述

    3️⃣  反面查询补充
        → 查询合规要求，用于对比判断
        → "保险产品风险提示 收益不确定性说明"
        → 目的: 理解"应该怎么做"

    4️⃣  通用兜底查询
        → 召回基础监管条文
        → "保险销售合规要求 禁止性规定"
        → 目的: 确保覆盖核心禁令
    """)

    # ========================================================================
    # 步骤4: 对比演示
    # ========================================================================
    print_step(4, "效果对比")

    print("""
    ⚠️  不使用问题重写:
        输入: "这个产品稳赚不赔"
        检索: 只用原文检索
        召回: 1-2个相关条文

    ✅ 使用问题重写:
        输入: "这个产品稳赚不赔"
        检索: [
            "这个产品稳赚不赔",           # 原文
            "保险产品收益保证承诺",       # 专业术语
            "保险产品风险提示说明",       # 反面查询
            "保险销售禁止性规定"          # 通用查询
        ]
        召回: 5-10个相关条文，覆盖更全面
    """)

    # ========================================================================
    # 步骤5: 完整调用链
    # ========================================================================
    print_step(5, "完整调用链 (cli_demo.py)")

    print("""
    📂 cli_demo.py (第125-151行)
       │
       ├── marketing_text = input("请输入营销文案: ")
       │
       └── result = enhanced_audit_marketing_text(
                   marketing_text=marketing_text,
                   kb=kb,
                   client=client,
                   top_k=6,
               )

           📂 enhanced_auditor.py
              │
              └── EnhancedAuditor.multi_stage_audit()
                  │
                  ├── ① identify_intent()          ← 意图识别
                  ├── ② rewrite_query()            ← 问题重写
                  ├── ③ 多轮检索 (用重写后的查询)
                  └── ④ LLM综合分析

    📤 返回结果:
        {
            "is_compliant": "no",
            "violations": [...],
            "intent_analysis": {...},    ← 意图识别结果
            "rewritten_queries": [...],  ← 重写后的查询
            "retrieved_rules": [...]     ← 检索到的条文
        }
    """)

    # ========================================================================
    # 总结
    # ========================================================================
    print("\n" + "=" * 70)
    print("  总结")
    print("=" * 70)

    print("""
    ┌─────────────────────────────────────────────────────────────┐
    │                                                             │
    │   意图识别 = 判断"这是什么类型的违规？"                      │
    │   问题重写 = 生成"怎么检索相关条文？"                        │
    │                                                             │
    │   目的: 提高 RAG 系统的召回准确率                            │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘
    """)


if __name__ == "__main__":
    main()
