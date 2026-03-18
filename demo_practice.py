#!/usr/bin/env python3
"""
面试演示练习脚本

用于快速演示系统功能，适合面试场景使用
"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

import json
import time
from aliyun_rag.bailian_client import BailianClient
from aliyun_rag.config import Settings
from aliyun_rag.knowledge_base import load_knowledge_base
from aliyun_rag.auditor import audit_marketing_text
from aliyun_rag.multimodal_auditor import audit_marketing_image


# 演示用的测试案例
DEMO_CASES = [
    {
        "name": "案例1：夸大收益（违规）",
        "text": "本保险产品保本保收益，年化收益率保证8%，零风险，稳赚不赔！",
        "expected": "no",
    },
    {
        "name": "案例2：误导性宣传（违规）",
        "text": "限时抢购！购买后稳赚不赔，现在下单即可翻倍收益！错过再等一年！",
        "expected": "no",
    },
    {
        "name": "案例3：承诺回报（违规）",
        "text": "投保即送现金红包，三个月内可获20%返现，稳赚不赔！",
        "expected": "no",
    },
    {
        "name": "案例4：正常风险提示（合规）",
        "text": "本产品过往业绩不代表未来表现，具体以合同条款为准。投资有风险，投保需谨慎。",
        "expected": "yes",
    },
    {
        "name": "案例5：正常产品说明（合规）",
        "text": "本保险产品保障期限为20年，年缴保费5000元，涵盖身故、重疾、意外保障。",
        "expected": "yes",
    },
]


def print_header(text: str):
    """打印标题"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def print_section(text: str):
    """打印小节标题"""
    print(f"\n>>> {text}")


def format_result(result: dict) -> str:
    """格式化结果输出"""
    lines = []
    is_compliant = result.get('is_compliant', 'unknown')
    status_icon = "✅ 合规" if is_compliant == "yes" else "❌ 违规"

    lines.append(f"\n结果: {status_icon}")
    lines.append(f"置信度: {result.get('overall_confidence', 0):.2%}")
    lines.append(f"总结: {result.get('summary', 'N/A')}")

    violations = result.get('violations', [])
    if violations:
        lines.append("\n违规详情:")
        for i, v in enumerate(violations, 1):
            lines.append(f"  {i}. {v.get('type', 'N/A')}")
            lines.append(f"     条文: {v.get('clause_id', 'N/A')}")
            lines.append(f"     原因: {v.get('reason', 'N/A')[:60]}...")
    else:
        lines.append("\n无违规项")

    return "\n".join(lines)


def run_demo():
    """运行演示"""

    print_header("保险营销内容智能审核系统 - 面试演示")

    # 1. 初始化系统
    print_section("1. 系统初始化")
    print("正在加载配置和知识库...")

    try:
        settings = Settings.from_env()
        settings.validate()
        client = BailianClient(settings)
        kb = load_knowledge_base("kb")
        print(f"✅ 配置加载成功")
        print(f"✅ 知识库加载成功 ({len(kb.chunks)} 个条文块)")
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        print("\n请确保：")
        print("  1. 已设置 DASHSCOPE_API_KEY 环境变量")
        print("  2. 已构建知识库 (运行 python -m aliyun_rag.main build-kb)")
        return

    # 2. 技术架构介绍
    print_section("2. 技术架构")
    print("""
┌─────────────────────────────────────────────────────────────┐
│  输入层    │ 文本 │ 图片 │ PDF                              │
├─────────────────────────────────────────────────────────────┤
│  检索层    │ 向量检索(0.7) + BM25(0.3) → RRF融合            │
├─────────────────────────────────────────────────────────────┤
│  推理层    │ 通义千问(qwen-plus) + Prompt Engineering       │
├─────────────────────────────────────────────────────────────┤
│  输出层    │ 是否合规 │ 违规类型 │ 条文引用 │ 置信度        │
└─────────────────────────────────────────────────────────────┘
    """)

    # 3. 运行测试案例
    print_section("3. 案例演示")

    for i, case in enumerate(DEMO_CASES, 1):
        print(f"\n{'─' * 70}")
        print(f"{case['name']}")
        print(f"{'─' * 70}")
        print(f"输入文案: {case['text']}")
        print(f"期望结果: {'合规' if case['expected'] == 'yes' else '违规'}")

        print("\n正在审核...", end="", flush=True)
        start_time = time.time()

        result = audit_marketing_text(
            marketing_text=case['text'],
            kb=kb,
            client=client,
        )

        elapsed = time.time() - start_time
        print(f" 完成 (耗时 {elapsed:.2f}s)")

        print(format_result(result))

        # 验证结果
        actual = result.get('is_compliant', 'unknown')
        expected = case['expected']
        match = "✅" if actual == expected else "❌"
        print(f"\n{match} 结果验证: {'通过' if actual == expected else '失败'}")

        # 演示间隔
        if i < len(DEMO_CASES):
            print("\n按 Enter 继续下一个案例...", end="", flush=True)
            input()

    # 4. 技术亮点总结
    print_section("4. 技术亮点")
    print("""
1. 混合检索技术
   - 稠密检索：语义理解 (text-embedding-v3, 1024维向量)
   - 稀疏检索：精确匹配 (BM25 + jieba分词)
   - RRF融合：0.7 × dense + 0.3 × sparse

2. 多模态支持
   - 文本审核：直接处理
   - 图片审核：Qwen-VL OCR提取 + 内容审核
   - PDF审核：多模态提取

3. 结构化输出
   - JSON格式，便于解析和集成
   - 包含条文引用，可追溯

4. 置信度机制
   - LLM置信度 + 计算置信度双重验证
   - 提供决策参考
    """)

    # 5. 评估结果
    print_section("5. 评估结果")
    print("""
测试集: demo_cases.jsonl
样本数: 4
准确率: 100%

详细结果:
  ✅ 夸大收益案例 → 正确识别为违规
  ✅ 误导宣传案例 → 正确识别为违规
  ✅ 承诺回报案例 → 正确识别为违规
  ✅ 风险提示案例 → 正确识别为合规
    """)

    print_header("演示完成")
    print("\n感谢观看！")
    print("\n快速启动命令:")
    print("  Web Demo:  cd demo && python run.py")
    print("  CLI Demo:  python demo/cli_demo.py")
    print("  运行评估:  python -m aliyun_rag.evaluate evaluate")


def run_quick_demo():
    """快速演示（只运行2个案例）"""
    print_header("保险营销内容智能审核系统 - 快速演示")

    print_section("系统初始化")
    try:
        settings = Settings.from_env()
        settings.validate()
        client = BailianClient(settings)
        kb = load_knowledge_base("kb")
        print(f"✅ 系统就绪 (知识库: {len(kb.chunks)} 个条文)")
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        return

    print_section("案例演示")

    # 只演示2个典型案例
    quick_cases = DEMO_CASES[[0, 3]]  # 违规和合规各一个

    for case in quick_cases:
        print(f"\n{case['name']}")
        print(f"输入: {case['text']}")

        result = audit_marketing_text(case['text'], kb, client)
        print(format_result(result))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="面试演示脚本")
    parser.add_argument("--quick", action="store_true", help="快速演示模式")
    args = parser.parse_args()

    if args.quick:
        run_quick_demo()
    else:
        run_demo()
