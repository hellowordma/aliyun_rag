#!/usr/bin/env python3
"""
保险审核Agent简化演示

直接使用工具进行审核，不通过ReAct循环
"""

import sys
import os
from pathlib import Path

# 添加项目路径
script_dir = Path(__file__).parent
project_root = script_dir.parent
workspace = project_root.parent

sys.path.insert(0, str(workspace))
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from aliyun_rag.bailian_client import BailianClient
from aliyun_rag.config import Settings
from aliyun_rag.knowledge_base_milvus import load_knowledge_base as load_milvus_kb
from aliyun_rag.knowledge_base import load_knowledge_base
from aliyun_rag.agent.tools import create_audit_tools


def print_header(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def main():
    print_header("保险审核Agent简化演示")

    # 加载配置
    print("[*] 加载配置和资源...")
    settings = Settings.from_env()
    settings.validate()

    client = BailianClient(settings)

    # 加载知识库
    try:
        kb = load_milvus_kb(collection_name='insurance_knowledge', meta_dir='kb_milvus')
        print(f"[OK] Milvus知识库加载成功 ({len(kb.chunks)} chunks)")
    except Exception as e:
        print(f"[!] Milvus加载失败，回退到NumPy: {e}")
        kb = load_knowledge_base('kb')
        print(f"[OK] NumPy知识库加载成功 ({len(kb.chunks)} chunks)")

    # 创建工具
    tools = create_audit_tools(kb, client)
    print(f"[OK] 工具注册完成，共 {len(tools.list_tools())} 个工具\n")

    # 测试用例
    test_cases = [
        ("违规-承诺收益", "本保险保证年化收益8%，稳赚不赔，零风险！"),
        ("违规-夸大宣传", "限时抢购！购买后稳赚不赔，现在下单即可翻倍收益！"),
        ("合规-风险提示", "本产品过往业绩不代表未来表现，具体以合同条款为准。"),
        ("合规-正常宣传", "请您仔细阅读保险条款，重点关注责任免除和犹豫期。"),
    ]

    print_header("开始批量审核")

    results = []
    for name, text in test_cases:
        print(f"\n审核中: {name}")
        print(f"文案: {text}")

        # 直接调用工具
        audit_tool = tools.get("audit_text")
        if audit_tool:
            tool_result = audit_tool.execute(marketing_text=text, use_enhanced=True)

            if tool_result.success:
                data = tool_result.data
                is_compliant = data.get("is_compliant", "unknown")
                violations = data.get("violations", [])
                confidence = data.get("overall_confidence", 0)

                print(f"  结果: [{is_compliant.upper()}] 置信度: {confidence:.2%}")
                if violations:
                    for v in violations[:2]:
                        print(f"    - {v.get('type', 'N/A')}: {v.get('reason', 'N/A')[:50]}...")

                results.append({
                    "name": name,
                    "text": text[:30] + "...",
                    "is_compliant": is_compliant,
                    "violations_count": len(violations),
                    "confidence": confidence,
                })
            else:
                print(f"  失败: {tool_result.error}")

    # 汇总
    print_header("审核汇总")
    compliant = sum(1 for r in results if r["is_compliant"] == "yes")
    non_compliant = sum(1 for r in results if r["is_compliant"] == "no")

    print(f"总测试数: {len(results)}")
    print(f"合规: {compliant}")
    print(f"违规: {non_compliant}")
    print()

    for r in results:
        icon = "[OK]" if r["is_compliant"] == "yes" else "[X]"
        print(f"{icon} {r['name']:20s} - {r['is_compliant'].upper():4s} (违规: {r['violations_count']})")

    print("\n" + "=" * 70)
    print("演示完成！")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序已退出")
