#!/usr/bin/env python3
"""
保险审核Agent使用演示

展示如何使用InsuranceAuditAgent进行各种审核任务
"""

import sys
import os
from pathlib import Path

# 添加项目路径并切换工作目录
script_dir = Path(__file__).parent
project_root = script_dir.parent  # aliyun_rag目录
workspace = project_root.parent

sys.path.insert(0, str(workspace))
sys.path.insert(0, str(project_root))
os.chdir(project_root)  # 切换到aliyun_rag目录，使相对路径正确

from aliyun_rag.bailian_client import BailianClient
from aliyun_rag.config import Settings
from aliyun_rag.knowledge_base_milvus import load_knowledge_base as load_milvus_kb
from aliyun_rag.knowledge_base import load_knowledge_base

from aliyun_rag.agent import InsuranceAuditAgent, AgentConfig
from aliyun_rag.agent.tools import create_audit_tools


def print_section(title: str):
    """打印分节标题"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def demo_text_audit(agent: InsuranceAuditAgent):
    """演示文本审核"""
    print_section("文本审核演示")

    test_cases = [
        "本保险保证年化收益8%，稳赚不赔，零风险！",
        "本产品过往业绩不代表未来表现，具体以合同条款为准。",
    ]

    for i, text in enumerate(test_cases, 1):
        print(f"【测试用例 {i}】")
        print(f"文案: {text}\n")

        result = agent.audit(text, content_type="text")

        print(f"审核结果:")
        print(f"  状态: {'成功' if result.success else '失败'}")
        print(f"  答案: {result.answer[:200]}...")

        if result.audit_result:
            print(f"  合规: {result.audit_result.get('is_compliant', 'unknown').upper()}")
            if 'violations' in result.audit_result:
                print(f"  违规项: {len(result.audit_result['violations'])}个")

        print(f"  执行时间: {result.execution_time:.2f}秒")
        print(f"  执行步骤: {len(result.steps)}步\n")


def demo_image_audit(agent: InsuranceAuditAgent):
    """演示图片审核"""
    print_section("图片审核演示")

    # 检查测试图片是否存在
    test_images = [
        "test_data_20260313/夸大收益.png",
        "test_data_20260313/合规宣传.png",
    ]

    for image_path in test_images:
        path = Path(image_path)
        if not path.exists():
            print(f"跳过不存在的图片: {image_path}")
            continue

        print(f"【测试图片】{path.name}")

        result = agent.audit(path, content_type="image")

        print(f"审核结果:")
        print(f"  状态: {'成功' if result.success else '失败'}")
        print(f"  答案: {result.answer[:200]}...")

        if result.audit_result:
            print(f"  合规: {result.audit_result.get('is_compliant', 'unknown').upper()}")

        print(f"  执行时间: {result.execution_time:.2f}秒\n")


def demo_chat_mode(agent: InsuranceAuditAgent):
    """演示对话模式"""
    print_section("对话模式演示")

    conversations = [
        "帮我审核这条文案：本保险保证年化收益8%",
        "违规了怎么办？有什么改进建议吗？",
        "给我解释一下什么是夸大收益",
    ]

    for msg in conversations:
        print(f"用户: {msg}")
        response = agent.chat(msg)
        print(f"Agent: {response[:200]}...\n")


def demo_trace_export(agent: InsuranceAuditAgent):
    """演示轨迹导出"""
    print_section("执行轨迹导出演示")

    result = agent.audit("本保险产品保本保收益，年化收益率保证10%", content_type="text")

    # 导出轨迹
    trace_file = Path("agent_trace.txt")
    agent.export_trace(result, str(trace_file))

    print(f"执行轨迹已导出到: {trace_file}")
    print(f"\n轨迹内容预览（前500字）:")
    print("-" * 70)
    trace = result.to_trace()
    print(trace[:500] + "...\n")


def demo_fallback(agent: InsuranceAuditAgent):
    """演示Fallback机制"""
    print_section("Fallback机制演示")

    print("测试场景: 工具执行失败时的处理")
    print("(尝试审核不存在的文件)\n")

    result = agent.audit("/nonexistent/file.pdf", content_type="pdf")

    print(f"结果: {'成功' if result.success else '失败（已触发Fallback）'}")
    print(f"错误信息: {result.error or '无'}")
    print(f"答案: {result.answer[:200]}\n")


def demo_batch_audit(agent: InsuranceAuditAgent):
    """演示批量审核"""
    print_section("批量审核演示")

    texts = [
        "保证年化收益8%",
        "过往业绩不代表未来表现",
        "限时抢购，最后机会",
        "具体以合同条款为准",
    ]

    print(f"批量审核 {len(texts)} 条文案...\n")

    # 通过工具调用批量审核
    tool = agent.tools.get("batch_audit")
    if tool:
        result = tool.execute(texts=texts)

        if result.success:
            results = result.data
            print(f"审核完成，共 {len(results)} 条结果:\n")
            for i, r in enumerate(results, 1):
                if isinstance(r, dict):
                    is_compliant = r.get('is_compliant', 'unknown')
                    original = r.get('original_text', 'N/A')
                    print(f"  {i}. [{is_compliant.upper()}] {original}")
                else:
                    print(f"  {i}. {r}")
        else:
            print(f"批量审核失败: {result.error}")
    print()


def demo_stats(agent: InsuranceAuditAgent):
    """演示统计信息"""
    print_section("Agent统计信息")

    stats = agent.get_stats()

    print(f"总查询次数: {stats['total_queries']}")
    print(f"成功审核: {stats['successful_audits']}")
    print(f"失败审核: {stats['failed_audits']}")
    print(f"平均步骤数: {stats['avg_steps']:.1f}")
    print()


def interactive_demo(agent: InsuranceAuditAgent):
    """交互式Demo"""
    print_section("交互式Agent演示")

    print("可用命令:")
    print("  1. 文本审核")
    print("  2. 图片审核")
    print("  3. PDF审核")
    print("  4. 对话模式")
    print("  5. 查看统计")
    print("  0. 退出")
    print()

    while True:
        choice = input("请选择操作 (0-5): ").strip()

        if choice == "0":
            print("\n再见！")
            break

        elif choice == "1":
            text = input("请输入营销文案: ").strip()
            if text:
                result = agent.audit(text, content_type="text")
                print(f"\n{result.answer}\n")

        elif choice == "2":
            path = input("请输入图片路径: ").strip()
            if path:
                result = agent.audit(path, content_type="image")
                print(f"\n{result.answer}\n")

        elif choice == "3":
            path = input("请输入PDF路径: ").strip()
            if path:
                result = agent.audit(path, content_type="pdf")
                print(f"\n{result.answer}\n")

        elif choice == "4":
            msg = input("请输入消息: ").strip()
            if msg:
                response = agent.chat(msg)
                print(f"\nAgent: {response}\n")

        elif choice == "5":
            stats = agent.get_stats()
            print(f"\n统计: {stats}\n")

        else:
            print("\n无效选项，请重试\n")


def main():
    """主函数"""
    print_section("保险审核Agent演示")

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
    print(f"[OK] 工具注册完成，共 {len(tools.list_tools())} 个工具")

    # 创建Agent
    config = AgentConfig(
        max_steps=8,
        max_retries=2,
        verbose=True,
    )
    agent = InsuranceAuditAgent(client, tools, config, kb=kb)
    print("[OK] Agent初始化完成\n")

    # 运行演示
    print("选择演示模式:")
    print("  1. 文本审核演示")
    print("  2. 图片审核演示")
    print("  3. 对话模式演示")
    print("  4. 轨迹导出演示")
    print("  5. Fallback机制演示")
    print("  6. 批量审核演示")
    print("  7. 全部演示")
    print("  8. 交互式Demo")
    print("  0. 退出")
    print()

    choice = input("请选择 (0-8): ").strip()

    if choice == "1":
        demo_text_audit(agent)
    elif choice == "2":
        demo_image_audit(agent)
    elif choice == "3":
        demo_chat_mode(agent)
    elif choice == "4":
        demo_trace_export(agent)
    elif choice == "5":
        demo_fallback(agent)
    elif choice == "6":
        demo_batch_audit(agent)
    elif choice == "7":
        demo_text_audit(agent)
        demo_image_audit(agent)
        demo_chat_mode(agent)
        demo_trace_export(agent)
        demo_batch_audit(agent)
    elif choice == "8":
        interactive_demo(agent)
    else:
        print("退出演示")

    # 显示最终统计
    demo_stats(agent)


if __name__ == "__main__":
    import sys

    # 支持命令行参数快速测试
    # 用法: python demo.py --quick 或 python demo.py --test
    if len(sys.argv) > 1 and sys.argv[1] in ["--quick", "--test"]:
        print_section("快速测试模式")

        # 加载配置
        print("[*] 加载配置和资源...")
        settings = Settings.from_env()
        settings.validate()
        client = BailianClient(settings)

        # 加载Milvus知识库
        try:
            kb = load_milvus_kb(collection_name='insurance_knowledge', meta_dir='kb_milvus')
            print(f"[OK] Milvus知识库加载成功 ({len(kb.chunks)} chunks)")
        except Exception as e:
            print(f"[!] Milvus加载失败，回退到NumPy: {e}")
            kb = load_knowledge_base('kb')
            print(f"[OK] NumPy知识库加载成功 ({len(kb.chunks)} chunks)")

        # 创建工具和Agent
        tools = create_audit_tools(kb, client)
        config = AgentConfig(max_steps=5, max_retries=2, verbose=False)
        agent = InsuranceAuditAgent(client, tools, config, kb=kb)
        print("[OK] Agent初始化完成\n")

        # 快速测试
        print("[*] 执行快速测试...\n")
        test_text = "本保险保证年化收益8%，稳赚不赔！"

        # 使用verbose=True查看详细过程
        config = AgentConfig(max_steps=8, max_retries=2, verbose=True)
        agent = InsuranceAuditAgent(client, tools, config, kb=kb)

        result = agent.audit(test_text, content_type="text")

        print("\n" + "=" * 70)
        print("快速测试结果")
        print("=" * 70)
        print(f"测试文案: {test_text}")
        print(f"执行成功: {result.success}")
        if result.error:
            print(f"错误信息: {result.error}")
        print(f"\nAgent回复:\n{result.answer[:300]}...")
        if result.audit_result:
            print(f"\n合规状态: {result.audit_result.get('is_compliant', 'unknown').upper()}")
            if result.audit_result.get('violations'):
                print(f"违规项:")
                for v in result.audit_result['violations'][:2]:
                    print(f"  - {v.get('type', 'N/A')}: {v.get('reason', 'N/A')[:60]}...")
        print(f"\n执行时间: {result.execution_time:.2f}秒")
        print(f"执行步骤: {len(result.steps)}步")
        print("=" * 70)
        sys.exit(0)

    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序已退出")
