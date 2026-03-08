#!/usr/bin/env python3
"""
批量审核营销内容示例脚本

使用方法:
    PYTHONPATH=/mnt/workspace:$PYTHONPATH python batch_audit.py
"""

import json
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from aliyun_rag.auditor import audit_marketing_text
from aliyun_rag.bailian_client import BailianClient
from aliyun_rag.config import Settings
from aliyun_rag.knowledge_base import load_knowledge_base


def batch_audit_texts(
    texts_dir: str,
    output_dir: str,
    kb_dir: str = "kb",
):
    """批量审核文本文件"""

    # 加载配置和知识库
    settings = Settings.from_env()
    settings.validate()
    client = BailianClient(settings)
    kb = load_knowledge_base(kb_dir)

    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 读取所有测试文本
    texts_dir_path = Path(texts_dir)
    text_files = sorted(texts_dir_path.glob("*.txt"))

    if not text_files:
        print(f"❌ 在 {texts_dir} 中未找到 .txt 文件")
        return

    print(f"📂 找到 {len(text_files)} 个测试文件")
    print("=" * 60)

    results = []

    for i, text_file in enumerate(text_files, 1):
        print(f"\n🔍 [{i}/{len(text_files)}] 审核: {text_file.name}")

        # 读取文本内容
        marketing_text = text_file.read_text(encoding="utf-8").strip()

        # 执行审核
        result = audit_marketing_text(
            marketing_text=marketing_text,
            kb=kb,
            client=client,
            top_k=6,
        )

        # 添加元数据
        result["input_file"] = text_file.name
        result["input_text"] = marketing_text

        # 保存单个结果
        output_file = output_path / f"{text_file.stem}_result.json"
        with output_file.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        # 打印摘要
        compliant = result.get("is_compliant", "unknown")
        summary = result.get("summary", "")
        confidence = result.get("overall_confidence", 0)

        status_icon = "✅" if compliant == "yes" else "❌"
        print(f"   {status_icon} 是否合规: {compliant}")
        print(f"   📊 置信度: {confidence:.2%}")
        print(f"   📝 总结: {summary}")

        if result.get("violations"):
            print(f"   ⚠️  违规项: {len(result['violations'])} 项")
            for v in result["violations"]:
                print(f"      - {v.get('type', 'N/A')}: {v.get('reason', 'N/A')[:60]}...")

        results.append({
            "file": text_file.name,
            "is_compliant": compliant,
            "confidence": confidence,
            "summary": summary,
            "violations_count": len(result.get("violations", [])),
        })

    # 保存汇总报告
    summary_file = output_path / "audit_summary.json"
    with summary_file.open("w", encoding="utf-8") as f:
        json.dump({
            "total_files": len(text_files),
            "compliant_count": sum(1 for r in results if r["is_compliant"] == "yes"),
            "non_compliant_count": sum(1 for r in results if r["is_compliant"] == "no"),
            "results": results,
        }, f, ensure_ascii=False, indent=2)

    # 生成 Markdown 报告
    md_report = generate_markdown_report(results)
    md_file = output_path / "audit_report.md"
    md_file.write_text(md_report, encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"✅ 批量审核完成！")
    print(f"📊 汇总: {sum(1 for r in results if r['is_compliant'] == 'yes')} 合规, {sum(1 for r in results if r['is_compliant'] == 'no')} 不合规")
    print(f"📁 结果保存到: {output_path}")
    print(f"   - 单个结果: *_result.json")
    print(f"   - 汇总JSON: audit_summary.json")
    print(f"   - Markdown报告: audit_report.md")


def generate_markdown_report(results: list) -> str:
    """生成 Markdown 格式的审核报告"""

    lines = [
        "# 保险营销内容审核报告",
        "",
        f"**生成时间**: {Path(__file__).stat().st_mtime}",
        f"**审核文件数**: {len(results)}",
        "",
        "## 📊 审核汇总",
        "",
        f"| 指标 | 数量 |",
        f"|------|------|",
        f"| 总文件数 | {len(results)} |",
        f"| 合规数量 | {sum(1 for r in results if r['is_compliant'] == 'yes')} |",
        f"| 不合规数量 | {sum(1 for r in results if r['is_compliant'] == 'no')} |",
        "",
        "## 📋 详细结果",
        "",
    ]

    for r in results:
        status_icon = "✅" if r["is_compliant"] == "yes" else "❌"
        status_text = "合规" if r["is_compliant"] == "yes" else "不合规"

        lines.extend([
            f"### {status_icon} {r['file']}",
            "",
            f"- **审核结果**: {status_text}",
            f"- **置信度**: {r['confidence']:.2%}",
            f"- **违规项数量**: {r['violations_count']}",
            f"- **总结**: {r['summary']}",
            "",
        ])

    return "\n".join(lines)


if __name__ == "__main__":
    batch_audit_texts(
        texts_dir="test_data/marketing_texts",
        output_dir="test_data/audit_results",
        kb_dir="kb",
    )
