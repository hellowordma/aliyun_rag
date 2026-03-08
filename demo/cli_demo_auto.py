#!/usr/bin/env python3
"""
保险营销内容智能审核系统 - 交互式命令行Demo（自动修复版）

支持文本、图片、图文混合审核
"""

import sys
import os
from pathlib import Path

# 自动添加项目路径
script_dir = Path(__file__).parent
project_root = script_dir.parent
workspace = project_root.parent
sys.path.insert(0, str(workspace))

# 切换到项目目录
os.chdir(project_root)

import json

from aliyun_rag.bailian_client import BailianClient
from aliyun_rag.config import Settings
from aliyun_rag.knowledge_base import load_knowledge_base
from aliyun_rag.auditor import audit_marketing_text
from aliyun_rag.enhanced_auditor import enhanced_audit_marketing_text
from aliyun_rag.multimodal_auditor import audit_marketing_image, audit_marketing_multimodal


class DemoLauncher:
    """Demo 启动器"""

    def __init__(self):
        self.client = None
        self.kb = None
        self.is_loaded = False

    def load_resources(self):
        """加载资源"""
        if self.is_loaded:
            return True

        try:
            print("[*] 正在加载资源...")
            settings = Settings.from_env()
            settings.validate()
            self.client = BailianClient(settings)
            self.kb = load_knowledge_base('kb')
            self.is_loaded = True
            print("[OK] 资源加载成功！")
            print(f"[KB] 知识库: {len(self.kb.chunks)} 个法规 chunks")
            print(f"[VL] 多模态模型: {settings.vl_model}")
            return True
        except Exception as e:
            print(f"[!] 加载失败: {str(e)}")
            print("[*] 尝试修复...")
            return self.fix_and_load()

    def fix_and_load(self):
        """自动修复并加载"""
        try:
            # 检查当前目录
            cwd = Path.cwd()
            print(f"[DIR] 当前目录: {cwd}")

            # 查找知识库文件
            kb_path = cwd / "kb"
            if kb_path.exists():
                print(f"[OK] 找到知识库: {kb_path}")
            else:
                print(f"[!] 知识库不存在: {kb_path}")
                print("[*] 尝试重建知识库...")

                # 重建知识库
                from aliyun_rag.main import cmd_build_kb
                import argparse

                docs = [
                    str(cwd / "references/保险销售行为管理办法.pdf"),
                    str(cwd / "references/互联网保险业务监管办法.docx"),
                ]

                # 检查文件是否存在
                existing_docs = [d for d in docs if Path(d).exists()]
                if not existing_docs:
                    print("[!] 监管文档文件不存在！")
                    return False

                # 构建知识库
                args = argparse.Namespace(
                    docs=existing_docs,
                    kb_dir="kb",
                    pdf_mode="vl",
                    vector_db="numpy",
                )

                cmd_build_kb(args)
                return True

            # 加载配置
            settings = Settings.from_env()
            settings.validate()
            self.client = BailianClient(settings)
            self.kb = load_knowledge_base('kb')
            self.is_loaded = True
            print("[OK] 资源加载成功！")
            print(f"[KB] 知识库: {len(self.kb.chunks)} 个法规 chunks")
            return True

        except Exception as e:
            print(f"[!] 修复失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def show_welcome(self):
        """显示欢迎信息"""
        print("\n" + "=" * 70)
        print("保险营销内容智能审核系统 - 多模态版")
        print("=" * 70 + "\n")
        print("[功能特点]")
        print("  [OK] 文本审核")
        print("  [OK] 图片审核（使用Qwen-VL多模态模型）")
        print("  [OK] 图文混合审核")
        print("  [OK] 意图识别与问题重写")
        print("  [OK] 混合检索（BM25 + 向量）")
        print("  [OK] 数学公式置信度计算")
        print("  [OK] 多维度合规分析")
        print()

    def show_menu(self):
        """显示主菜单"""
        print("请选择操作：")
        print("  1. [TXT] 文本审核")
        print("  2. [IMG] 图片审核")
        print("  3. [MIX] 图文混合审核")
        print("  4. [EX] 示例文案测试")
        print("  5. [BAT] 批量测试")
        print("  6. [KB] 查看知识库统计")
        print("  0. [EXIT] 退出")
        print()

    def text_audit(self):
        """文本审核"""
        print("\n[TXT] 文本审核")
        print("-" * 50)

        marketing_text = input("请输入营销文案（输入 'q' 返回）: ").strip()

        if marketing_text.lower() == 'q':
            return

        if not marketing_text:
            print("[!] 输入不能为空")
            return

        # 选择模式
        print("\n选择审核模式：")
        print("  1. 基础模式")
        print("  2. 增强模式（意图识别 + 问题重写 + 数学置信度）")
        mode_choice = input("请选择 (1/2, 默认1): ").strip() or "1"
        use_enhanced = mode_choice == "2"

        print(f"\n[*] 开始审核...")
        print(f"   模式: {'增强模式' if use_enhanced else '基础模式'}")
        print()

        try:
            if use_enhanced:
                result = enhanced_audit_marketing_text(
                    marketing_text=marketing_text,
                    kb=self.kb,
                    client=self.client,
                    top_k=6,
                    enable_math_confidence=True,
                )
            else:
                result = audit_marketing_text(
                    marketing_text=marketing_text,
                    kb=self.kb,
                    client=self.client,
                    top_k=6,
                )

            self.display_result(result, use_enhanced)

        except Exception as e:
            print(f"\n[!] 审核失败: {str(e)}")
            import traceback
            traceback.print_exc()

    def image_audit(self):
        """图片审核"""
        print("\n[IMG] 图片审核")
        print("-" * 50)

        print("支持的图片格式: png, jpg, jpeg")
        image_path = input("请输入图片路径（输入 'q' 返回）: ").strip()

        if image_path.lower() == 'q':
            return

        if not image_path:
            print("[!] 输入不能为空")
            return

        # 检查文件是否存在
        path = Path(image_path)
        if not path.exists():
            print(f"[!] 文件不存在: {image_path}")
            return

        # 判断图片类型
        ext = path.suffix.lower()
        mime_map = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
        }
        if ext not in mime_map:
            print(f"[!] 不支持的图片格式: {ext}")
            return

        image_mime = mime_map[ext]

        print(f"\n[*] 正在读取图片: {image_path}")
        print(f"[*] 开始审核...")

        try:
            with open(image_path, 'rb') as f:
                image_bytes = f.read()

            result = audit_marketing_image(
                image_bytes=image_bytes,
                kb=self.kb,
                client=self.client,
                image_mime=image_mime,
                top_k=6,
            )
            self.display_image_result(result)

        except Exception as e:
            print(f"\n[!] 审核失败: {str(e)}")
            import traceback
            traceback.print_exc()

    def multimodal_audit(self):
        """图文混合审核"""
        print("\n[MIX] 图文混合审核")
        print("-" * 50)

        print("请同时提供图片和文字说明")
        image_path = input("请输入图片路径（输入 'q' 返回）: ").strip()

        if image_path.lower() == 'q':
            return

        if not image_path:
            print("[!] 图片路径不能为空")
            return

        # 检查文件是否存在
        path = Path(image_path)
        if not path.exists():
            print(f"[!] 文件不存在: {image_path}")
            return

        # 判断图片类型
        ext = path.suffix.lower()
        mime_map = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
        }
        if ext not in mime_map:
            print(f"[!] 不支持的图片格式: {ext}")
            return

        image_mime = mime_map[ext]

        # 输入文字说明
        text_context = input("请输入文字说明（可选，直接回车跳过）: ").strip()

        print(f"\n[*] 正在读取图片: {image_path}")
        print(f"[*] 开始审核...")

        try:
            with open(image_path, 'rb') as f:
                image_bytes = f.read()

            result = audit_marketing_image(
                image_bytes=image_bytes,
                kb=self.kb,
                client=self.client,
                image_mime=image_mime,
                text_context=text_context if text_context else None,
                top_k=6,
            )
            self.display_image_result(result)

        except Exception as e:
            print(f"\n[!] 审核失败: {str(e)}")
            import traceback
            traceback.print_exc()

    def example_test(self):
        """示例文案测试"""
        print("\n[EX] 示例文案测试")
        print("-" * 50)

        examples = [
            ("1", "违规-夸大收益", "本保险产品保本保收益，年化收益率保证8%，零风险，稳赚不赔！"),
            ("2", "违规-误导宣传", "限时抢购！购买后稳赚不赔，现在下单即可翻倍收益！"),
            ("3", "合规-风险提示", "本产品过往业绩不代表未来表现，具体以合同条款为准。"),
            ("4", "合规-正常宣传", "请您仔细阅读保险条款，重点关注责任免除和犹豫期。"),
        ]

        print("\n选择示例：")
        for num, name, text in examples:
            print(f"  [{num}] {name}")
            print(f"      {text[:60]}...")

        choice = input("\n请选择示例 (1-4): ").strip()

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(examples):
                num, name, text = examples[idx]
                print(f"\n[TXT] 测试文案: {name}")
                print(f"完整内容: {text}\n")

                result = audit_marketing_text(text, self.kb, self.client, top_k=6)
                self.display_result(result, False)
            else:
                print("\n[!] 无效选项")
        except ValueError:
            print("\n[!] 请输入数字")
        except Exception as e:
            print(f"\n[!] 测试失败: {str(e)}")

    def batch_test(self):
        """批量测试"""
        print("\n[BAT] 批量测试")
        print("-" * 50)

        test_cases = [
            "本保险保证年化收益8%，稳赚不赔。",
            "限时抢购，购买后稳赚不赔，现在下单即可翻倍。",
            "本产品过往业绩不代表未来表现，具体以合同条款为准。",
            "请您仔细阅读保险条款，重点关注责任免除和犹豫期。",
        ]

        print(f"\n正在测试 {len(test_cases)} 个文案...\n")

        results = []
        for i, text in enumerate(test_cases, 1):
            print(f"[{i}/{len(test_cases)}] 审核中...")
            try:
                result = audit_marketing_text(text, self.kb, self.client, top_k=6)
                results.append({
                    'text': text[:50] + '...',
                    'is_compliant': result['is_compliant'],
                    'confidence': result['overall_confidence'],
                })
                print(f"  [OK] 完成 - {result['is_compliant']} ({result['overall_confidence']:.2%})")
            except Exception as e:
                print(f"  [!] 失败: {str(e)}")

        # 显示结果
        print("\n" + "=" * 70)
        print("测试结果汇总")
        print("=" * 70 + "\n")

        for i, r in enumerate(results, 1):
            icon = "[OK]" if r['is_compliant'] == "yes" else "[X]"
            print(f"{i}. {icon} {r['text']}")
            print(f"   是否合规: {r['is_compliant']}")
            print(f"   置信度: {r['confidence']:.2%}")
            print()

        # 统计
        compliant_count = sum(1 for r in results if r['is_compliant'] == 'yes')
        print(f"[STAT] 统计:")
        print(f"   - 总测试数: {len(results)}")
        print(f"   - 合规数量: {compliant_count}")
        print(f"   - 不合规数量: {len(results) - compliant_count}")

    def show_kb_stats(self):
        """显示知识库统计"""
        print("\n[KB] 知识库统计")
        print("-" * 50)

        print(f"总 Chunks: {len(self.kb.chunks)}")

        # 来源文件统计
        source_files = {}
        for chunk in self.kb.chunks:
            source = chunk.source_file
            source_files[source] = source_files.get(source, 0) + 1

        print("\n[FILE] 来源文件:")
        for source, count in source_files.items():
            print(f"  - {source}: {count} chunks")

        # Chunk 长度统计
        lengths = [len(c.clause_text) for c in self.kb.chunks]
        print(f"\n[LEN] Chunk 长度:")
        print(f"  - 最短: {min(lengths)} 字符")
        print(f"  - 最长: {max(lengths)} 字符")
        print(f"  - 平均: {sum(lengths) / len(lengths):.0f} 字符")

    def display_result(self, result, use_enhanced):
        """显示审核结果"""
        print("\n" + "=" * 70)
        print("审核结果")
        print("=" * 70 + "\n")

        # 是否合规
        is_compliant = result.get('is_compliant', 'unknown')
        status_icon = "[OK]" if is_compliant == "yes" else "[X]"
        print(f"{status_icon} 是否合规: {is_compliant.upper()}")

        # 置信度
        if 'calculated_confidence' in result:
            print(f"\n[CONF] 置信度分析:")
            print(f"   - LLM置信度: {result['llm_confidence']:.2%}")
            print(f"   - 计算置信度: {result['calculated_confidence']:.2%}")
            print(f"   - 融合置信度: {result['overall_confidence']:.2%}")
        else:
            print(f"\n[CONF] 整体置信度: {result.get('overall_confidence', 0):.2%}")

        # 总结
        print(f"\n[SUM] 总结: {result.get('summary', 'N/A')}")

        # 意图分析
        if use_enhanced and 'intent_analysis' in result:
            intent = result['intent_analysis']
            print(f"\n[INTENT] 意图分析:")
            print(f"   - 主要意图: {intent.get('primary_intent', 'N/A')}")
            print(f"   - 风险等级: {intent.get('risk_level', 'N/A')}")
            print(f"   - 检测到的风险: {', '.join(intent.get('detected_risks', []))}")

        # 违规项
        violations = result.get('violations', [])
        if violations:
            print(f"\n[!] 违规项 ({len(violations)}):")
            for i, v in enumerate(violations, 1):
                print(f"\n  [{i}] {v.get('type', 'N/A')}")
                print(f"      条文: {v.get('clause_id', 'N/A')}")
                if 'llm_confidence' in v:
                    print(f"      置信度: LLM={v['llm_confidence']:.2%}, 计算={v['calculated_confidence']:.2%}")
                else:
                    print(f"      置信度: {v.get('confidence', 0):.2%}")
                print(f"      原因: {v.get('reason', 'N/A')[:80]}...")

                if v.get('implicit_violations'):
                    print(f"      隐含违规: {', '.join(v['implicit_violations'])}")
        else:
            print("\n[OK] 无违规项")

        # 检索到的条文
        print(f"\n[KB] 检索到的相关条文 (Top-3):")
        rules = result.get('retrieved_rules', [])[:3]
        for i, rule in enumerate(rules, 1):
            print(f"\n  [{i}] {rule.get('clause_id', 'N/A')} | 相似度: {rule.get('score', 0):.4f}")
            print(f"      来源: {rule.get('source_file', 'N/A')}")
            print(f"      条文: {rule.get('clause_text', 'N/A')[:100]}...")

        print("\n" + "=" * 70)

    def display_image_result(self, result):
        """显示图片审核结果"""
        print("\n" + "=" * 70)
        print("图片审核结果")
        print("=" * 70 + "\n")

        # 图片分析结果
        if 'image_analysis' in result:
            img_analysis = result['image_analysis']
            print("[IMG] 图片分析:")

            if 'extracted_text' in img_analysis:
                extracted = img_analysis['extracted_text']
                if extracted:
                    print(f"  提取文字: {extracted[:200]}{'...' if len(extracted) > 200 else ''}")

            if 'visual_elements' in img_analysis and img_analysis['visual_elements']:
                print(f"  视觉元素: {', '.join(img_analysis['visual_elements'])}")

            if 'detected_issues' in img_analysis and img_analysis['detected_issues']:
                print(f"  检测问题: {', '.join(img_analysis['detected_issues'])}")

            print()

        # 是否合规
        is_compliant = result.get('is_compliant', 'unknown')
        status_icon = "[OK]" if is_compliant == "yes" else "[X]"
        print(f"{status_icon} 是否合规: {is_compliant.upper()}")

        # 置信度
        print(f"\n[CONF] 整体置信度: {result.get('overall_confidence', 0):.2%}")

        # 总结
        print(f"\n[SUM] 总结: {result.get('summary', 'N/A')}")

        # 违规项
        violations = result.get('violations', [])
        if violations:
            print(f"\n[!] 违规项 ({len(violations)}):")
            for i, v in enumerate(violations, 1):
                print(f"\n  [{i}] {v.get('type', 'N/A')}")
                print(f"      条文: {v.get('clause_id', 'N/A')}")
                print(f"      置信度: {v.get('confidence', 0):.2%}")
                print(f"      原因: {v.get('reason', 'N/A')[:100]}...")
        else:
            print("\n[OK] 无违规项")

        # 检索到的条文
        print(f"\n[KB] 检索到的相关条文 (Top-3):")
        rules = result.get('retrieved_rules', [])[:3]
        for i, rule in enumerate(rules, 1):
            print(f"\n  [{i}] {rule.get('clause_id', 'N/A')} | 相似度: {rule.get('score', 0):.4f}")
            print(f"      来源: {rule.get('source_file', 'N/A')}")
            print(f"      条文: {rule.get('clause_text', 'N/A')[:100]}...")

        print("\n" + "=" * 70)

    def run(self):
        """运行 Demo"""
        self.show_welcome()

        # 加载资源
        if not self.load_resources():
            print("\n[!] 无法加载资源，请检查配置和知识库")
            return

        # 主循环
        while True:
            self.show_menu()
            choice = input("请输入选项 (0-6): ").strip()

            if choice == "0":
                print("\n再见！")
                break
            elif choice == "1":
                self.text_audit()
            elif choice == "2":
                self.image_audit()
            elif choice == "3":
                self.multimodal_audit()
            elif choice == "4":
                self.example_test()
            elif choice == "5":
                self.batch_test()
            elif choice == "6":
                self.show_kb_stats()
            else:
                print("\n[!] 无效选项，请重新输入")


if __name__ == "__main__":
    try:
        launcher = DemoLauncher()
        launcher.run()
    except KeyboardInterrupt:
        print("\n\n程序已退出")
    except Exception as e:
        print(f"\n[!] 程序错误: {str(e)}")
        import traceback
        traceback.print_exc()
