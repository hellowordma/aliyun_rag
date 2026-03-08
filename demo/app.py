"""
保险营销内容智能审核系统 - Gradio Web Demo

一个友好的 Web 界面，用于演示保险营销内容审核功能。
"""

import gradio as gr
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from aliyun_rag.bailian_client import BailianClient
from aliyun_rag.config import Settings
from aliyun_rag.knowledge_base import load_knowledge_base, KnowledgeBase
from aliyun_rag.hybrid_retriever import create_hybrid_retriever, hybrid_retrieve_rules
from aliyun_rag.auditor import audit_marketing_text
from aliyun_rag.enhanced_auditor import enhanced_audit_marketing_text
from aliyun_rag.multimodal_auditor import audit_marketing_image
from aliyun_rag.extractors import extract_text_from_file


class AuditDemo:
    """审核 Demo 类"""

    def __init__(self, kb_dir: str = "kb", use_milvus: bool = False):
        """初始化 Demo"""
        self.kb_dir = kb_dir
        self.use_milvus = use_milvus
        self.client = None
        self.kb = None
        self.hybrid_retriever = None

    def load_resources(self):
        """加载资源"""
        try:
            # 加载配置
            settings = Settings.from_env()
            settings.validate()
            self.client = BailianClient(settings)

            # 加载知识库
            self.kb = load_knowledge_base(self.kb_dir)

            # 创建混合检索器
            self.hybrid_retriever = create_hybrid_retriever(self.kb_dir)

            return "✅ 资源加载成功！"
        except Exception as e:
            return f"❌ 加载失败: {str(e)}"

    def audit_text(
        self,
        marketing_text: str,
        mode: str,
        top_k: int,
        use_hybrid: bool,
    ):
        """审核文本"""
        if not self.client or not self.kb:
            return "❌ 请先加载资源", None, None, None

        if not marketing_text.strip():
            return "❌ 请输入营销文案", None, None, None

        try:
            # 选择审核模式
            if mode == "增强模式" and use_hybrid:
                # 增强模式 + 混合检索
                result = enhanced_audit_marketing_text(
                    marketing_text=marketing_text,
                    kb=self.kb,
                    client=self.client,
                    top_k=top_k,
                    enable_math_confidence=True,
                )
            elif mode == "增强模式":
                # 增强模式
                result = enhanced_audit_marketing_text(
                    marketing_text=marketing_text,
                    kb=self.kb,
                    client=self.client,
                    top_k=top_k,
                    enable_math_confidence=False,
                )
            else:
                # 基础模式
                if use_hybrid and self.hybrid_retriever:
                    # 基础模式 + 混合检索
                    from aliyun_rag.auditor import audit_marketing_text
                    from aliyun_rag.knowledge_base import retrieve_relevant_rules

                    # 使用混合检索
                    rules = hybrid_retrieve_rules(
                        query=marketing_text,
                        kb=self.kb,
                        client=self.client,
                        retriever=self.hybrid_retriever,
                        top_k=top_k,
                    )

                    # 构建结果
                    from aliyun_rag.auditor import _build_rule_context, _extract_json_block
                    rule_context = _build_rule_context(rules)

                    system_prompt = "你是金融保险营销合规审核助手。你只能依据给定监管条文进行判断，不得编造条文。输出必须是严格JSON，不要输出Markdown。"

                    user_prompt = f"""请审核以下营销内容是否合规。

营销内容：
{marketing_text}

可参考监管条文：
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
3. 必须引用上方出现的条文编号与原文。"""

                    raw = self.client.chat(
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=0.0,
                    )

                    result = _extract_json_block(raw)
                    result["retrieved_rules"] = rules
                else:
                    # 基础模式
                    result = audit_marketing_text(
                        marketing_text=marketing_text,
                        kb=self.kb,
                        client=self.client,
                        top_k=top_k,
                    )

            # 格式化输出
            output = self._format_result(result, mode, use_hybrid)
            return "✅ 审核完成", output, self._get_violations_table(result), self._get_retrieved_rules_table(result)

        except Exception as e:
            import traceback
            error_msg = f"❌ 审核失败: {str(e)}\n\n{traceback.format_exc()}"
            return error_msg, None, None, None

    def audit_file(
        self,
        file_path: str,
        mode: str,
        top_k: int,
        use_hybrid: bool,
    ):
        """审核文件"""
        if not self.client or not self.kb:
            return "❌ 请先加载资源", None, None, None

        if not file_path:
            return "❌ 请先上传文件", None, None, None

        try:
            # 提取文本
            extracted_text = extract_text_from_file(
                file_path=file_path,
                client=self.client,
                pdf_mode="vl",
            )

            # 审核
            return self.audit_text(extracted_text, mode, top_k, use_hybrid)

        except Exception as e:
            import traceback
            error_msg = f"❌ 文件处理失败: {str(e)}\n\n{traceback.format_exc()}"
            return error_msg, None, None, None

    def audit_image(
        self,
        image_path: str,
        text_context: str,
        mode: str,
        top_k: int,
    ):
        """审核图片"""
        if not self.client or not self.kb:
            return "❌ 请先加载资源", None, None, None

        if not image_path:
            return "❌ 请先上传图片", None, None, None

        try:
            # 判断图片类型
            from pathlib import Path
            ext = Path(image_path).suffix.lower()
            mime_map = {
                '.png': 'image/png',
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
            }
            if ext not in mime_map:
                return f"❌ 不支持的图片格式: {ext}", None, None, None

            image_mime = mime_map[ext]

            # 读取图片
            with open(image_path, 'rb') as f:
                image_bytes = f.read()

            # 调用多模态审核
            result = audit_marketing_image(
                image_bytes=image_bytes,
                kb=self.kb,
                client=self.client,
                image_mime=image_mime,
                text_context=text_context if text_context else None,
                top_k=top_k,
            )

            # 格式化输出
            output = self._format_image_result(result, text_context)
            return "✅ 审核完成", output, self._get_violations_table(result), self._get_retrieved_rules_table(result)

        except Exception as e:
            import traceback
            error_msg = f"❌ 审核失败: {str(e)}\n\n{traceback.format_exc()}"
            return error_msg, None, None, None

    def _format_result(self, result: dict, mode: str, use_hybrid: bool) -> str:
        """格式化结果"""
        lines = []
        lines.append(f"## 审核结果 ({mode})")
        lines.append(f"{'混合检索' if use_hybrid else '向量检索'} | {mode}")
        lines.append("")

        # 是否合规
        is_compliant = result.get('is_compliant', 'unknown')
        status_icon = "✅" if is_compliant == "yes" else "❌"
        lines.append(f"### {status_icon} 是否合规: {is_compliant.upper()}")
        lines.append("")

        # 置信度
        if 'calculated_confidence' in result:
            lines.append(f"### 📊 置信度分析")
            lines.append(f"- LLM置信度: {result['llm_confidence']:.2%}")
            lines.append(f"- 计算置信度: {result['calculated_confidence']:.2%}")
            lines.append(f"- 融合置信度: {result['overall_confidence']:.2%}")
        else:
            lines.append(f"### 📊 整体置信度: {result.get('overall_confidence', 0):.2%}")
        lines.append("")

        # 意图分析
        if 'intent_analysis' in result:
            intent = result['intent_analysis']
            lines.append(f"### 🎯 意图分析")
            lines.append(f"- 主要意图: {intent.get('primary_intent', 'N/A')}")
            lines.append(f"- 检测到的风险: {', '.join(intent.get('detected_risks', []))}")
            lines.append(f"- 风险等级: {intent.get('risk_level', 'N/A')}")
            lines.append("")

        # 总结
        lines.append(f"### 📝 总结")
        lines.append(result.get('summary', 'N/A'))
        lines.append("")

        # 上下文分析
        if result.get('context_analysis'):
            lines.append(f"### 🔗 上下文分析")
            lines.append(result['context_analysis'][:500])
            lines.append("")

        return "\n".join(lines)

    def _format_image_result(self, result: dict, text_context: str = "") -> str:
        """格式化图片审核结果"""
        lines = []
        lines.append(f"## 图片审核结果")
        lines.append("")

        # 图片分析结果
        if 'image_analysis' in result:
            img_analysis = result['image_analysis']
            lines.append(f"### 📷 图片分析")

            if 'extracted_text' in img_analysis:
                extracted = img_analysis['extracted_text']
                if extracted:
                    lines.append(f"**提取文字:** {extracted[:300]}{'...' if len(extracted) > 300 else ''}")

            if 'visual_elements' in img_analysis and img_analysis['visual_elements']:
                lines.append(f"**视觉元素:** {', '.join(img_analysis['visual_elements'])}")

            if text_context:
                lines.append(f"**文字说明:** {text_context}")

            lines.append("")

        # 是否合规
        is_compliant = result.get('is_compliant', 'unknown')
        status_icon = "✅" if is_compliant == "yes" else "❌"
        lines.append(f"### {status_icon} 是否合规: {is_compliant.upper()}")
        lines.append("")

        # 置信度
        lines.append(f"### 📊 整体置信度: {result.get('overall_confidence', 0):.2%}")
        lines.append("")

        # 总结
        lines.append(f"### 📝 总结")
        lines.append(result.get('summary', 'N/A'))
        lines.append("")

        return "\n".join(lines)

    def _get_violations_table(self, result: dict) -> list:
        """获取违规项表格"""
        violations = result.get('violations', [])
        if not violations:
            return [["无违规项"]]

        table_data = []
        for v in violations:
            row = [
                v.get('type', 'N/A'),
                v.get('clause_id', 'N/A'),
                f"{v.get('confidence', 0):.2%}",
                v.get('reason', 'N/A')[:100] + "..."
            ]
            table_data.append(row)

        return table_data

    def _get_retrieved_rules_table(self, result: dict) -> list:
        """获取检索到的条文表格"""
        rules = result.get('retrieved_rules', [])[:10]
        if not rules:
            return [["未检索到相关条文"]]

        table_data = []
        for r in rules:
            row = [
                r.get('clause_id', 'N/A'),
                r.get('source_file', 'N/A'),
                f"{r.get('score', 0):.4f}",
                r.get('clause_text', 'N/A')[:80] + "..."
            ]
            table_data.append(row)

        return table_data


# 创建 Demo 实例
demo = AuditDemo(kb_dir="kb", use_milvus=False)


def create_ui():
    """创建 Gradio UI"""

    with gr.Blocks(title="保险营销内容智能审核系统", theme=gr.themes.Soft()) as app:
        gr.Markdown("""
        # 🛡️ 保险营销内容智能审核系统

        基于 **RAG（检索增强生成）** 技术，使用通义千问大模型自动判断金融营销内容是否违反监管规定。

        ## ✨ 功能特点
        - ✅ 文本/图片/PDF 审核支持
        - ✅ 混合检索（BM25 + 向量检索）
        - ✅ 意图识别与问题重写
        - ✅ 数学公式置信度计算
        - ✅ 多维度合规分析

        ## 📋 使用说明
        1. 点击 **"加载资源"** 按钮初始化系统
        2. 输入营销文案或上传文件
        3. 选择审核模式和检索方式
        4. 点击 **"开始审核"** 查看结果
        """)

        with gr.Row():
            load_btn = gr.Button("🔄 加载资源", variant="primary", size="lg")
            resource_status = gr.Textbox(label="资源状态", value="⏳ 未加载", interactive=False)

        with gr.Tabs():
            # 文本审核
            with gr.Tab("📝 文本审核"):
                with gr.Row():
                    with gr.Column(scale=3):
                        text_input = gr.Textbox(
                            label="营销文案",
                            placeholder="请输入需要审核的营销文案...",
                            lines=5,
                        )
                    with gr.Column(scale=1):
                        mode_choice = gr.Radio(
                            choices=["基础模式", "增强模式"],
                            value="基础模式",
                            label="审核模式"
                        )
                        hybrid_toggle = gr.Checkbox(
                            label="使用混合检索（BM25 + 向量）",
                            value=False
                        )
                        top_k_slider = gr.Slider(
                            minimum=3,
                            maximum=12,
                            value=6,
                            step=1,
                            label="检索相关条文数量"
                        )

                audit_btn = gr.Button("🔍 开始审核", variant="primary", size="lg")

            # 文件审核
            with gr.Tab("📁 文件审核"):
                file_input = gr.File(
                    label="上传文件",
                    file_types=[".txt", ".pdf", ".docx", ".doc", ".png", ".jpg", ".jpeg", ".md"]
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        mode_choice_file = gr.Radio(
                            choices=["基础模式", "增强模式"],
                            value="基础模式",
                            label="审核模式"
                        )
                    with gr.Column(scale=1):
                        hybrid_toggle_file = gr.Checkbox(
                            label="使用混合检索",
                            value=False
                        )

                audit_file_btn = gr.Button("🔍 审核文件", variant="primary", size="lg")

            # 图片审核
            with gr.Tab("🖼️ 图片审核"):
                with gr.Row():
                    with gr.Column(scale=2):
                        image_input = gr.Image(
                            label="上传营销图片",
                            type="filepath",
                            sources=["upload", "clipboard"],
                        )
                    with gr.Column(scale=1):
                        text_context_input = gr.Textbox(
                            label="文字说明（可选）",
                            placeholder="可以输入额外的文字说明...",
                            lines=3,
                        )
                        top_k_slider_img = gr.Slider(
                            minimum=3,
                            maximum=12,
                            value=6,
                            step=1,
                            label="检索相关条文数量"
                        )

                audit_image_btn = gr.Button("🔍 审核图片", variant="primary", size="lg")

        # 输出区域
        with gr.Row():
            with gr.Column():
                output_status = gr.Textbox(label="审核状态", interactive=False)
                output_markdown = gr.Markdown(label="审核结果")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### ⚠️ 违规项")
                violations_table = gr.Dataframe(
                    label="违规详情",
                    headers=["违规类型", "条文编号", "置信度", "违规原因"],
                    datatype=["str", "str", "str", "str"],
                    row_count=5,
                    col_count=(4, "fixed")
                )

            with gr.Column():
                gr.Markdown("### 📚 检索到的相关条文")
                rules_table = gr.Dataframe(
                    label="相关法规",
                    headers=["条文编号", "来源文件", "相似度", "条文内容"],
                    datatype=["str", "str", "str", "str"],
                    row_count=5,
                    col_count=(4, "fixed")
                )

        # 示例
        gr.Markdown("---")
        gr.Markdown("## 💡 示例文案")

        with gr.Row():
            example_1 = gr.Button("📌 违规示例：夸大收益")
            example_2 = gr.Button("📌 违规示例：误导宣传")
            example_3 = gr.Button("✅ 合规示例：风险提示")

        # 事件绑定
        load_btn.click(
            fn=demo.load_resources,
            outputs=resource_status
        )

        audit_btn.click(
            fn=demo.audit_text,
            inputs=[text_input, mode_choice, top_k_slider, hybrid_toggle],
            outputs=[output_status, output_markdown, violations_table, rules_table]
        )

        audit_file_btn.click(
            fn=demo.audit_file,
            inputs=[file_input, mode_choice_file, gr.Number(value=6, visible=False), hybrid_toggle_file],
            outputs=[output_status, output_markdown, violations_table, rules_table]
        )

        audit_image_btn.click(
            fn=demo.audit_image,
            inputs=[image_input, text_context_input, gr.Textbox(value="基础模式", visible=False), top_k_slider_img],
            outputs=[output_status, output_markdown, violations_table, rules_table]
        )

        # 示例按钮
        example_1.click(
            lambda: "本保险产品保本保收益，年化收益率保证8%，零风险，稳赚不赔！",
            outputs=text_input
        )
        example_2.click(
            lambda: "限时抢购！购买后稳赚不赔，现在下单即可翻倍收益！",
            outputs=text_input
        )
        example_3.click(
            lambda: "本产品过往业绩不代表未来表现，具体以合同条款为准。投资有风险，投保需谨慎。",
            outputs=text_input
        )

        gr.Markdown("---")
        gr.Markdown("""
        ### 📊 技术架构

        - **大模型**: 通义千问（qwen-plus）
        - **向量检索**: text-embedding-v3 + Milvus Lite
        - **混合检索**: BM25 + 向量检索（可选）
        - **多模态**: Qwen-VL 用于 PDF/图片 OCR
        - **置信度**: 数学公式 + LLM 输出

        ### 📞 联系方式

        - 百炼控制台: https://bailian.console.aliyun.com/
        """)

    return app


if __name__ == "__main__":
    ui = create_ui()
    ui.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
