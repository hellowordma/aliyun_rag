"""
保险营销内容智能审核Agent

结合ReAct框架和保险审核领域知识，实现智能审核Agent。
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from pathlib import Path

from .react_agent import ReActAgent, ReActResult, Thought, Action, Observation, ReActStep
from .tools import ToolRegistry, ToolResult


@dataclass
class AgentConfig:
    """Agent配置"""
    max_steps: int = 10
    max_retries: int = 3
    enable_fallback: bool = True
    enable_intent_routing: bool = True
    verbose: bool = True
    temperature: float = 0.0
    timeout: int = 60


@dataclass
class AgentResult:
    """Agent执行结果"""
    success: bool
    answer: str
    audit_result: Optional[Dict[str, Any]] = None
    steps: List[ReActStep] = field(default_factory=list)
    execution_time: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "success": self.success,
            "answer": self.answer,
            "audit_result": self.audit_result,
            "steps_count": len(self.steps),
            "execution_time": self.execution_time,
            "error": self.error,
            "metadata": self.metadata,
        }

    def to_trace(self) -> str:
        """生成执行轨迹"""
        lines = ["=" * 70, "保险审核Agent执行轨迹", "=" * 70]
        lines.append(f"\n执行时间: {datetime.now().isoformat()}")
        lines.append(f"执行时长: {self.execution_time:.2f}秒")
        lines.append(f"执行步骤: {len(self.steps)}步")
        lines.append(f"最终状态: {'成功' if self.success else '失败'}")

        if self.metadata:
            lines.append("\n元数据:")
            for k, v in self.metadata.items():
                lines.append(f"  {k}: {v}")

        lines.append("\n" + "-" * 70)
        lines.append("执行步骤详情:")
        lines.append("-" * 70)

        for step in self.steps:
            lines.append(f"\n[步骤 {step.step}]")
            if step.thought:
                lines.append(f"  思考: {step.thought.content}")
            if step.action:
                params = json.dumps(step.action.parameters, ensure_ascii=False)
                lines.append(f"  行动: {step.action.tool_name}({params})")
            if step.observation:
                status = "✓" if step.observation.success else "✗"
                lines.append(f"  观察: {status} {step.observation.to_summary()}")

        lines.append("\n" + "=" * 70)
        lines.append("最终答案:")
        lines.append("=" * 70)
        lines.append(self.answer)

        if self.audit_result:
            lines.append("\n" + "=" * 70)
            lines.append("审核结果:")
            lines.append("=" * 70)
            is_compliant = self.audit_result.get("is_compliant", "unknown")
            lines.append(f"合规状态: {is_compliant.upper()}")
            if "violations" in self.audit_result:
                violations = self.audit_result["violations"]
                lines.append(f"违规项: {len(violations)}个")
                for i, v in enumerate(violations[:3], 1):
                    lines.append(f"  {i}. {v.get('type', 'N/A')}: {v.get('reason', 'N/A')[:50]}...")

        lines.append("\n" + "=" * 70)
        return "\n".join(lines)


class InsuranceAuditAgent:
    """
    保险营销内容智能审核Agent

    特性：
    1. 意图路由：根据用户输入自动选择合适的审核策略
    2. 多工具协同：协调多个工具完成复杂审核任务
    3. Fallback机制：主流程失败时的备用方案
    4. 可观测性：完整的执行轨迹记录
    5. 上下文记忆：支持多轮对话
    """

    def __init__(
        self,
        client: Any,  # BailianClient
        tools: ToolRegistry,
        config: AgentConfig = AgentConfig(),
        kb: Any = None,  # 知识库对象，用于动态获取法规文件列表
    ):
        """
        初始化Agent

        Args:
            client: Bailian客户端
            tools: 工具注册表
            config: Agent配置
            kb: 知识库对象
        """
        self.client = client
        self.tools = tools
        self.config = config
        self.kb = kb

        # 对话历史
        self.conversation_history: List[Dict[str, str]] = []

        # 统计信息
        self.stats = {
            "total_queries": 0,
            "successful_audits": 0,
            "failed_audits": 0,
            "avg_steps": 0,
        }

    def audit(
        self,
        content: Union[str, bytes, Path],
        content_type: str = "auto",
        context: str = "",
    ) -> AgentResult:
        """
        审核营销内容

        Args:
            content: 待审核内容
                - str: 文本内容或文件路径
                - Path: 文件路径
                - bytes: 图片数据
            content_type: 内容类型 ("text", "image", "pdf", "auto")
            context: 额外上下文信息

        Returns:
            AgentResult: 审核结果
        """
        import time
        start_time = time.time()

        # 更新统计
        self.stats["total_queries"] += 1

        try:
            # 意图识别和路由
            if self.config.enable_intent_routing:
                detected_type = self._detect_content_type(content, content_type)
                query = self._build_query(content, detected_type, context)
            else:
                query = str(content)
                detected_type = content_type

            # 执行审核
            if detected_type in ("text", "auto"):
                result = self._audit_text(query)
            elif detected_type == "image":
                result = self._audit_image(str(content), context)
            elif detected_type == "pdf":
                result = self._audit_pdf(str(content))
            else:
                result = AgentResult(
                    success=False,
                    answer=f"不支持的内容类型: {content_type}",
                    error=f"Unsupported content type: {content_type}",
                )

            # 记录执行时间
            result.execution_time = time.time() - start_time
            result.metadata["content_type"] = detected_type

            # 更新统计
            if result.success:
                self.stats["successful_audits"] += 1
            else:
                self.stats["failed_audits"] += 1

            # 更新平均步骤数
            if result.steps:
                total = self.stats["avg_steps"] * (self.stats["total_queries"] - 1)
                total += len(result.steps)
                self.stats["avg_steps"] = total / self.stats["total_queries"]

            return result

        except Exception as e:
            return AgentResult(
                success=False,
                answer=f"审核过程发生错误: {str(e)}",
                execution_time=time.time() - start_time,
                error=str(e),
            )

    def chat(self, message: str) -> str:
        """
        对话模式

        Args:
            message: 用户消息

        Returns:
            str: Agent回复
        """
        # 添加到历史
        self.conversation_history.append({"role": "user", "content": message})

        # 构建上下文
        context = "\n".join([
            f"{msg['role']}: {msg['content']}"
            for msg in self.conversation_history[-5:]  # 只保留最近5轮
        ])

        # 调用ReAct Agent
        system_prompt = self._build_chat_system_prompt()
        react_agent = ReActAgent(
            client=self.client,
            tools=self.tools,
            system_prompt=system_prompt,
            max_steps=self.config.max_steps,
            verbose=self.config.verbose,
            kb=self.kb,
        )

        result = react_agent.run(message)

        # 提取回复
        response = result.answer

        # 添加到历史
        self.conversation_history.append({"role": "assistant", "content": response})

        return response

    def _detect_content_type(self, content: Any, hint_type: str) -> str:
        """检测内容类型"""
        if hint_type != "auto":
            return hint_type

        if isinstance(content, bytes):
            return "image"

        if isinstance(content, Path):
            suffix = content.suffix.lower()
            if suffix in ['.png', '.jpg', '.jpeg']:
                return "image"
            elif suffix == '.pdf':
                return "pdf"
            elif suffix == '.txt':
                return "text"

        if isinstance(content, str):
            # 检查是否是文件路径
            if Path(content).exists():
                suffix = Path(content).suffix.lower()
                if suffix in ['.png', '.jpg', '.jpeg']:
                    return "image"
                elif suffix == '.pdf':
                    return "pdf"
                elif suffix == '.txt':
                    return "text"
            # 否则视为文本内容
            return "text"

        return "text"

    def _build_query(self, content: Any, content_type: str, context: str) -> str:
        """构建审核查询"""
        base_query = f"请审核以下保险营销内容的合规性：\n\n"

        if content_type == "text":
            base_query += f"营销文案:\n{content}\n"
        elif content_type == "image":
            base_query += f"图片路径: {content}\n"
        elif content_type == "pdf":
            base_query += f"PDF文件: {content}\n"

        if context:
            base_query += f"\n附加说明:\n{context}\n"

        return base_query

    def _audit_text(self, query: str) -> AgentResult:
        """文本审核"""
        # 创建ReAct Agent
        system_prompt = """你是保险营销内容审核专家。

【重要约束】
1. 你只能使用工具返回的法规条文进行判断，严禁编造或引用其他法律法规
2. 知识库中只有以下3个文件，引用时必须准确：
   - 保险销售行为管理办法.pdf
   - 互联网保险业务监管办法.docx
   - 金融产品网络营销管理办法（征求意见稿）.doc
3. 如果检索结果中没有相关条文，明确说明"未检索到相关条文"
4. **特别注意**：Final Answer中引用的每一条法规都必须在工具返回的"检索到的相关条文"列表中出现

【审核流程】（只需1步完成）
1. 使用audit_text工具进行详细合规审核
2. 工具返回结果后，立即给出Final Answer

【回复格式】
- Action: audit_text with parameters: {"marketing_text": "文案内容"}
- Final Answer: [基于工具返回结果的审核结论]

【重要】
- 最多调用1次audit_text工具
- 得到工具返回结果后，必须立即给出Final Answer
- Final Answer必须引用工具返回的"检索到的相关条文"列表中的具体条文
- 绝对禁止引用检索结果列表中不存在的条文"""

        agent = ReActAgent(
            client=self.client,
            tools=self.tools,
            system_prompt=system_prompt,
            max_steps=3,  # 限制步骤数
            verbose=self.config.verbose,
            kb=self.kb,
        )

        result = agent.run(query)

        # 提取审核结果
        audit_data = self._extract_audit_data(result)

        return AgentResult(
            success=result.success,
            answer=result.answer,
            audit_result=audit_data,
            steps=result.steps,
            error=result.error,
        )

    def _audit_image(self, image_path: str, context: str) -> AgentResult:
        """图片审核"""
        query = f"请审核保险营销图片的合规性。图片路径: {image_path}"
        if context:
            query += f"\n附加说明: {context}"

        system_prompt = """你是保险营销图片审核专家。

审核流程：
1. 使用audit_image工具分析图片内容和提取文字
2. 检查是否包含违规内容（夸大收益、误导宣传等）
3. 给出明确的审核结论

请重点关注：
- 图片中的文字内容（OCR）
- 视觉元素（明星代言、夸张图表等）
- 整体营销信息是否合规"""

        agent = ReActAgent(
            client=self.client,
            tools=self.tools,
            system_prompt=system_prompt,
            max_steps=self.config.max_steps,
            verbose=self.config.verbose,
            kb=self.kb,
        )

        result = agent.run(query)

        audit_data = self._extract_audit_data(result)

        return AgentResult(
            success=result.success,
            answer=result.answer,
            audit_result=audit_data,
            steps=result.steps,
            error=result.error,
        )

    def _audit_pdf(self, pdf_path: str) -> AgentResult:
        """PDF文档审核"""
        query = f"请审核保险营销PDF文档的合规性。文件路径: {pdf_path}"

        system_prompt = """你是保险营销文档审核专家。

审核流程：
1. 使用audit_pdf工具提取PDF内容并审核
2. 分析审核结果
3. 给出明确的审核结论"""

        agent = ReActAgent(
            client=self.client,
            tools=self.tools,
            system_prompt=system_prompt,
            max_steps=self.config.max_steps,
            verbose=self.config.verbose,
            kb=self.kb,
        )

        result = agent.run(query)

        audit_data = self._extract_audit_data(result)

        return AgentResult(
            success=result.success,
            answer=result.answer,
            audit_result=audit_data,
            steps=result.steps,
            error=result.error,
        )

    def _extract_audit_data(self, react_result: ReActResult) -> Optional[Dict[str, Any]]:
        """从ReAct结果中提取审核数据"""
        for step in react_result.steps:
            if step.observation and step.observation.success:
                result = step.observation.result
                if isinstance(result, dict) and "is_compliant" in result:
                    return result
        return None

    def _build_chat_system_prompt(self) -> str:
        """构建对话系统提示词"""
        tool_descriptions = self.tools.get_tool_descriptions()

        return f"""你是保险营销合规审核专家助手，可以帮助用户审核各种保险营销内容。

可用工具：
{tool_descriptions}

对话原则：
1. 理解用户需求，选择合适的工具
2. 给出专业、准确的审核意见
3. 解释违规原因和改进建议
4. 保持简洁友好的交流风格

回复格式：
- Thought: [分析思路]
- Action: [工具调用] 或直接回答
- Final Answer: [最终结论]"""

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.stats.copy()

    def reset_history(self) -> None:
        """重置对话历史"""
        self.conversation_history = []

    def export_trace(self, result: AgentResult, filepath: str) -> None:
        """导出执行轨迹到文件"""
        trace = result.to_trace()
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(trace)
