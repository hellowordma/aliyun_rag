"""
ReAct框架实现

ReAct (Reasoning + Acting) 是一种让LLM能够推理和行动的框架。

核心思想：
1. Thought: LLM进行推理，决定下一步要做什么
2. Action: LLM选择并执行一个工具
3. Observation: LLM观察工具执行的结果
4. 重复上述步骤直到得到最终答案

参考资料：
- ReAct: Synergizing Reasoning and Acting in Language Models (https://arxiv.org/abs/2210.03629)
"""

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Set
from datetime import datetime


def get_reference_files(project_root: Optional[Path] = None) -> List[str]:
    """
    动态获取references目录中的法规文件列表

    Args:
        project_root: 项目根目录路径，如果为None则自动检测

    Returns:
        法规文件名列表（不含后缀）
    """
    if project_root is None:
        # 尝试自动检测项目根目录
        current = Path(__file__).resolve()
        # agent/aliyun_rag/agent -> agent/aliyun_rag -> agent
        for parent in current.parents:
            if (parent / "references").exists():
                project_root = parent
                break
            if (parent / "aliyun_rag" / "references").exists():
                project_root = parent / "aliyun_rag"
                break

    if project_root is None:
        return []

    # 检查references目录
    refs_dirs = [
        project_root / "references",
        project_root / "aliyun_rag" / "references",
    ]

    legal_extensions = {'.txt', '.doc', '.docx', '.pdf'}
    reference_files: Set[str] = set()

    for refs_dir in refs_dirs:
        if not refs_dir.exists():
            continue
        for file_path in refs_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in legal_extensions:
                # 去掉文件后缀，使用文件名作为法规名称
                reference_files.add(file_path.stem)

    return sorted(list(reference_files))


def get_reference_files_from_kb(kb: Any) -> List[str]:
    """
    从知识库中获取法规文件列表

    Args:
        kb: 知识库对象（KnowledgeBase或MilvusKnowledgeBase）

    Returns:
        法规文件名列表（去重后）
    """
    source_files = set()
    for chunk in kb.chunks:
        source_files.add(chunk.source_file)

    return sorted(list(source_files))


@dataclass
class Thought:
    """推理步骤"""
    content: str
    step: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class Action:
    """行动步骤"""
    tool_name: str
    parameters: Dict[str, Any]
    step: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class Observation:
    """观察结果"""
    tool_name: str
    result: Any
    step: int
    success: bool
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat)

    def to_summary(self, max_length: int = 500) -> str:
        """将结果转换为摘要文本"""
        if isinstance(self.result, dict):
            # 处理ToolResult嵌套结构 {success, data, error, ...}
            data = self.result.get("data", self.result)

            # 提取关键信息
            if "is_compliant" in data:
                summary = f"合规状态: {data.get('is_compliant', 'unknown')}"
                if "violations" in data and data["violations"]:
                    v_count = len(data["violations"])
                    summary += f", 违规项: {v_count}个"
                    # 添加第一个违规项
                    first_v = data["violations"][0]
                    summary += f" ({first_v.get('type', 'N/A')})"
                if "summary" in data:
                    summary += f", 总结: {data['summary'][:80]}"
                return summary
            elif "error" in self.result and self.result["error"]:
                return f"错误: {self.result['error']}"
            elif self.success and "data" in self.result:
                # 有成功返回但data不是审核结果格式
                return f"成功: {str(self.result.get('data', ''))[:max_length-10]}"
            else:
                # 简化JSON输出
                return json.dumps(self.result, ensure_ascii=False)[:max_length]
        else:
            return str(self.result)[:max_length]


@dataclass
class ReActStep:
    """完整的ReAct步骤"""
    step: int
    thought: Optional[Thought] = None
    action: Optional[Action] = None
    observation: Optional[Observation] = None


@dataclass
class ReActResult:
    """ReAct执行结果"""
    answer: str
    steps: List[ReActStep]
    success: bool
    error: Optional[str] = None
    total_steps: int = 0
    total_execution_time: float = 0.0

    def to_trace(self) -> str:
        """生成执行轨迹（用于调试和可观测性）"""
        lines = ["=" * 60, "ReAct Execution Trace", "=" * 60]
        for step in self.steps:
            lines.append(f"\n--- Step {step.step} ---")
            if step.thought:
                lines.append(f"[Thought] {step.thought.content}")
            if step.action:
                params = json.dumps(step.action.parameters, ensure_ascii=False)
                lines.append(f"[Action] {step.action.tool_name}({params})")
            if step.observation:
                status = "✓" if step.observation.success else "✗"
                lines.append(f"[Observation] {status} {step.observation.to_summary()}")

        lines.append("\n" + "=" * 60)
        lines.append(f"[Final Answer] {self.answer}")
        lines.append(f"[Total Steps] {self.total_steps}")
        lines.append(f"[Execution Time] {self.total_execution_time:.2f}s")
        lines.append("=" * 60)
        return "\n".join(lines)


class ReActAgent:
    """
    ReAct Agent实现

    支持的功能：
    - 自动推理和工具选择
    - 多轮对话和决策
    - 错误处理和重试
    - 可观测性（轨迹记录）
    - Fallback机制
    - 动态获取知识库中的法规文件
    """

    def __init__(
        self,
        client: Any,  # BailianClient
        tools: Any,  # ToolRegistry
        system_prompt: str = "",
        max_steps: int = 10,
        max_retries: int = 3,
        verbose: bool = True,
        kb: Any = None,  # 知识库对象，用于动态获取法规文件列表
        project_root: Optional[Path] = None,  # 项目根目录
    ):
        """
        初始化ReAct Agent

        Args:
            client: LLM客户端
            tools: 工具注册表
            system_prompt: 系统提示词
            max_steps: 最大推理步骤数
            max_retries: 工具执行失败时的最大重试次数
            verbose: 是否输出详细日志
        """
        self.client = client
        self.tools = tools
        self.kb = kb
        self.project_root = project_root or (Path(__file__).parent.parent if hasattr(Path(__file__), 'parent') else Path.cwd())
        self.system_prompt = system_prompt or self._default_system_prompt()
        self.max_steps = max_steps
        self.max_retries = max_retries
        self.verbose = verbose

        # 执行历史
        self.history: List[ReActStep] = []

        # 缓存的法规文件列表
        self._cached_reference_files: Optional[List[str]] = None

    def _get_reference_files(self) -> List[str]:
        """获取知识库中的法规文件列表（带缓存）"""
        if self._cached_reference_files is None:
            # 优先从知识库获取
            if self.kb is not None:
                self._cached_reference_files = get_reference_files_from_kb(self.kb)
            else:
                # 从references目录获取
                self._cached_reference_files = get_reference_files(self.project_root)
        return self._cached_reference_files

    def _default_system_prompt(self) -> str:
        """默认系统提示词（动态获取法规文件列表）"""
        tool_descriptions = self.tools.get_tool_descriptions()

        # 动态获取法规文件列表
        reference_files = self._get_reference_files()

        if reference_files:
            files_list = "\n".join([f"   - {f}" for f in reference_files])
            files_count = len(reference_files)
            constraint_text = f"""2. 知识库中包含以下{files_count}个法规文件：
{files_list}
3. 只能引用上述法规文件，不得引用其他法律（如《广告法》《保险法》等）"""
        else:
            constraint_text = """2. 只能引用工具检索结果中的法规文件
3. 不得引用知识库中不存在的法律"""

        return f"""你是一个保险营销内容智能审核助手，可以帮助用户审核各种营销内容的合规性。

【重要约束 - 防止幻觉】
1. 你只能使用工具返回的信息进行判断，严禁编造法规条文
{constraint_text}
4. 所有法规引用必须来自工具的检索结果，且必须包含完整的条文编号和来源文件名

你可以使用以下工具来完成任务：

{tool_descriptions}

工作流程：
1. 分析用户的问题，思考需要使用哪些工具
2. 选择合适的工具并执行
3. 观察工具返回的结果（包括检索到的具体条文）
4. 根据工具返回结果给出最终答案

回复格式要求：
- Thought: [你的推理过程]
- Action: [工具名称] with parameters: {{参数名: 参数值}}
- Final Answer: [基于工具返回结果的答案]

注意：
- 一次只执行一个工具
- Final Answer 必须基于工具的实际返回结果
- 如果工具返回的检索结果为空，说明"未找到相关条文"
- 绝对禁止编造工具执行结果或法规条文
"""

    def run(self, query: str) -> ReActResult:
        """
        运行ReAct Agent

        Args:
            query: 用户查询

        Returns:
            ReActResult: 执行结果
        """
        import time
        start_time = time.time()

        self.history = []
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"用户问题: {query}\n\n请开始分析。"},
        ]

        step_count = 0
        retry_count = 0

        if self.verbose:
            print(f"\n[ReAct] 开始处理查询: {query[:50]}...")

        try:
            while step_count < self.max_steps:
                step_count += 1

                # 调用LLM进行推理
                if self.verbose:
                    print(f"\n[ReAct] Step {step_count}: 调用LLM...")

                response = self.client.chat(
                    messages=messages,
                    temperature=0.0,
                )

                if self.verbose:
                    print(f"[ReAct] LLM响应: {response[:200]}...")

                # 解析响应
                thought, action, final_answer = self._parse_response(response, verbose=self.verbose)

                # 记录推理步骤
                if thought:
                    thought_obj = Thought(content=thought, step=step_count)
                    self.history.append(ReActStep(step=step_count, thought=thought_obj))
                    if self.verbose:
                        print(f"[Step {step_count}][Thought] {thought}")

                # 检查是否给出最终答案
                if final_answer:
                    result = ReActResult(
                        answer=final_answer,
                        steps=self.history,
                        success=True,
                        total_steps=step_count,
                    )
                    if self.verbose:
                        print(f"[Step {step_count}][Final Answer] {final_answer}")
                    return result

                # 执行行动
                if action:
                    action_obj = Action(
                        tool_name=action["tool"],
                        parameters=action["parameters"],
                        step=step_count,
                    )
                    self.history.append(ReActStep(step=step_count, action=action_obj))

                    if self.verbose:
                        print(f"[Step {step_count}][Action] {action['tool']}({action['parameters']})")

                    # 执行工具
                    tool_result = self.tools.get(action["tool"])
                    if not tool_result:
                        observation = Observation(
                            tool_name=action["tool"],
                            result={"error": f"Tool not found: {action['tool']}"},
                            step=step_count,
                            success=False,
                        )
                        retry_count += 1
                    else:
                        execution_result = tool_result.execute(**action["parameters"])
                        observation = Observation(
                            tool_name=action["tool"],
                            result=execution_result.to_dict(),
                            step=step_count,
                            success=execution_result.success,
                        )
                        if execution_result.success:
                            retry_count = 0  # 成功则重置重试计数
                        else:
                            retry_count += 1

                    self.history.append(ReActStep(step=step_count, observation=observation))

                    if self.verbose:
                        status = "✓" if observation.success else "✗"
                        print(f"[Step {step_count}][Observation] {status} {observation.to_summary()}")

                    # 构建下一步的消息
                    # 如果是audit_text工具，需要包含完整的检索条文信息
                    if observation.tool_name == "audit_text" and observation.success:
                        detailed_result = observation.result.get("data", observation.result)

                        # 首先构建检索条文列表（放在最前面）
                        observation_text = f"Tool {observation.tool_name} 返回结果：\n\n"
                        if "retrieved_rules" in detailed_result:
                            observation_text += f"【检索到的相关条文】（共{len(detailed_result['retrieved_rules'])}条）\n"
                            observation_text += "【重要】Final Answer中只能引用以下条文，禁止引用其他条文：\n\n"
                            for i, rule in enumerate(detailed_result["retrieved_rules"], 1):
                                observation_text += f"  [{i}] {rule.get('source_file', 'N/A')} | 条文: {rule.get('clause_id', 'N/A')}\n"
                                observation_text += f"      内容: {rule.get('clause_text', 'N/A')[:150]}...\n\n"

                        # 审核结果
                        if "is_compliant" in detailed_result:
                            observation_text += f"【审核结果】\n合规状态: {detailed_result['is_compliant']}\n"

                        if "violations" in detailed_result and detailed_result["violations"]:
                            observation_text += f"\n违规项 ({len(detailed_result['violations'])}个):\n"
                            for v in detailed_result["violations"][:3]:
                                observation_text += f"  - {v.get('type', 'N/A')}: {v.get('reason', 'N/A')[:80]}...\n"

                        if "summary" in detailed_result:
                            observation_text += f"\n总结: {detailed_result['summary'][:150]}"

                        # 最后再次强调
                        observation_text += "\n\n【再次强调】Final Answer必须只引用上述【检索到的相关条文】列表中的条文。"
                    else:
                        observation_text = f"Tool {observation.tool_name} returned: {observation.to_summary()}"

                    messages.append({"role": "assistant", "content": response})
                    messages.append({"role": "user", "content": observation_text})

                    # 检查重试次数
                    if retry_count >= self.max_retries:
                        result = ReActResult(
                            answer=f"经过{step_count}步推理后，未能完成审核任务。工具执行多次失败，请稍后重试。",
                            steps=self.history,
                            success=False,
                            error="Max retries exceeded",
                            total_steps=step_count,
                        )
                        return result

            # 达到最大步骤数
            result = ReActResult(
                answer=f"经过{step_count}步推理后，未能给出最终答案。可能问题过于复杂，请简化后重试。",
                steps=self.history,
                success=False,
                error="Max steps exceeded",
                total_steps=step_count,
            )
            return result

        except Exception as e:
            return ReActResult(
                answer=f"执行过程中发生错误: {str(e)}",
                steps=self.history,
                success=False,
                error=str(e),
                total_steps=step_count,
            )

    def _parse_response(self, response: str, verbose: bool = False) -> tuple[Optional[str], Optional[Dict], Optional[str]]:
        """
        解析LLM响应

        Returns:
            (thought, action, final_answer)
        """
        thought = None
        action = None
        final_answer = None

        if verbose:
            print(f"[Parse] 原始响应长度: {len(response)}")

        # 查找 Thought (支持中英文)
        thought_patterns = [
            r'Thought:\s*(.*?)(?=\n(?:Action|Final Answer|最终答案)|$)',
            r'思考:\s*(.*?)(?=\n(?:Action|行动|Final Answer|最终答案)|$)',
            r'[思考|Thought]:\s*(.*?)(?=\n[行动|Action]|$)',
        ]
        for pattern in thought_patterns:
            thought_match = re.search(pattern, response, re.IGNORECASE)
            if thought_match:
                thought = thought_match.group(1).strip()
                if verbose:
                    print(f"[Parse] 找到Thought: {thought[:50]}...")
                break

        # 先查找 Action (优先级高于Final Answer，避免误判)
        # 支持不完整JSON的情况
        action_patterns = [
            r'Action:\s*(\w+)\s+with\s+parameters:\s*(\{.*?\})(?=\n\n|Final|$)',
            r'Action:\s*(\w+)\s+with\s+parameters:\s*(\{[^}]*"marketing_text"[^}]*\})',
            r'行动:\s*(\w+)\s+参数:\s*(\{.*?\})',
            r'行动：\s*(\w+)\s+参数：\s*(\{.*?\})',
        ]

        # 首先尝试完整匹配
        for pattern in action_patterns:
            action_match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if action_match:
                tool_name = action_match.group(1).strip()
                try:
                    parameters = json.loads(action_match.group(2))
                    action = {"tool": tool_name, "parameters": parameters}
                    if verbose:
                        print(f"[Parse] 找到Action: {tool_name}")
                    return thought, action, final_answer
                except json.JSONDecodeError:
                    pass

        # 尝试宽松的Action解析 - 提取工具名和参数
        loose_action = re.search(r'Action:\s*(\w+)\s+with\s+parameters:\s*\{[^\n]*\}?', response, re.IGNORECASE)
        if loose_action:
            tool_name = loose_action.group(1).strip()
            # 尝试提取JSON部分
            json_match = re.search(r'\{[^}]*"marketing_text"[^}]*\}', response, re.IGNORECASE)
            if json_match:
                try:
                    parameters = json.loads(json_match.group(0))
                    action = {"tool": tool_name, "parameters": parameters}
                    if verbose:
                        print(f"[Parse] 宽松解析找到Action: {tool_name}")
                    return thought, action, final_answer
                except json.JSONDecodeError:
                    pass

        # 查找 Final Answer (支持中英文) - 但要确保不是在Action之后
        final_patterns = [
            r'\nFinal Answer:\s*(.*)',
            r'最终答案:\s*(.*)',
            r'最终答案：\s*(.*)',
        ]
        for pattern in final_patterns:
            final_match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if final_match:
                final_answer = final_match.group(1).strip()
                # 确保答案不为空且不是占位符
                if final_answer and '等待' not in final_answer and '...' not in final_answer[:20]:
                    if verbose:
                        print(f"[Parse] 找到Final Answer")
                    return thought, None, final_answer

        # 最后尝试：如果包含Action但没有完整参数
        very_loose = re.search(r'Action:\s*(\w+)', response, re.IGNORECASE)
        if very_loose:
            tool_name = very_loose.group(1).strip()
            action = {"tool": tool_name, "parameters": {}}
            if verbose:
                print(f"[Parse] 宽松解析找到Action: {tool_name}")
            return thought, action, final_answer

        # 如果都没找到，检查是否直接给出了答案
        if not action and not final_answer:
            # 检查响应中是否包含审核结果关键词
            if any(kw in response for kw in ['合规', '违规', '不合规', 'compliant', 'violation']):
                if verbose:
                    print(f"[Parse] 响应包含审核结果，作为最终答案处理")
                final_answer = response
                return thought, action, final_answer

        if verbose:
            print(f"[Parse] 未找到可解析的结构")

        return thought, action, final_answer


class ReActAgentWithFunctionCalling(ReActAgent):
    """
    使用Function Calling的ReAct Agent

    相比原始ReAct的优势：
    - 结构化的工具调用（LLM直接生成JSON参数）
    - 更少的解析错误
    - 更好的工具选择准确性
    """

    def run(self, query: str) -> ReActResult:
        """使用Function Calling运行Agent"""
        import time
        start_time = time.time()

        self.history = []
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": query},
        ]

        step_count = 0
        retry_count = 0

        tool_schemas = self.tools.get_tool_schema()

        try:
            while step_count < self.max_steps:
                step_count += 1

                # 调用LLM（这里简化实现，实际需要支持function calling的API）
                response = self.client.chat(
                    messages=messages,
                    temperature=0.0,
                )

                # 检查是否需要调用工具（简化版，假设响应格式）
                if "tool_call" in response.lower() or "action:" in response.lower():
                    # 解析工具调用
                    action = self._parse_tool_call(response)
                    if action:
                        thought = f"需要调用工具 {action['tool']} 来获取信息"

                        thought_obj = Thought(content=thought, step=step_count)
                        action_obj = Action(
                            tool_name=action["tool"],
                            parameters=action["parameters"],
                            step=step_count,
                        )
                        self.history.extend([
                            ReActStep(step=step_count, thought=thought_obj),
                            ReActStep(step=step_count, action=action_obj),
                        ])

                        if self.verbose:
                            print(f"[Step {step_count}][Thought] {thought}")
                            print(f"[Step {step_count}][Action] {action['tool']}({action['parameters']})")

                        # 执行工具
                        tool_result = self.tools.get(action["tool"])
                        if tool_result:
                            execution_result = tool_result.execute(**action["parameters"])
                            observation = Observation(
                                tool_name=action["tool"],
                                result=execution_result.to_dict(),
                                step=step_count,
                                success=execution_result.success,
                            )

                            self.history.append(ReActStep(step=step_count, observation=observation))

                            if self.verbose:
                                status = "✓" if observation.success else "✗"
                                print(f"[Step {step_count}][Observation] {status} {observation.to_summary()}")

                            # 添加到消息
                            messages.append({"role": "assistant", "content": response})
                            messages.append({
                                "role": "user",
                                "content": f"工具返回: {observation.to_summary()}\n\n请基于这个结果给出分析。"
                            })

                            if execution_result.success:
                                retry_count = 0
                            else:
                                retry_count += 1

                            if retry_count >= self.max_retries:
                                return ReActResult(
                                    answer=f"工具执行多次失败，请稍后重试。",
                                    steps=self.history,
                                    success=False,
                                    error="Max retries exceeded",
                                    total_steps=step_count,
                                )
                        else:
                            step_count += 1
                            continue
                else:
                    # 直接回答
                    return ReActResult(
                        answer=response,
                        steps=self.history,
                        success=True,
                        total_steps=step_count,
                    )

            # 达到最大步骤
            return ReActResult(
                answer="未能完成审核任务。",
                steps=self.history,
                success=False,
                error="Max steps exceeded",
                total_steps=step_count,
            )

        except Exception as e:
            return ReActResult(
                answer=f"执行错误: {str(e)}",
                steps=self.history,
                success=False,
                error=str(e),
                total_steps=step_count,
            )

    def _parse_tool_call(self, response: str) -> Optional[Dict]:
        """解析工具调用"""
        # 简化版解析
        action_match = re.search(r'Action:\s*(\w+)\s+with\s+parameters:\s*(\{.*?\})', response, re.IGNORECASE | re.DOTALL)
        if action_match:
            tool_name = action_match.group(1).strip()
            try:
                parameters = json.loads(action_match.group(2))
                return {"tool": tool_name, "parameters": parameters}
            except json.JSONDecodeError:
                pass
        return None
