"""
Agent工具定义模块

定义Agent可以调用的所有工具，包括：
- 文本审核工具
- 图片审核工具
- PDF文档审核工具
- 知识库查询工具
- 意图分析工具
"""

import base64
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

from ..bailian_client import BailianClient
from ..auditor import audit_marketing_text
from ..enhanced_auditor import enhanced_audit_marketing_text, EnhancedAuditor
from ..multimodal_auditor import audit_marketing_image
from ..extractors import extract_text_from_file


T = TypeVar('T')


@dataclass
class ToolParameter:
    """工具参数定义"""
    name: str
    type: str  # "string", "number", "boolean", "file", "image"
    description: str
    required: bool = True
    default: Any = None


@dataclass
class ToolResult:
    """工具执行结果"""
    success: bool
    data: Any
    error: Optional[str] = None
    tool_name: str = ""
    execution_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "tool_name": self.tool_name,
            "execution_time": self.execution_time,
        }


@dataclass
class Tool:
    """工具定义"""
    name: str
    description: str
    parameters: List[ToolParameter] = field(default_factory=list)
    func: Optional[Callable] = None
    return_description: str = ""

    def execute(self, **kwargs) -> ToolResult:
        """执行工具"""
        import time
        start_time = time.time()

        try:
            if self.func is None:
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Tool {self.name} has no implementation",
                    tool_name=self.name,
                )

            # 验证必需参数
            for param in self.parameters:
                if param.required and param.name not in kwargs:
                    return ToolResult(
                        success=False,
                        data=None,
                        error=f"Missing required parameter: {param.name}",
                        tool_name=self.name,
                    )

            result = self.func(**kwargs)
            execution_time = time.time() - start_time

            return ToolResult(
                success=True,
                data=result,
                tool_name=self.name,
                execution_time=execution_time,
            )
        except Exception as e:
            execution_time = time.time() - start_time
            return ToolResult(
                success=False,
                data=None,
                error=str(e),
                tool_name=self.name,
                execution_time=execution_time,
            )


class ToolRegistry:
    """工具注册表"""

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """注册工具"""
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        """获取工具"""
        return self._tools.get(name)

    def list_tools(self) -> List[Tool]:
        """列出所有工具"""
        return list(self._tools.values())

    def get_tool_descriptions(self) -> str:
        """获取工具描述文本，用于Prompt"""
        descriptions = []
        for tool in self._tools.values():
            param_desc = ", ".join([
                f"{p.name}: {p.type}"
                for p in tool.parameters
            ])
            descriptions.append(
                f"- {tool.name}({param_desc}): {tool.description}\n"
                f"  返回: {tool.return_description}"
            )
        return "\n\n".join(descriptions)

    def get_tool_schema(self) -> List[Dict[str, Any]]:
        """获取工具的JSON Schema格式（用于Function Calling）"""
        schemas = []
        for tool in self._tools.values():
            schema = {
                "name": tool.name,
                "description": tool.description,
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            }
            for param in tool.parameters:
                schema["parameters"]["properties"][param.name] = {
                    "type": param.type,
                    "description": param.description,
                }
                if param.required:
                    schema["parameters"]["required"].append(param.name)
                if param.default is not None:
                    schema["parameters"]["properties"][param.name]["default"] = param.default

            schemas.append(schema)
        return schemas


def create_audit_tools(kb: Any, client: BailianClient) -> ToolRegistry:
    """创建审核工具集

    Args:
        kb: 知识库实例
        client: Bailian客户端实例

    Returns:
        ToolRegistry: 包含所有审核工具的注册表
    """
    registry = ToolRegistry()

    # 1. 文本审核工具
    def audit_text(marketing_text: str, use_enhanced: bool = True) -> Dict[str, Any]:
        """审核营销文本内容"""
        if use_enhanced:
            result = enhanced_audit_marketing_text(
                marketing_text=marketing_text,
                kb=kb,
                client=client,
                top_k=6,
            )
        else:
            result = audit_marketing_text(
                marketing_text=marketing_text,
                kb=kb,
                client=client,
                top_k=6,
            )
        return result

    registry.register(Tool(
        name="audit_text",
        description="审核保险营销文本内容的合规性，检测是否包含违规内容如夸大收益、误导宣传等",
        parameters=[
            ToolParameter("marketing_text", "string", "待审核的营销文案内容", required=True),
            ToolParameter("use_enhanced", "boolean", "是否使用增强审核（包含意图识别和多阶段推理）", required=False, default=True),
        ],
        func=audit_text,
        return_description="审核结果，包含是否合规、违规项列表、置信度等",
    ))

    # 2. 意图识别工具
    def analyze_intent(marketing_text: str) -> Dict[str, Any]:
        """分析营销文案的意图和风险点"""
        return EnhancedAuditor.identify_intent(marketing_text)

    registry.register(Tool(
        name="analyze_intent",
        description="分析营销文案的意图，识别主要违规类型和风险等级",
        parameters=[
            ToolParameter("marketing_text", "string", "待分析的营销文案内容", required=True),
        ],
        func=analyze_intent,
        return_description="意图分析结果，包含主要意图、检测到的风险、风险等级等",
    ))

    # 3. 图片审核工具
    def audit_image(image_path: str, text_context: str = "") -> Dict[str, Any]:
        """审核营销图片内容"""
        path = Path(image_path)
        if not path.exists():
            return {
                "success": False,
                "error": f"File not found: {image_path}"
            }

        # 判断图片类型
        ext = path.suffix.lower()
        mime_map = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
        }
        if ext not in mime_map:
            return {
                "success": False,
                "error": f"Unsupported image format: {ext}"
            }

        with open(image_path, 'rb') as f:
            image_bytes = f.read()

        result = audit_marketing_image(
            image_bytes=image_bytes,
            kb=kb,
            client=client,
            image_mime=mime_map[ext],
            text_context=text_context if text_context else None,
            top_k=6,
        )
        return result

    registry.register(Tool(
        name="audit_image",
        description="审核保险营销图片内容，支持OCR识别图片中的文字并进行合规审核",
        parameters=[
            ToolParameter("image_path", "string", "图片文件路径", required=True),
            ToolParameter("text_context", "string", "可选的文字说明，与图片一起分析", required=False, default=""),
        ],
        func=audit_image,
        return_description="图片审核结果，包含是否合规、违规项、图片分析等",
    ))

    # 4. PDF文档审核工具
    def audit_pdf(file_path: str, max_pages: int = 10) -> Dict[str, Any]:
        """审核PDF文档内容"""
        path = Path(file_path)
        if not path.exists():
            return {
                "success": False,
                "error": f"File not found: {file_path}"
            }

        if path.suffix.lower() != '.pdf':
            return {
                "success": False,
                "error": "Only PDF files are supported"
            }

        # 提取文本
        try:
            extracted_text = extract_text_from_file(
                file_path=str(path),
                client=client,
                pdf_mode="vl",
                max_pages=max_pages,
            )

            # 审核文本
            result = audit_marketing_text(
                marketing_text=extracted_text,
                kb=kb,
                client=client,
                top_k=6,
            )
            result["extracted_length"] = len(extracted_text)
            return result
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    registry.register(Tool(
        name="audit_pdf",
        description="审核PDF营销文档，使用OCR提取文档内容后进行合规审核",
        parameters=[
            ToolParameter("file_path", "string", "PDF文件路径", required=True),
            ToolParameter("max_pages", "number", "最大处理页数", required=False, default=10),
        ],
        func=audit_pdf,
        return_description="PDF审核结果，包含是否合规、违规项、提取文本长度等",
    ))

    # 5. 知识库查询工具
    def search_knowledge(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """在知识库中搜索相关法规条文"""
        if hasattr(kb, 'collection'):  # Milvus
            from .. import knowledge_base_milvus
            rules = knowledge_base_milvus.retrieve_relevant_rules(
                query=query,
                kb=kb,
                client=client,
                top_k=top_k,
            )
        else:  # NumPy
            from ..knowledge_base import retrieve_relevant_rules
            rules = retrieve_relevant_rules(
                query=query,
                kb=kb,
                client=client,
                top_k=top_k,
            )
        return rules

    registry.register(Tool(
        name="search_knowledge",
        description="在保险法规知识库中搜索相关的监管条文",
        parameters=[
            ToolParameter("query", "string", "搜索查询内容", required=True),
            ToolParameter("top_k", "number", "返回结果数量", required=False, default=5),
        ],
        func=search_knowledge,
        return_description="相关的法规条文列表，包含条文编号、原文、相似度等",
    ))

    # 6. 规则解释工具
    def explain_rule(rule_text: str) -> str:
        """解释法规条文的含义"""
        prompt = f"""请用通俗易懂的语言解释以下保险监管条文的含义和要求：

条文内容：
{rule_text}

请解释：
1. 这条规定的主要要求是什么？
2. 违反这条规定的常见情形有哪些？
3. 合规的建议是什么？

输出简洁明了的解释（200字以内）。"""

        return client.chat(
            messages=[
                {"role": "system", "content": "你是保险法规解释专家，擅长用通俗语言解释专业条文。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
        )

    registry.register(Tool(
        name="explain_rule",
        description="解释保险法规条文的含义和要求，帮助理解监管意图",
        parameters=[
            ToolParameter("rule_text", "string", "需要解释的法规条文内容", required=True),
        ],
        func=explain_rule,
        return_description="条文的通俗解释，包含主要要求、常见违规情形、合规建议等",
    ))

    # 7. 批量审核工具
    def batch_audit(texts: List[str]) -> List[Dict[str, Any]]:
        """批量审核多个营销文案"""
        results = []
        for text in texts:
            try:
                result = audit_marketing_text(
                    marketing_text=text,
                    kb=kb,
                    client=client,
                    top_k=6,
                )
                result["original_text"] = text[:100]  # 保存原始文本前100字
                results.append(result)
            except Exception as e:
                results.append({
                    "success": False,
                    "error": str(e),
                    "original_text": text[:100]
                })
        return results

    registry.register(Tool(
        name="batch_audit",
        description="批量审核多个营销文案，提高审核效率",
        parameters=[
            ToolParameter("texts", "array", "待审核的营销文案列表", required=True),
        ],
        func=batch_audit,
        return_description="批量审核结果列表",
    ))

    return registry
