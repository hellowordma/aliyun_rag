# 保险审核Agent实现原理与设计文档

## 目录

1. [概述](#概述)
2. [架构设计](#架构设计)
3. [ReAct框架实现](#react框架实现)
4. [工具调用机制](#工具调用机制)
5. [防幻觉机制](#防幻觉机制)
6. [Fallback机制](#fallback机制)
7. [使用示例](#使用示例)
8. [实现细节](#实现细节)
9. [扩展指南](#扩展指南)

---

## 概述

### 什么是Agent？

AI Agent是一种能够自主感知环境、做出决策并执行行动的智能系统。与传统程序的固定流程不同，Agent具有：

- **自主性**：能够自主决定执行什么操作
- **反应性**：能够感知环境变化并做出响应
- **主动性**：能够主动采取行动实现目标
- **社交性**：能够与其他Agent或人类协作

### 本项目的Agent设计

保险审核Agent基于**ReAct框架**实现，结合保险营销审核领域知识，能够：

1. 理解用户的审核需求
2. 自动选择合适的审核工具
3. 分析审核结果并给出专业建议
4. 处理各种异常情况
5. **自动适应法规文件变动**

---

## 架构设计

### 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         InsuranceAuditAgent                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐          │
│  │ 意图路由器   │───▶│ ReAct引擎   │───▶│ 结果聚合器   │          │
│  │ IntentRouter│    │ ReActEngine │    │ ResultAggr  │          │
│  └─────────────┘    └─────────────┘    └─────────────┘          │
│                                │                                 │
│                                ▼                                 │
│                       ┌─────────────┐                            │
│                       │  工具注册表   │                            │
│                       │ ToolRegistry │                            │
│                       └─────────────┘                            │
│                                │                                 │
│            ┌───────────────────┼───────────────────┐             │
│            ▼                   ▼                   ▼             │
│    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│    │ audit_text  │    │ audit_image │    │ audit_pdf   │        │
│    │ analyze_intent│   │search_knowledge│    │explain_rule │        │
│    │ batch_audit  │    │              │    │              │        │
│    └─────────────┘    └─────────────┘    └─────────────┘        │
│                                                                   │
├─────────────────────────────────────────────────────────────────┤
│                         核心能力                                   │
│  • 意图识别与路由    • 多工具协同      • 错误重试               │
│  • 执行轨迹记录      • 统计分析        • 对话记忆               │
│  • 动态法规获取      • 防幻觉机制      • 多层Fallback           │
└─────────────────────────────────────────────────────────────────┘
```

### 模块说明

| 模块 | 文件 | 功能 |
|------|------|------|
| 工具定义 | `tools.py` | 定义所有可用的审核工具，支持工具注册 |
| ReAct框架 | `react_agent.py` | 实现推理-行动循环，支持动态法规获取 |
| 审核Agent | `insurance_audit_agent.py` | 保险审核领域Agent，包含意图路由 |
| 演示程序 | `demo.py` | 使用示例和测试 |
| 技术文档 | `README.md` | 本文档 |
| 快速开始 | `QUICKSTART.md` | 快速入门指南 |

---

## ReAct框架实现

### ReAct原理

ReAct (Reasoning and Acting) 是一种让LLM能够进行推理和行动的框架。

**核心循环**：

```
┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
│ Thought │ ──▶ │ Action  │ ──▶ │Observation│ ──▶ │ Thought │ ...
│ (思考)  │     │ (行动)  │     │ (观察)  │     │ (思考)  │
└─────────┘     └─────────┘     └─────────�     └─────────�
                    │
                    ▼ (完成)
              ┌─────────┐
              │ Answer  │
              │ (答案)  │
              └─────────┘
```

### 代码实现

```python
class ReActAgent:
    def __init__(
        self,
        client: Any,           # LLM客户端
        tools: ToolRegistry,     # 工具注册表
        kb: Any = None,          # 知识库（用于获取法规文件列表）
        system_prompt: str = "", # 系统提示词
        max_steps: int = 10,      # 最大推理步骤
        max_retries: int = 3,      # 最大重试次数
        verbose: bool = True,     # 是否输出详细日志
    ):

    def run(self, query: str) -> ReActResult:
        """运行ReAct Agent"""
        # 动态获取系统提示词（包含实际法规文件列表）
        system_prompt = self.system_prompt or self._default_system_prompt()

        for step_count in range(self.max_steps):
            # 1. LLM推理
            response = self.client.chat(messages)

            # 2. 解析响应（Thought/Action/Final Answer）
            thought, action, final_answer = self._parse_response(response)

            # 3. 检查是否有最终答案
            if final_answer:
                return ReActResult(answer=final_answer, ...)

            # 4. 执行行动
            observation = self._execute_action(action)

            # 5. 观察结果并继续
            messages.extend([response, observation])
```

### 动态法规文件获取

```python
def get_reference_files(project_root: Optional[Path] = None) -> List[str]:
    """
    动态获取references目录中的法规文件列表

    支持的文件后缀：.txt, .doc, .docx, .pdf
    """
    # 自动定位references目录
    refs_dirs = [
        project_root / "references",
        project_root / "aliyun_rag" / "references",
    ]

    legal_extensions = {'.txt', '.doc', '.docx', '.pdf'}
    reference_files: Set[str] = set()

    for refs_dir in refs_dirs:
        for file_path in refs_dir.iterdir():
            if file_path.suffix.lower() in legal_extensions:
                reference_files.add(file_path.stem)

    return sorted(list(reference_files))


def get_reference_files_from_kb(kb: Any) -> List[str]:
    """
    从知识库chunks中提取法规文件列表
    """
    source_files = set()
    for chunk in kb.chunks:
        source_files.add(chunk.source_file)
    return sorted(list(source_files))
```

### Prompt设计

ReAct框架的Prompt设计至关重要。本项目的Prompt包含：

1. **动态生成的法规文件列表**：
   - 自动扫描references目录
   - 根据实际存在的法规文件生成约束
   - 支持法规文件变动

2. **严格的防幻觉约束**：
   - 只能引用工具返回的检索结果
   - Final Answer中的每一条法规都必须在检索结果中
   - 不得引用其他法律

3. **工具描述和格式要求**：
   - 清晰的工具描述
   - 明确的回复格式要求
   - Thought/Action/Final Answer格式

---

## 工具调用机制

### 工具定义

工具是Agent与外部世界交互的接口：

```python
@dataclass
class Tool:
    name: str                    # 工具名称
    description: str             # 工具描述（给LLM看）
    parameters: List[ToolParameter]  # 参数定义
    func: Callable               # 执行函数
    return_description: str      # 返回值描述
```

### 工具注册

```python
registry = ToolRegistry()

registry.register(Tool(
    name="audit_text",
    description="审核保险营销文本内容的合规性",
    parameters=[
        ToolParameter("marketing_text", "string", "待审核的营销文案", required=True),
        ToolParameter("use_enhanced", "boolean", "是否使用增强审核", required=False),
    ],
    func=audit_text_function,
    return_description="审核结果，包含是否合规、违规项列表等",
))
```

### Function Calling vs 文本解析

**文本解析方式**：
- 优点: 兼容所有LLM
- 缺点: 解析容易出错

**Function Calling方式**：
- 优点: 结构化调用，准确率高
- 缺点: 需要LLM支持

本项目实现了混合方案，支持两种方式。

### 工具Schema

工具Schema用于Function Calling，定义了工具的JSON Schema格式，供支持Function Calling的LLM使用。

---

## 防幻觉机制

### 问题背景

RAG系统中常见的幻觉问题：
1. LLM引用知识库中不存在的法规
2. LLM编造具体的条文款项
3. LLM引用来源错误的法规

### 解决方案

#### 1. 动态法规文件获取

```python
# 自动扫描references目录
# 根据实际文件生成约束
def _default_system_prompt(self) -> str:
    reference_files = self._get_reference_files()

    if reference_files:
        files_list = "\n".join([f"   - {f}" for f in reference_files])
        return f"""知识库中包含以下{len(reference_files)}个法规文件：
{files_list}
只能引用上述法规文件，不得引用其他法律。"""
```

#### 2. 完整条文传递

```python
# 在Observation中传递完整的检索条文
observation_text = f"""
【检索到的相关条文】（共{len(retrieved_rules)}条）
【重要】Final Answer中只能引用以下条文，禁止引用其他条文：

"""

for i, rule in enumerate(retrieved_rules, 1):
    observation_text += f"[{i}] {rule['source_file']} | {rule['clause_id']}\n"
    observation_text += f"      内容: {rule['clause_text'][:300]}...\n\n"
```

#### 3. 多层约束

- **Prompt层**：明确要求只能引用检索结果
- **Observation层**：优先展示完整条文，末尾再次强调
- **结果层**：可在后处理中验证引用的法规是否在检索结果中

### 效果

通过这些机制，Agent能够：
- 自动适应法规文件变动
- 只引用实际检索到的法规
- 大幅降低引用幻觉的风险

---

## Fallback机制

### 什么是Fallback？

Fallback是当主要方法失败时的备用方案，保证Agent的鲁棒性。

### Fallback层次

```
┌─────────────────────────────────────────────────────┐
│                   Fallback层次                        │
├─────────────────────────────────────────────────────┤
│                                                     │
│  Level 1: 工具级Fallback                             │
│  ┌─────────────────────────────────────────────┐   │
│  │ audit_text (增强版)                          │   │
│  │     ↓ 失败                                   │   │
│  │ audit_text (基础版)                          │   │
│  └─────────────────────────────────────────────┘   │
│                                                     │
│  Level 2: 检索级Fallback                             │
│  ┌─────────────────────────────────────────────┐   │
│  │ Milvus向量检索                               │   │
│  │     ↓ 失败                                   │   │
│  │ NumPy向量检索                                │   │
│  └─────────────────────────────────────────────┘   │
│                                                     │
│  Level 3: 模型级Fallback                             │
│  ┌─────────────────────────────────────────────┐   │
│  │ qwen-plus (主力模型)                         │   │
│  │     ↓ 失败/超限                               │   │
│  │ qwen-turbo (备用模型)                        │   │
│  └─────────────────────────────────────────────┘   │
│                                                     │
│  Level 4: Agent级Fallback                            │
│  ┌─────────────────────────────────────────────┐   │
│  │ ReAct Agent                                  │   │
│  │     ↓ 多次失败                                │   │
│  │ 直接调用审核函数                              │   │
│  └─────────────────────────────────────────────┘   │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### 重试机制

```python
class ReActAgent:
    def run(self, query: str) -> ReActResult:
        retry_count = 0

        while step_count < self.max_steps:
            # 执行工具
            result = tool.execute(**params)

            if not result.success:
                retry_count += 1
                if retry_count >= self.max_retries:
                    return ReActResult(
                        success=False,
                        error="Max retries exceeded",
                    )
                # 尝试其他工具
                continue
```

---

## 使用示例

### 基础使用

```python
from aliyun_rag.bailian_client import BailianClient
from aliyun_rag.config import Settings
from aliyun_rag.knowledge_base_milvus import load_knowledge_base
from aliyun_rag.agent import InsuranceAuditAgent, AgentConfig
from aliyun_rag.agent.tools import create_audit_tools

# 初始化
settings = Settings.from_env()
client = BailianClient(settings)
kb = load_knowledge_base('kb_milvus')

# 创建工具和Agent
tools = create_audit_tools(kb, client)
agent = InsuranceAuditAgent(client, tools, kb=kb)  # 传入kb以启用动态法规获取

# 文本审核
result = agent.audit("本保险保证年化收益8%")
print(result.answer)
print(result.audit_result['is_compliant'])  # 'no'
```

### 对话模式

```python
# 多轮对话
response1 = agent.chat("帮我审核：本保险保本保收益")
print(response1)

response2 = agent.chat("具体违反了哪些规定？")
print(response2)
```

### 导出轨迹

```python
# 审核并导出执行轨迹
result = agent.audit("限时抢购，最后机会！")
agent.export_trace(result, "trace.txt")

# 轨迹包含完整的思考过程、工具调用、结果观察
```

---

## 实现细节

### 数据流

```
用户输入
   │
   ▼
┌─────────────┐
│意图路由     │  ← 检测输入类型（文本/图片/文档）
└─────────────┘
   │
   ▼
┌─────────────┐
│ReAct循环    │  ← 推理-行动-观察
└─────────────┘
   │
   ├─▶ analyze_intent()    → 意图分析结果
   ├─▶ audit_text()        → 审核结果
   ├─▶ search_knowledge()  → 相关法规
   └─▶ explain_rule()      → 法规解释
   │
   ▼
┌─────────────┐
│结果聚合     │  ← 整合所有信息
└─────────────┘
   │
   ▼
最终答案
```

### 异常处理

```python
# 工具执行异常
try:
    result = tool.execute(**params)
except Exception as e:
    # 记录错误
    logger.error(f"Tool execution failed: {e}")
    # 触发Fallback
    if fallback_available:
        result = fallback_tool.execute(**params)
    else:
        raise
```

### 可观测性

```python
@dataclass
class ReActStep:
    thought: Optional[Thought]
    action: Optional[Action]
    observation: Optional[Observation]

# 完整记录每个步骤
# 便于调试和分析
```

---

## 扩展指南

### 添加新工具

```python
# 1. 定义工具函数
def my_custom_tool(param1: str, param2: int) -> dict:
    # 实现逻辑
    return {"result": "..."}

# 2. 注册工具
registry.register(Tool(
    name="my_custom_tool",
    description="工具描述",
    parameters=[
        ToolParameter("param1", "string", "参数1描述", required=True),
        ToolParameter("param2", "number", "参数2描述", required=True),
    ],
    func=my_custom_tool,
    return_description="返回值描述",
))
```

### 添加新Agent类型

```python
class MyCustomAgent:
    def __init__(self, client, tools, config, kb=None):
        self.client = client
        self.tools = tools
        self.config = config
        self.kb = kb

    def run(self, query: str) -> AgentResult:
        # 实现自定义逻辑
        pass
```

### 集成其他LLM

```python
# 实现统一的客户端接口
class MyLLMClient:
    def chat(self, messages, **kwargs) -> str:
        # 调用其他LLM API
        pass

    def chat_with_functions(self, messages, functions, **kwargs):
        # 实现Function Calling
        pass
```

---

## 总结

本Agent实现的核心特点：

1. **ReAct框架**：推理与行动相结合，更智能的决策
2. **模块化设计**：工具可独立开发和测试
3. **鲁棒性**：多层Fallback机制
4. **可观测性**：完整的执行轨迹
5. **可扩展性**：易于添加新工具和功能
6. **动态适应性**：自动适应法规文件变动
7. **防幻觉机制**：多层约束确保引用准确性

通过这种设计，Agent能够自主完成复杂的审核任务，同时保持良好的可控性和可维护性。
