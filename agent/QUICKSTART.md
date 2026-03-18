# Agent模块快速入门指南

## 概述

本Agent模块为aliyun_rag项目提供了基于ReAct框架的智能审核Agent实现。

## 文件结构

```
agent/
├── __init__.py                 # 模块入口，导出主要类和函数
├── tools.py                    # 工具定义和注册表
├── react_agent.py              # ReAct框架核心实现
├── insurance_audit_agent.py    # 保险审核Agent
├── demo.py                     # 使用演示
├── README.md                   # 详细的设计文档
└── QUICKSTART.md               # 本文件
```

## 快速开始

### 1. 初始化Agent

```python
from aliyun_rag.bailian_client import BailianClient
from aliyun_rag.config import Settings
from aliyun_rag.knowledge_base_milvus import load_knowledge_base
from aliyun_rag.agent import InsuranceAuditAgent, AgentConfig
from aliyun_rag.agent.tools import create_audit_tools

# 加载配置和资源
settings = Settings.from_env()
client = BailianClient(settings)
kb = load_knowledge_base('kb_milvus')

# 创建工具和Agent
tools = create_audit_tools(kb, client)
agent = InsuranceAuditAgent(client, tools)
```

### 2. 文本审核

```python
# 简单文本审核
result = agent.audit("本保险保证年化收益8%")

print(f"合规状态: {result.audit_result.get('is_compliant')}")
print(f"审核答案: {result.answer}")
print(f"执行步骤: {len(result.steps)}步")
```

### 3. 图片审核

```python
from pathlib import Path

# 图片审核
result = agent.audit(Path("path/to/image.png"), content_type="image")
print(result.answer)
```

### 4. 对话模式

```python
# 多轮对话
response = agent.chat("帮我审核这条文案是否合规")
print(response)

response = agent.chat("具体违反了哪条规定？")
print(response)
```

### 5. 导出执行轨迹

```python
# 审核并导出详细轨迹
result = agent.audit("限时抢购，错过不再！")
agent.export_trace(result, "audit_trace.txt")
```

## 可用工具

| 工具名 | 功能 | 参数 |
|--------|------|------|
| `audit_text` | 文本审核 | marketing_text, use_enhanced |
| `analyze_intent` | 意图分析 | marketing_text |
| `audit_image` | 图片审核 | image_path, text_context |
| `audit_pdf` | PDF审核 | file_path, max_pages |
| `search_knowledge` | 法规检索 | query, top_k |
| `explain_rule` | 条文解释 | rule_text |
| `batch_audit` | 批量审核 | texts |

## 运行演示

```bash
cd /mnt/workspace/aliyun_rag
python -m agent.demo
```

演示菜单：
1. 文本审核演示
2. 图片审核演示
3. 对话模式演示
4. 轨迹导出演示
5. Fallback机制演示
6. 批量审核演示
7. 全部演示
8. 交互式Demo

## 配置选项

```python
from aliyun_rag.agent import AgentConfig

config = AgentConfig(
    max_steps=10,           # 最大推理步骤
    max_retries=3,          # 工具失败重试次数
    enable_fallback=True,   # 启用Fallback机制
    enable_intent_routing=True,  # 启用意图路由
    verbose=True,           # 详细日志输出
)
```

## 核心概念

### ReAct框架

ReAct = Reasoning (推理) + Acting (行动)

```
用户问题
  ↓
[Thought] 思考需要做什么
  ↓
[Action] 选择并执行工具
  ↓
[Observation] 观察工具结果
  ↓
重复 → 直到得到答案
```

### Fallback机制

多层保护确保Agent鲁棒性：

1. **工具级**: 增强版失败 → 基础版
2. **检索级**: Milvus失败 → NumPy
3. **模型级**: 主模型失败 → 备用模型
4. **Agent级**: ReAct失败 → 直接调用

### 执行轨迹

每次审核都会记录完整轨迹：
- 每一步的思考过程
- 工具调用和参数
- 工具返回结果
- 执行时间统计

## 扩展开发

### 添加新工具

```python
from aliyun_rag.agent.tools import Tool, ToolParameter

def my_tool(param: str) -> dict:
    return {"result": "..."}

# 注册工具
tools.register(Tool(
    name="my_tool",
    description="工具描述",
    parameters=[
        ToolParameter("param", "string", "参数描述", required=True),
    ],
    func=my_tool,
    return_description="返回值描述",
))
```

### 自定义Agent

```python
from aliyun_rag.agent.react_agent import ReActAgent

class MyAgent(ReActAgent):
    def _default_system_prompt(self) -> str:
        return "自定义系统提示词"

    def run(self, query: str):
        # 自定义逻辑
        pass
```

## 参考文档

详细设计请参阅 `agent/README.md`，包含：
- 完整架构图
- ReAct框架原理
- 工具调用机制详解
- Fallback机制设计
- 实现细节和扩展指南

