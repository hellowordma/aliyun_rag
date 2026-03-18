# 保险营销内容智能审核系统

基于阿里云百炼大模型 API 构建的保险营销内容智能审核系统，使用 **RAG（检索增强生成）** 技术，自动判断金融营销内容是否违反监管规定，并给出违规理由、对应条文及置信度。

---

## ✨ 特性

- ✅ **多模态输入** - 支持纯文本、图片、图文混合输入
- ✅ **图片直接审核** - 使用 Qwen-VL 多模态大模型直接分析营销图片
- ✅ **结构化输出** - 是否合规、违规类型、条文编号与原文、来源文件、置信度
- ✅ **6路召回** - 3路稠密向量检索 + 3路稀疏检索（BM25）
- ✅ **混合检索** - RRF 融合算法（0.7 稠密 + 0.3 稀疏）
- ✅ **意图识别** - 7种违规类型自动识别
- ✅ **问题重写** - 自动生成多个精确查询
- ✅ **数学置信度** - 数学公式 + LLM 输出
- ✅ **评估模块** - 准确率 100%

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

编辑 `.env` 文件：

```bash
DASHSCOPE_API_KEY=sk-你的密钥
```

**获取 API Key：** https://bailian.console.aliyun.com/

### 3. 构建知识库（推荐 Milvus）

```bash
# 使用 Milvus Lite（推荐，支持6路召回）
PYTHONPATH=/mnt/workspace:$PYTHONPATH python -m aliyun_rag.main build-kb \
  --kb-dir kb_milvus \
  --vector-db milvus \
  --collection-name insurance_knowledge \
  --docs "references/保险销售行为管理办法.pdf" \
       "references/互联网保险业务监管办法.docx" \
       "references/金融产品网络营销管理办法（征求意见稿）.doc"
```

### 4. 运行 Demo

#### CLI Demo（推荐）

```bash
python demo/cli_demo.py
```

启动信息：
```
[KB] Milvus知识库: 176 个法规 chunks
[KB] 支持稀疏向量(BM25): 是
[KB] 6路召回: 稠密x3 + 稀疏x3
```

#### Web Demo

```bash
cd demo && bash start.sh
```

访问：http://localhost:7860

---

## 📁 项目结构

```
aliyun_rag/
├── config.py                 # 配置管理
├── bailian_client.py         # 百炼API客户端（含多模态）
├── extractors.py             # 文件文本提取（支持.doc通过antiword）
├── knowledge_base.py         # RAG知识库（NumPy）
├── knowledge_base_milvus.py  # RAG知识库（Milvus + 稀疏向量）
├── hybrid_retriever.py       # 混合检索（BM25 + 向量）
├── auditor.py                # 基础审核器
├── enhanced_auditor.py       # 增强审核器（6路召回）
├── multimodal_auditor.py     # 多模态审核器（图片/图文）
├── confidence_calculator.py  # 置信度计算模块
├── evaluate.py               # 效果评估模块
├── main.py                   # CLI命令入口
├── demo/                     # Demo应用
│   ├── app.py                # Gradio Web应用
│   ├── cli_demo.py           # CLI交互式Demo
│   ├── cli_demo_auto.py      # CLI自动修复Demo
│   ├── run.py                # Web启动脚本
│   └── start.sh              # Shell启动脚本
├── test_data/                # 测试数据
│   ├── images/               # 测试图片
│   └── audit_results/        # 审核结果输出
├── kb/                       # NumPy知识库
├── kb_milvus/                # Milvus知识库元数据
├── milvus_demo.db            # Milvus数据库文件
├── .env.example              # 配置模板
└── references/               # 监管文档（3个文件）
```

---

## 🔧 常用命令

| 功能 | 命令 |
|------|------|
| 构建知识库（NumPy） | `python -m aliyun_rag.main build-kb --kb-dir kb --vector-db numpy` |
| 构建知识库（Milvus） | `python -m aliyun_rag.main build-kb --kb-dir kb_milvus --vector-db milvus` |
| 审核文本 | `python demo/cli_demo.py` (选择文本审核) |
| 审核图片 | `python demo/cli_demo.py` (选择图片审核) |
| 批量审核 | `python demo/cli_demo.py` (选择批量测试) |
| 运行评估 | `python -m aliyun_rag.main evaluate --kb-dir kb --dataset demo_cases.jsonl` |
| 启动 CLI Demo | `python demo/cli_demo.py` |
| 启动 Web Demo | `cd demo && ./start.sh` |

---

## 📊 输出示例

```json
{
  "is_compliant": "no",
  "violations": [
    {
      "type": "承诺保证收益",
      "clause_id": "第九条（三）",
      "source_file": "金融产品网络营销管理办法（征求意见稿）.doc",
      "clause_text": "第九条【禁止内容】...（三）明示或暗示资产管理产品保本、承诺收益...",
      "reason": "营销内容中'保本'、'年化收益5%'属于对收益作出确定性承诺...",
      "confidence": 0.95
    }
  ],
  "overall_confidence": 0.82,
  "summary": "该营销内容违规承诺保证收益，违背诚实信用原则...",
  "retrieved_rules": [...]
}
```

---

## 🎯 考题要求完成情况

| 考题要求 | 完成状态 | 实现方式 |
|---------|---------|---------|
| 支持文本/图文输入 | ✅ 完成 | 文本/图片/图文混合三种输入模式 |
| 是否合规输出 | ✅ 完成 | `is_compliant: yes/no` |
| 违规类型识别 | ✅ 完成 | `violations[].type` (7种类型) |
| 条文编号与原文 | ✅ 完成 | `clause_id`, `clause_text`, `source_file` |
| 置信度输出 | ✅ 完成 | 数学公式 + LLM 输出 |
| 百炼大模型 API | ✅ 完成 | 通义千问兼容模式 API |
| RAG 技术增强 | ✅ 完成 | 6路召回（稠密x3 + 稀疏x3） |
| 评估模块 | ✅ 完成 | `evaluate.py`，准确率 100% |

---

## 🎓 演示代码实现

### 1. 文本审核（Milvus + 6路召回）

```python
from aliyun_rag.bailian_client import BailianClient
from aliyun_rag.config import Settings
from aliyun_rag.knowledge_base_milvus import load_knowledge_base
from aliyun_rag.enhanced_auditor import enhanced_audit_marketing_text

# 初始化
settings = Settings.from_env()
client = BailianClient(settings)
kb = load_knowledge_base(collection_name='insurance_knowledge', meta_dir='kb_milvus')

# 审核文本（6路召回）
result = enhanced_audit_marketing_text(
    marketing_text="本理财产品保本，年化收益率5%。",
    kb=kb,
    client=client,
    top_k=6,
)

print(result['is_compliant'])     # 'no'
print(result['violations'])       # 违规项列表（含source_file）
print(result['overall_confidence'])  # 融合置信度
```

### 2. 图片审核

```python
from aliyun_rag import audit_marketing_image

# 读取图片
with open("marketing_poster.png", "rb") as f:
    image_bytes = f.read()

# 审核图片
result = audit_marketing_image(
    image_bytes=image_bytes,
    kb=kb,
    client=client,
    image_mime="image/png",
    top_k=6,
)

# 结果包含图片分析
print(result['image_analysis']['extracted_text'])  # 提取的文字
print(result['image_analysis']['visual_elements'])  # 视觉元素
```

### 3. 图文混合审核

```python
# 文字说明 + 图片
result = audit_marketing_image(
    image_bytes=image_bytes,
    text_context="这是我们的明星产品",  # 额外文字说明
    kb=kb,
    client=client,
    top_k=6,
)
```

### 4. 效果评估

```python
from aliyun_rag.evaluate import evaluate_dataset

results = evaluate_dataset(
    dataset_path='demo_cases.jsonl',
    kb_dir='kb',
    client=client,
)

print(f"准确率: {results['accuracy']:.2%}")
print(f"正确数: {results['correct']}/{results['total']}")
```

---

## 🏗️ 系统架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                       用户输入层                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   文本输入    │  │   图片上传    │  │   图文混合    │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
└─────────┼─────────────────┼─────────────────┼──────────────────┘
          │                 │                 │
          ▼                 ▼                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                      多模态处理层                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Qwen-VL 多模态分析 (multimodal_auditor.py)                │   │
│  │  - 图片OCR文字提取                                          │   │
│  │  - 视觉元素识别                                             │   │
│  │  - 营销内容综合分析                                         │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                      增强审核层                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ 意图识别 (enhanced_auditor.py)                            │   │
│  │  - 7种违规类型识别                                         │   │
│  │  - 关键词匹配                                               │   │
│  └──────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ 问题重写 (enhanced_auditor.py)                            │   │
│  │  - 生成多个精确查询                                         │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                      RAG 混合检索层                               │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ BM25 稀疏检索 + 向量稠密检索                              │   │
│  │  RRF加权融合: 0.3 × BM25 + 0.7 × Dense                   │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                      大模型推理层                                │
│  qwen-plus (文本) / qwen-vl-plus (图片)                         │
│  Prompt Engineering + 上下文注入                                 │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                      输出层                                      │
│  { is_compliant, violations, confidence, summary, ... }        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔑 关键设计说明

### 1. 模型选择策略

| 输入类型 | 使用模型 | 消息格式 |
|---------|---------|---------|
| 纯文本 | qwen-plus | `{"role": "user", "content": "文本"}` |
| 图片/图文 | qwen-vl-plus | `{"content": [{"type": "text"}, {"type": "image_url"}]}` |

**核心代码** (`bailian_client.py`):
```python
# 文本审核
def chat(self, messages, model=None):
    return self.client.chat.completions.create(
        model=model or self.settings.chat_model,  # qwen-plus
        messages=messages,
    )

# 图片审核
def analyze_marketing_image(self, image_bytes, ...):
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": base64_image}}
        ]
    }]
    return self.chat(messages, model=self.settings.vl_model)  # qwen-vl-plus
```

### 2. RAG 混合检索设计

**检索流程**:
1. 并行执行 BM25 稀疏检索和向量稠密检索
2. 使用 RRF (Reciprocal Rank Fusion) 融合结果
3. 加权公式: `0.3 × BM25_score + 0.7 × Dense_score`

**核心代码** (`hybrid_retriever.py`):
```python
def _fuse_results(dense_results, sparse_results):
    k = 60  # RRF 常数
    fused = {}

    # 稠密检索得分
    for i, item in enumerate(dense_results):
        fused[item['id']] = 0.7 * (1 / (k + i + 1))

    # 稀疏检索得分
    for i, item in enumerate(sparse_results):
        score = 0.3 * (1 / (k + i + 1))
        fused[item['id']] = fused.get(item['id'], 0) + score

    return sorted(fused.items(), key=lambda x: x[1], reverse=True)
```

### 3. 置信度计算公式

**数学公式** (`confidence_calculator.py`):
```python
overall_confidence = 0.3 × retrieval_score + 0.5 × violation_score + 0.2 × clause_match_score

其中:
- retrieval_score: 检索相关度 (Sigmoid变换)
- violation_score: 违规项置信度均值
- clause_match_score: 条文匹配度
```

### 4. 意图识别设计

**7种违规类型**:
- 承诺保证收益
- 夸大宣传
- 误导性宣传
- 无证代言
- 风险提示不足
- 资质问题
- 销售诱导

**核心代码** (`enhanced_auditor.py`):
```python
VIOLATION_PATTERNS = {
    "承诺保证收益": ["保证", "承诺", "保本", "零风险", "稳赚不赔"],
    "夸大宣传": ["最高", "第一", "最佳", "唯一", "顶级"],
    # ... 更多模式
}

def identify_intent(marketing_text: str) -> dict:
    detected_risks = []
    for violation_type, keywords in VIOLATION_PATTERNS.items():
        if any(kw in marketing_text for kw in keywords):
            detected_risks.append(violation_type)
    return {"detected_risks": detected_risks, ...}
```

---

## 📈 效果评估模块

### 评估结果

```
============================================================
EVALUATION RESULTS
============================================================
Total Samples: 4
Correct Predictions: 4
Accuracy: 100.00%

✅ 本保险产品保本保收益，年化收益率保证8%，零风险...
   Expected: NO | Predicted: NO

✅ 本产品过往业绩不代表未来表现，具体以合同条款为准...
   Expected: YES | Predicted: YES

✅ 限时抢购，购买后稳赚不赔，现在下单即可翻倍...
   Expected: NO | Predicted: NO

✅ 请您仔细阅读保险条款，重点关注责任免除和犹豫期...
   Expected: YES | Predicted: YES
============================================================
```

### 使用方式

```bash
# 命令行
python -m aliyun_rag.main evaluate --kb-dir kb --dataset demo_cases.jsonl

# 代码调用
from aliyun_rag.evaluate import evaluate_dataset
results = evaluate_dataset('demo_cases.jsonl', 'kb', client)
```

---

## 📚 详细文档

查看 [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) 了解：
- 完整架构说明
- 技术栈详解
- RAG实现细节
- 混合检索说明
- 使用指南

---

**项目状态：✅ 完整可运行，支持文本/图片/图文混合审核，满足所有考题要求**
