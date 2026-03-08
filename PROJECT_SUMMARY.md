# 保险营销内容智能审核系统 - 完整项目文档

## 📋 项目概述

基于阿里云百炼大模型 API 构建的保险营销内容智能审核系统，使用 **RAG（检索增强生成）** 技术，自动判断金融营销内容是否违反监管规定，并给出违规理由、对应条文及置信度。

---

## 🎯 考题要求完成情况

| 考题要求 | 完成状态 | 实现方式 |
|---------|---------|---------|
| 支持文本/图文输入 | ✅ 完成 | 文本/图片/图文混合三种输入模式 |
| 是否合规输出 | ✅ 完成 | `is_compliant: yes/no` |
| 违规类型识别 | ✅ 完成 | `violations[].type` (7种类型) |
| 条文编号与原文 | ✅ 完成 | `clause_id`, `clause_text` |
| 置信度输出 | ✅ 完成 | 数学公式 + LLM 输出 |
| 百炼大模型 API | ✅ 完成 | 通义千问兼容模式 API |
| RAG 技术增强 | ✅ 完成 | 向量嵌入 + 相似度检索 + BM25 |
| 评估模块 | ✅ 完成 | `evaluate.py`，准确率 100% |

---

## 🏗️ 系统架构图

### 整体架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                         用户输入层                                   │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐         │
│  │    文本输入     │  │    图片上传     │  │   图文混合      │         │
│  │  "本保险保证..."  │  │  poster.png    │  │  图+文字说明    │         │
│  └────────┬───────┘  └────────┬───────┘  └────────┬───────┘         │
└───────────┼──────────────────┼──────────────────┼───────────────────┘
            │                  │                  │
            ▼                  ▼                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      输入类型路由层                                   │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  audit_marketing_multimodal(content, content_type)           │   │
│  │                                                             │   │
│  │  if content_type == "text":                                 │   │
│  │      → audit_marketing_text()  ──────→  qwen-plus          │   │
│  │  elif content_type == "image":                              │   │
│  │      → audit_marketing_image()  ─────→  qwen-vl-plus       │   │
│  │  elif content_type == "multimodal":                         │   │
│  │      → audit_marketing_image(text_context=...)  → qwen-vl  │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    多模态内容分析层                                    │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  图片处理路径 (multimodal_auditor.py)                        │   │
│  │                                                             │   │
│  │  1. analyze_marketing_image()                               │   │
│  │     - 图片转 base64 URL                                      │   │
│  │     - 调用 qwen-vl-plus 提取:                                │   │
│  │       * extracted_text (图片文字)                            │   │
│  │       * visual_elements (视觉元素)                           │   │
│  │       * marketing_content (营销内容)                         │   │
│  │                                                             │   │
│  │  2. 基于提取内容进行向量检索                                  │   │
│  │                                                             │   │
│  │  3. 多模态合规分析 (图文+规则)                               │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      增强审核层                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  意图识别 (enhanced_auditor.py)                             │   │
│  │  ┌────────────────────────────────────────────────────────┐  │   │
│  │  │ VIOLATION_PATTERNS = {                                  │  │   │
│  │  │   "承诺保证收益": ["保证", "承诺", "保本", ...],         │  │   │
│  │  │   "夸大宣传": ["最高", "第一", "最佳", ...],             │  │   │
│  │  │   "误导性宣传": ["稳赚不赔", "翻倍", ...],               │  │   │
│  │  │   ... (共7种类型)                                        │  │   │
│  │  │ }                                                        │  │   │
│  │  └────────────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  问题重写 (Query Rewriting)                                  │   │
│  │  根据检测到的违规类型，生成针对性检索查询                      │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   RAG 混合检索层                                       │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                      并行检索                                  │   │
│  │  ┌─────────────────────┐    ┌─────────────────────┐         │   │
│  │  │  BM25 稀疏检索       │    │  向量稠密检索        │         │   │
│  │  │  - jieba 分词        │    │  - text-embedding   │         │   │
│  │  │  - 关键词匹配        │    │  - 余弦相似度        │         │   │
│  │  │  - Top-K 结果        │    │  - Top-K 结果        │         │   │
│  │  └──────────┬──────────┘    └──────────┬──────────┘         │   │
│  │             │                           │                     │   │
│  │             └───────────┬───────────────┘                     │   │
│  │                         ▼                                    │   │
│  │              ┌─────────────────────┐                         │   │
│  │              │   RRF 融合算法      │                         │   │
│  │              │  0.3 × BM25         │                         │   │
│  │              │  + 0.7 × Dense      │                         │   │
│  │              └─────────────────────┘                         │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    大模型推理层                                        │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Prompt Engineering                                           │   │
│  │  ┌────────────────────────────────────────────────────────┐  │   │
│  │  │ System: "你是金融保险营销合规审核助手..."              │  │   │
│  │  │ User: 营销内容 + 检索到的法规条文                       │  │   │
│  │  │ Output: JSON格式 {is_compliant, violations, ...}       │  │   │
│  │  └────────────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  模型选择:                                                           │
│  - 文本输入 → qwen-plus                                            │
│  - 图片/图文 → qwen-vl-plus                                        │
└─────────────────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    置信度计算层                                        │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  confidence_calculator.py                                    │   │
│  │                                                             │   │
│  │  overall_confidence =                                       │   │
│  │    0.3 × retrieval_score      (检索相关度)                 │   │
│  │    + 0.5 × violation_score     (违规置信度)                 │   │
│  │    + 0.2 × clause_match_score (条文匹配度)                 │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      输出层                                          │
│  {                                                                 │
│    "is_compliant": "yes/no",                                       │
│    "violations": [...],                                           │
│    "overall_confidence": 0.97,                                    │
│    "summary": "...",                                              │
│    "retrieved_rules": [...],                                      │
│    "image_analysis": {...}  # 图片审核时包含                      │
│  }                                                                 │
└─────────────────────────────────────────────────────────────────────┘
```

### 数据流图

```
┌──────────────┐
│  用户输入    │
└──────┬───────┘
       │
       ▼
┌─────────────────────────────────────────┐
│  判断输入类型                            │
│  - isinstance(str) → 文本                │
│  - isinstance(bytes) → 图片              │
│  - isinstance(tuple) → 图文混合          │
└───────┬─────────────────────────────────┘
        │
   ┌────┴────┬────────────┐
   ▼         ▼             ▼
文本       图片         图文混合
   │         │             │
   ▼         ▼             ▼
┌────────┐ ┌────────┐ ┌────────────────┐
│chat()  │ │analyze│ │analyze_image()  │
│qwen-   │ │_image()│ │+text_context    │
│plus    │ │qwen-vl │ │qwen-vl-plus     │
└───┬────┘ └───┬────┘ └────┬───────────┘
    │          │            │
    └──────┬───┴────────────┘
           ▼
    ┌──────────────┐
    │  RAG检索     │
    │  (KB: 137)   │
    └──────┬───────┘
           ▼
    ┌──────────────┐
    │ LLM合规分析  │
    └──────┬───────┘
           ▼
    ┌──────────────┐
    │  置信度计算  │
    └──────┬───────┘
           ▼
    ┌──────────────┐
    │  输出结果    │
    └──────────────┘
```

---

## 📁 核心代码结构

```
aliyun_rag/
├── __init__.py               # 模块导出
├── config.py                 # 配置管理（环境变量、.env支持）
│   └── Settings             # 数据类，包含API配置、模型选择
│
├── bailian_client.py         # 百炼API客户端封装
│   ├── BailianClient        # 客户端类
│   ├── chat()               # 文本对话 (qwen-plus)
│   ├── embed_texts()        # 文本向量化 (text-embedding-v3)
│   ├── ocr_image_bytes()    # 图片OCR (qwen-vl-plus)
│   └── analyze_marketing_image()  # 营销图片分析
│
├── extractors.py             # 文件文本提取
│   ├── extract_text_from_file()  # 统一入口
│   ├── extract_text_from_pdf()   # PDF提取
│   ├── extract_text_from_pdf_vl_ocr()  # PDF OCR提取
│   ├── extract_text_from_docx()  # DOCX提取
│   └── extract_text_from_markdown()  # Markdown提取
│
├── knowledge_base.py         # RAG知识库（NumPy）
│   ├── RuleChunk            # 条文数据类
│   ├── KnowledgeBase        # 知识库类
│   ├── split_into_rule_chunks()  # 智能分块
│   ├── build_knowledge_base()    # 构建知识库
│   ├── load_knowledge_base()     # 加载知识库
│   └── retrieve_relevant_rules() # 向量检索
│
├── knowledge_base_milvus.py  # RAG知识库（Milvus）
│   └── MilvusKnowledgeBase  # Milvus实现
│
├── hybrid_retriever.py       # 混合检索（BM25 + 向量）
│   ├── HybridRetriever      # 混合检索器
│   ├── create_hybrid_retriever()  # 创建检索器
│   ├── hybrid_retrieve_rules()    # 混合检索
│   └── _fuse_results()       # RRF融合
│
├── auditor.py                # 基础审核器
│   ├── audit_marketing_text()  # 文本审核主函数
│   ├── _extract_json_block()   # JSON解析
│   └── _build_rule_context()   # 规则上下文构建
│
├── enhanced_auditor.py       # 增强审核器
│   ├── VIOLATION_PATTERNS    # 违规模式字典
│   ├── identify_intent()     # 意图识别
│   ├── rewrite_query()       # 问题重写
│   ├── multi_stage_audit()   # 多阶段审核
│   └── enhanced_audit_marketing_text()  # 增强审核入口
│
├── multimodal_auditor.py     # 多模态审核器 (NEW)
│   ├── audit_marketing_image()      # 图片审核
│   ├── audit_marketing_multimodal()  # 统一入口
│   └── _extract_json_block()         # JSON解析
│
├── confidence_calculator.py  # 置信度计算模块
│   ├── ConfidenceCalculator  # 计算器类
│   ├── calculate_overall_confidence()  # 整体置信度
│   ├── _calculate_retrieval_score()    # 检索相关度
│   ├── _calculate_violation_score()    # 违规置信度
│   └── _calculate_clause_match_score() # 条文匹配度
│
├── evaluate.py               # 效果评估模块
│   └── evaluate_dataset()    # 数据集评估
│
├── main.py                   # CLI命令入口
│   ├── cmd_build_kb()        # 构建知识库命令
│   ├── cmd_audit_text()      # 审核文本命令
│   ├── cmd_audit_file()      # 审核文件命令
│   └── cmd_evaluate()        # 评估命令
│
├── batch_audit.py            # 批量审核脚本
│
├── demo/                     # Demo应用
│   ├── app.py                # Gradio Web应用
│   ├── cli_demo.py           # CLI交互式Demo
│   ├── cli_demo_auto.py      # CLI自动修复Demo
│   ├── run.py                # Web启动脚本
│   └── start.sh              # Shell启动脚本
│
├── test_data/                # 测试数据
│   ├── images/               # 测试图片
│   │   ├── violation_exaggerated_return.png
│   │   ├── violation_misleading.png
│   │   └── compliant_risk_warning.png
│   ├── create_test_images.py # 测试图片生成脚本
│   └── audit_results/        # 审核结果输出
│
├── requirements.txt          # 依赖清单
├── .env                      # API密钥配置（本地）
├── .env.example              # 配置模板
├── demo_cases.jsonl          # 测试数据集
├── kb/                       # NumPy知识库（自动生成）
│   ├── chunks.json           # 条文数据
│   └── embeddings.npy        # 向量数据
├── kb_milvus/                # Milvus知识库元数据（自动生成）
├── milvus_demo.db            # Milvus数据库文件（自动生成）
└── references/               # 监管文档
    ├── 保险销售行为管理办法.pdf
    ├── 互联网保险业务监管办法.docx
    └── 金融产品网络营销管理办法（征求意见稿）.doc
```

---

## 🔧 技术栈

| 组件 | 技术选型 | 说明 |
|------|---------|------|
| 大模型 API | 阿里云百炼（通义千问） | 兼容 OpenAI SDK |
| 聊天模型 | qwen-plus | 用于合规判定 |
| 视觉模型 | qwen-vl-plus | 用于图片审核/OCR |
| 嵌入模型 | text-embedding-v3 | 用于向量检索 (1024维) |
| 向量数据库 | Milvus Lite / NumPy | 稠密向量检索 |
| 稀疏检索 | BM25 + jieba | 关键词精确匹配 |
| 检索算法 | 混合检索（RRF融合） | BM25 + 向量加权融合 |
| 文件处理 | docx2txt, PyMuPDF | 多格式支持 |
| 配置管理 | python-dotenv | 环境变量管理 |
| Web界面 | Gradio 4.0+ | 友好的交互界面 |

---

## 🔑 关键设计说明

### 1. 模型选择策略

系统根据输入类型自动选择合适的模型：

```python
# bailian_client.py 中的关键设计

class BailianClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        # 配置了两个模型
        # settings.chat_model = "qwen-plus"      # 文本审核
        # settings.vl_model = "qwen-vl-plus"     # 图片审核

    def chat(self, messages, model=None):
        """文本对话 - 使用 qwen-plus"""
        return self.client.chat.completions.create(
            model=model or self.settings.chat_model,  # 默认 qwen-plus
            messages=messages,
        )

    def analyze_marketing_image(self, image_bytes, ...):
        """图片分析 - 使用 qwen-vl-plus"""
        # 构建多模态消息（包含图片）
        messages = [{
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": base64_image}}
            ]
        }]
        # 强制使用多模态模型
        return self.chat(messages, model=self.settings.vl_model)
```

**调用路径**:
```
文本输入 → audit_marketing_text() → client.chat() → qwen-plus
图片输入 → audit_marketing_image() → client.analyze_marketing_image() → qwen-vl-plus
```

### 2. RAG 混合检索设计

#### 检索流程

```python
# hybrid_retriever.py

def retrieve(query, kb, client, retriever, top_k):
    # 1. 并行执行两种检索
    dense_results = dense_retrieve(query, top_k=20)  # 向量检索
    sparse_results = sparse_retrieve(query, top_k=20)  # BM25检索

    # 2. RRF融合
    fused = _fuse_results(dense_results, sparse_results)

    # 3. 返回Top-K
    return fused[:top_k]

def _fuse_results(dense_results, sparse_results):
    k = 60  # RRF常数
    fused = {}

    # 向量检索权重: 0.7
    for i, item in enumerate(dense_results):
        fused[item['id']] = 0.7 * (1 / (k + i + 1))

    # BM25检索权重: 0.3
    for i, item in enumerate(sparse_results):
        score = 0.3 * (1 / (k + i + 1))
        fused[item['id']] = fused.get(item['id'], 0) + score

    return sorted(fused.items(), key=lambda x: x[1], reverse=True)
```

#### 权重配置

| 检索方式 | 权重 | 说明 |
|---------|------|------|
| 向量检索 | 0.7 | 语义相似，捕捉隐含关系 |
| BM25检索 | 0.3 | 精确匹配，专业术语准确 |

### 3. 置信度计算公式

```python
# confidence_calculator.py

def calculate_overall_confidence(retrieved_rules, violations, ...):
    """
    overall_confidence = 0.3 × retrieval_score
                       + 0.5 × violation_score
                       + 0.2 × clause_match_score
    """

    # 1. 检索相关度 (Sigmoid变换)
    retrieval_score = sigmoid(max_score * 10 - 5)

    # 2. 违规项置信度
    violation_score = weighted_average([
        v['confidence'] * (1 + len(v['reason']) / 100)
        for v in violations
    ]) if violations else 0

    # 3. 条文匹配度
    clause_match_score = calculate_clause_similarity(retrieved_rules, violations)

    # 加权融合
    return 0.3 * retrieval_score + 0.5 * violation_score + 0.2 * clause_match_score
```

**权重分配理由**:
- `violation_score (0.5)`: 违规项是核心判断依据
- `retrieval_score (0.3)`: 检索质量反映上下文相关性
- `clause_match_score (0.2)`: 条文匹配度补充验证

### 4. 意图识别设计

#### 7种违规类型

```python
# enhanced_auditor.py

VIOLATION_PATTERNS = {
    "承诺保证收益": ["保证", "承诺", "保本", "零风险", "稳赚不赔", "保本保息"],
    "夸大宣传": ["最高", "第一", "最佳", "唯一", "顶级", "最优秀", "极致"],
    "误导性宣传": ["稳赚不赔", "翻倍", "保本保息", "零风险", "无风险"],
    "无证代言": ["明星代言", "名人推荐", "网红推广", "KOL推荐"],
    "风险提示不足": ["高收益", "轻松赚钱", "躺着赚钱", "无风险"],
    "资质问题": ["独家", "特批", "内部渠道", "限量发售"],
    "销售诱导": ["限时", "抢购", "最后机会", "仅限今天", "错过等一年"],
}
```

#### 意图识别流程

```
输入文本
    │
    ▼
关键词匹配 → 检测到的违规类型列表
    │
    ▼
风险等级评估 → "高" / "中" / "低"
    │
    ▼
问题重写 → 生成针对性检索查询
```

### 5. 多模态审核设计

#### 图片审核流程

```python
# multimodal_auditor.py

def audit_marketing_image(image_bytes, kb, client, ...):
    # 步骤1: 使用多模态模型分析图片
    analysis = client.analyze_marketing_image(
        image_bytes=image_bytes,
        text_context=text_context,  # 可选
    )
    # 返回: {extracted_text, visual_elements, marketing_content}

    # 步骤2: 基于提取内容进行向量检索
    rules = retrieve_relevant_rules(
        query=analysis['marketing_content'],
        kb=kb,
        client=client,
        top_k=top_k,
    )

    # 步骤3: 多模态合规分析
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt_with_rules},
            {"type": "image_url", "image_url": {"url": image_base64}}
        ]
    }]
    result = client.chat(messages, model=client.settings.vl_model)

    return parsed_result
```

#### 输入类型路由

```python
def audit_marketing_multimodal(content, kb, client, content_type):
    """统一入口，根据类型路由"""
    if content_type == "text":
        return audit_marketing_text(content, kb, client)  # qwen-plus
    elif content_type == "image":
        return audit_marketing_image(content, kb, client)  # qwen-vl-plus
    elif content_type == "multimodal":
        text, image = content
        return audit_marketing_image(image, text_context=text, ...)  # qwen-vl-plus
```

---

## 📖 RAG 技术实现详解

### 1. Chunk 分块策略

**核心代码：** `knowledge_base.py:split_into_rule_chunks()`

| 参数 | 配置 | 说明 |
|------|------|------|
| 分块模式 | 条文级智能分块 | 识别"第X条"模式 |
| 最大长度 | 700字符 | 超长条文自动分割 |
| 最小长度 | 15字符 | 过小内容过滤 |
| 总Chunks | 137个 | 84种不同条文 |

**分块逻辑**:
```python
def split_into_rule_chunks(text: str, source_file: str):
    chunks = []
    # 按"第X条"分割
    sections = re.split(r'(第[一二三四五六七八九十百]+条)', text)

    for i in range(1, len(sections), 2):
        clause_id = sections[i]
        clause_text = sections[i+1] if i+1 < len(sections) else ""

        # 超长分割
        if len(clause_text) > 700:
            sub_chunks = split_long_text(clause_text)
            chunks.extend(sub_chunks)
        else:
            chunks.append(RuleChunk(clause_id, clause_text, source_file))

    return chunks
```

### 2. 向量嵌入

```python
# knowledge_base.py

def build_knowledge_base(documents, kb_dir, client):
    # 1. 分块
    chunks = split_into_rule_chunks(documents)

    # 2. 批量嵌入
    embeddings = client.embed_texts(
        [c.clause_text for c in chunks],
        batch_size=10,
    )

    # 3. 保存
    np.save(kb_dir / "embeddings.npy", embeddings)
    with open(kb_dir / "chunks.json", "w") as f:
        json.dump([asdict(c) for c in chunks], f)
```

### 3. 相似度检索

```python
# knowledge_base.py

def retrieve_relevant_rules(query, kb, client, top_k=6):
    # 1. 查询向量化
    query_embedding = client.embed_texts([query])[0]

    # 2. 余弦相似度
    similarities = np.dot(kb.embeddings, query_embedding) / (
        np.linalg.norm(kb.embeddings, axis=1) * np.linalg.norm(query_embedding)
    )

    # 3. Top-K
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    return [kb.chunks[i] for i in top_indices]
```

---

## 📈 效果评估模块

### 评估实现

**文件**: `evaluate.py`

```python
def evaluate_dataset(dataset_path, kb_dir, client, limit=None):
    """
    评估模型在测试集上的准确率

    Args:
        dataset_path: JSONL文件路径
        kb_dir: 知识库目录
        client: BailianClient实例
        limit: 限制评估数量

    Returns:
        {
            "total": 总样本数,
            "correct": 正确数,
            "accuracy": 准确率,
            "details": 详细结果
        }
    """
    # 1. 加载数据集
    rows = [json.loads(line) for line in Path(dataset_path).read_lines()]

    # 2. 逐个预测
    total, correct = 0, 0
    for row in rows:
        result = audit_marketing_text(row["text"], kb, client)
        pred = result["is_compliant"]
        expected = row["is_compliant"]

        if pred == expected:
            correct += 1
        total += 1

    # 3. 计算准确率
    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total,
        "details": [...]
    }
```

### 评估数据集

**文件**: `demo_cases.jsonl`

```jsonl
{"text":"本保险产品保本保收益，年化收益率保证8%，零风险。", "is_compliant":"no"}
{"text":"本产品过往业绩不代表未来表现，具体以合同条款为准。", "is_compliant":"yes"}
{"text":"限时抢购，购买后稳赚不赔，现在下单即可翻倍。", "is_compliant":"no"}
{"text":"请您仔细阅读保险条款，重点关注责任免除和犹豫期。", "is_compliant":"yes"}
```

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

### 使用评估模块

```bash
# 命令行
python -m aliyun_rag.main evaluate --kb-dir kb --dataset demo_cases.jsonl

# 代码调用
from aliyun_rag.evaluate import evaluate_dataset
results = evaluate_dataset('demo_cases.jsonl', 'kb', client)
print(f"Accuracy: {results['accuracy']:.2%}")
```

---

## 🖼️ 多模态图片审核功能

### 功能说明

系统支持直接审核营销图片，使用 Qwen-VL 多模态大模型：

1. **图片文字提取**：自动识别图片中的营销文案
2. **视觉元素分析**：识别图片中的代言人物、图表、Logo等
3. **合规性判断**：综合图文内容进行合规分析

### 核心文件

- `multimodal_auditor.py` - 多模态审核器
- `bailian_client.py` - 包含 `analyze_marketing_image()` 方法

### 使用方式

```python
from aliyun_rag.multimodal_auditor import audit_marketing_image

# 读取图片
with open("marketing_poster.png", "rb") as f:
    image_bytes = f.read()

# 审核图片
result = audit_marketing_image(
    image_bytes=image_bytes,
    kb=kb,
    client=client,
    image_mime="image/png",
    text_context=None,  # 可选的文字说明
    top_k=6,
)

# 结果包含
print(result['image_analysis']['extracted_text'])  # 提取的文字
print(result['image_analysis']['visual_elements'])  # 视觉元素
print(result['is_compliant'])  # 是否合规
```

### Demo界面

- **CLI Demo**: `python demo/cli_demo.py` (选择"图片审核"或"图文混合审核")
- **Web Demo**: `python demo/run.py` (选择"图片审核"Tab)

---

## 🎓 考题对应关系

| 考题要求 | 代码实现 |
|---------|---------|
| 使用百炼大模型 API | `bailian_client.py` |
| RAG 技术增强 | `knowledge_base.py`, `hybrid_retriever.py` |
| Prompt Engineering | `auditor.py`, `enhanced_auditor.py` |
| 输入文本/图文 | `multimodal_auditor.py`, `main.py` |
| 是否合规输出 | `auditor.py` - is_compliant |
| 违规类型 | `auditor.py` - violations[].type |
| 条文编号与原文 | `auditor.py` - clause_id, clause_text |
| 置信度 | `confidence_calculator.py` |
| 评估模块 | `evaluate.py` |

---

## 📞 获取 API Key

访问：https://bailian.console.aliyun.com/

---

**项目状态：✅ 完整可运行，支持文本/图片/图文混合审核，满足所有考题要求**
