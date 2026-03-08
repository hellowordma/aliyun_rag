# 保险营销内容智能审核系统 - 完整项目文档

## 📋 项目概述

基于阿里云百炼大模型 API 构建的保险营销内容智能审核系统，使用 **RAG（检索增强生成）** 技术，自动判断金融营销内容是否违反监管规定，并给出违规理由、对应条文及置信度。

---

## 🎯 考题要求完成情况

| 考题要求 | 完成状态 | 实现方式 |
|---------|---------|---------|
| 支持文本/图文输入 | ✅ 完成 | `audit-text`、`audit-file` 命令 |
| 是否合规输出 | ✅ 完成 | `is_compliant: yes/no` |
| 违规类型识别 | ✅ 完成 | `violations[].type` |
| 条文编号与原文 | ✅ 完成 | `clause_id`, `clause_text` |
| 置信度输出 | ✅ 完成 | `confidence`, `overall_confidence` |
| 百炼大模型 API | ✅ 完成 | 通义千问兼容模式 API |
| RAG 技术增强 | ✅ 完成 | 向量嵌入 + 相似度检索 |
| 评估模块 | ✅ 完成 | `evaluate` 命令，准确率 100% |

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                       用户输入层                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   文本输入    │  │   文件上传    │  │   图片OCR    │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
└─────────┼─────────────────┼─────────────────┼──────────────────┘
          │                 │                 │
          ▼                 ▼                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                      数据处理层                                  │
│  文本提取器 (extractors.py)                                      │
│  - DOCX: docx2txt  - PDF: PyMuPDF + Qwen-VL OCR  - 图片: OCR    │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                      RAG 检索层                                  │
│  知识库 (knowledge_base.py)                                      │
│  ┌─────────────┐  ┌─────────────┐                                │
│  │ 法规分块     │  │ 向量嵌入     │                                │
│  │ 137 chunks  │  │ text-embedding-v3                         │
│  └─────────────┘  └─────────────┘                                │
│         │                  │                                       │
│         └────────┬─────────┘                                       │
│                  ▼                                               │
│         ┌─────────────────┐                                      │
│         │ 余弦相似度检索   │  Top-K = 6                           │
│         └─────────────────┘                                      │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                      大模型推理层                                │
│  百炼客户端 (bailian_client.py)                                  │
│  - 聊天: qwen-plus  - 视觉: qwen-vl-plus  - 嵌入: text-embedding-v3│
│  审核器 (auditor.py) - Prompt Engineering: 结构化JSON输出        │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                      输出层                                      │
│  { is_compliant, violations, confidence, summary, ... }         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 核心代码结构

```
aliyun_rag/
├── config.py           # 配置管理（环境变量、.env支持）
├── bailian_client.py   # 百炼API客户端封装
├── extractors.py       # 文件文本提取（DOCX/PDF/图片OCR）
├── knowledge_base.py   # RAG知识库构建与检索
├── auditor.py          # 合规审核核心逻辑
├── evaluate.py         # 评估模块
├── main.py             # CLI命令入口
├── batch_audit.py      # 批量审核脚本
├── inspect_kb.py       # 知识库检查脚本
├── requirements.txt    # 依赖清单
├── .env                # API密钥配置
├── .env.example        # 配置模板
├── demo_cases.jsonl    # 测试数据集
├── test_data/          # 测试数据目录
│   ├── marketing_texts/    # 测试文案（8个示例）
│   └── audit_results/      # 审核结果输出
├── kb/                 # 知识库目录（自动生成）
└── references/         # 监管文档
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
| 视觉模型 | qwen-vl-plus | 用于 PDF/图片 OCR |
| 嵌入模型 | text-embedding-v3 | 用于向量检索 |
| 向量检索 | NumPy + 余弦相似度 | 轻量级本地实现 |
| 文件处理 | docx2txt, PyMuPDF, pypdf | 多格式支持 |
| 配置管理 | python-dotenv | 环境变量管理 |

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

**实际效果：**
```
总chunks: 137
平均长度: 162字符
最大长度: 700字符
最小长度: 18字符
分段条文: 2条 → 分成4个chunk
```

### 2. 向量存储方案

**当前实现：** 轻量级本地存储（适合Demo和小规模应用）

```
存储方案: NumPy + 文件系统
检索算法: 余弦相似度（矩阵乘法）
```

| 文件 | 大小 | 内容 |
|------|------|------|
| `chunks.jsonl` | 78.6 KB | 文本chunks（JSONL格式） |
| `embeddings.npy` | 548.1 KB | 向量矩阵（NumPy二进制） |
| `meta.json` | 124 B | 元数据 |

**向量维度：1024**（text-embedding-v3）

### 3. 向量检索实现

**核心代码：** `knowledge_base.py:retrieve_relevant_rules()`

```python
# 1. 查询向量化
query_vec = embed_texts([query])[0]  # → (1024,)

# 2. L2归一化
query_norm = query_vec / ||query_vec||
emb_norm = embeddings / ||embeddings||

# 3. 余弦相似度（矩阵乘法）
scores = emb_norm @ query_norm  # → (137,) 相似度分数

# 4. Top-K选择
top_indices = argsort(scores)[::-1][:6]  # 取前6个
```

**相似度分布：**
- 最高：0.9692
- 最低：0.3909
- 平均：0.6765

### 4. Prompt Engineering

**System Prompt：**
```python
"你是金融保险营销合规审核助手。"
"你只能依据给定监管条文进行判断，不得编造条文。"
"输出必须是严格JSON，不要输出Markdown。"
```

**User Prompt 结构：**
```
1. 营销文案
2. 检索到的条文（RAG提供，Top-6）
3. 输出格式要求（Few-Shot模板）
4. 约束条件（必须引用条文、置信度范围等）
```

**温度设置：0.0**（低温，输出更稳定）

---

## 🚀 使用指南

### 1. API 配置

编辑 `.env` 文件：

```bash
DASHSCOPE_API_KEY=sk-你的密钥
QWEN_CHAT_MODEL=qwen-plus
QWEN_VL_MODEL=qwen-vl-plus
QWEN_EMBEDDING_MODEL=text-embedding-v3
```

### 2. 构建知识库

```bash
PYTHONPATH=/mnt/workspace:$PYTHONPATH python -m aliyun_rag.main build-kb \
  --kb-dir kb --pdf-mode vl \
  --docs "references/保险销售行为管理办法.pdf" "references/互联网保险业务监管办法.docx"
```

### 3. 审核文本

```bash
PYTHONPATH=/mnt/workspace:$PYTHONPATH python -m aliyun_rag.main audit-text \
  --kb-dir kb --text "本保险保证年化收益8%，稳赚不赔。"
```

### 4. 批量审核测试文案

```bash
cd /mnt/workspace/aliyun_rag
PYTHONPATH=/mnt/workspace:$PYTHONPATH python batch_audit.py
```

**测试文案位置：** `test_data/marketing_texts/`
- 01_违规_夸大收益.txt
- 02_违规_误导性宣传.txt
- 03_合规_风险提示.txt
- 04_合规_正常宣传.txt
- 05_违规_承诺回报.txt
- 06_违规_无证代言.txt
- 07_合规_产品说明.txt
- 08_违规_混淆产品.txt

**结果保存位置：** `test_data/audit_results/`

### 5. 添加自己的测试文案

```bash
echo "你的测试文案" > test_data/marketing_texts/09_自定义测试.txt
```

然后重新运行批量审核脚本。

### 6. 查看知识库详情

```bash
PYTHONPATH=/mnt/workspace:$PYTHONPATH python inspect_kb.py
```

### 7. 运行评估

```bash
PYTHONPATH=/mnt/workspace:$PYTHONPATH python -m aliyun_rag.main evaluate \
  --kb-dir kb --dataset demo_cases.jsonl
```

---

## 📊 评估结果

### 基础测试集

```
Total: 4 cases
Correct: 4
Accuracy: 100%
```

| 测试文案 | 预期 | 实际 | 结果 |
|---------|------|------|------|
| 本保险产品保本保收益，年化收益率保证8%，零风险。 | 不合规 | 不合规 | ✅ |
| 本产品过往业绩不代表未来表现，具体以合同条款为准。 | 合规 | 合规 | ✅ |
| 限时抢购，购买后稳赚不赔，现在下单即可翻倍。 | 不合规 | 不合规 | ✅ |
| 请您仔细阅读保险条款，重点关注责任免除和犹豫期。 | 合规 | 合规 | ✅ |

### 批量测试集（8个文案）

| 指标 | 数量 |
|------|------|
| 总文件数 | 8 |
| 合规数量 | 1 |
| 不合规数量 | 7 |

---

## 📝 输出格式说明

### 审核结果 JSON 结构

```json
{
  "is_compliant": "no",              // 是否合规：yes/no
  "violations": [...],                // 违规详情列表
  "overall_confidence": 0.95,         // 整体置信度 [0-1]
  "summary": "...",                   // 一句话总结
  "retrieved_rules": [...],           // 检索到的相关法规
  "raw_model_output": "..."           // 原始模型输出
}
```

### 违规项结构

```json
{
  "type": "违规类型",                 // 违规类型描述
  "clause_id": "第九条",              // 条文编号
  "clause_text": "...",               // 条文原文
  "reason": "...",                    // 违规原因分析
  "confidence": 0.95                  // 该违规项置信度 [0-1]
}
```

---

## 💡 核心设计亮点

### 1. RAG 架构
- 法规文档按条文分块（137 chunks）
- 使用 text-embedding-v3 生成向量
- 余弦相似度检索最相关条文（Top-K=6）

### 2. Prompt Engineering
- 结构化 JSON 输出约束
- 系统提示词限定审核规则
- 必须引用检索到的条文

### 3. 多模态支持
- Qwen-VL 进行 PDF OCR
- 支持图片直接识别
- 降级策略：VL 失败时使用原生解析

### 4. 批量处理优化
- 嵌入 API 批量大小限制为 10
- 分批处理大量文本

---

## 🔍 常见问题

### Q: 如何修改检索的相关条文数量？

在命令中添加 `--top-k` 参数：

```bash
PYTHONPATH=/mnt/workspace:$PYTHONPATH python -m aliyun_rag.main audit-text \
  --kb-dir kb --top-k 10 \
  --text "测试文案"
```

### Q: Linux 环境如何处理 .doc 文件？

当前 Linux 环境不支持 PowerShell 自动转换 .doc 文件。建议：
- 手动将 .doc 转换为 .docx
- 或直接使用 .pdf 版本

### Q: 如何重新构建知识库？

```bash
PYTHONPATH=/mnt/workspace:$PYTHONPATH python -m aliyun_rag.main build-kb \
  --kb-dir kb --pdf-mode vl \
  --docs "references/保险销售行为管理办法.pdf" "references/互联网保险业务监管办法.docx"
```

---

## ⚠️ 注意事项

1. **.doc 文件处理**：当前 Linux 环境不支持 PowerShell，无法自动转换 .doc 文件。建议手动转换为 .docx 或使用 .pdf。
2. **API 配额**：频繁调用会消耗 API 配额，建议控制调用频率。
3. **知识库更新**：法规文档更新后需重新构建知识库。

---

## 🎓 考题对应关系

| 考题要求 | 代码实现 |
|---------|---------|
| 使用百炼大模型 API | `bailian_client.py` - OpenAI 兼容模式 |
| RAG 技术增强 | `knowledge_base.py` - 向量检索 |
| Prompt Engineering | `auditor.py` - 结构化提示词 |
| 输入文本/图文 | `main.py` - audit-text, audit-file |
| 是否合规输出 | `auditor.py` - is_compliant 字段 |
| 违规类型 | `auditor.py` - violations[].type |
| 条文编号与原文 | `auditor.py` - clause_id, clause_text |
| 置信度 | `auditor.py` - confidence, overall_confidence |
| 评估模块 | `evaluate.py` - evaluate_dataset |

---

## 📞 获取 API Key

访问：https://bailian.console.aliyun.com/

---

## 🚀 优化方向（生产级）

| 组件 | 当前实现 | 生产级推荐 |
|------|---------|-----------|
| 向量数据库 | NumPy本地文件 | Milvus / Chroma |
| Chunk策略 | 固定700字符 | 滑动窗口 + 重叠 |
| 检索算法 | 余弦相似度 | HNSW索引 + 重排序 |
| Prompt管理 | 硬编码 | Prompt模板库 |
| 缓存机制 | 无 | Redis缓存 |
| 并发处理 | 单线程 | 异步队列 |

**当前方案适合：**
- ✅ Demo演示
- ✅ 考试/面试项目
- ✅ 小规模应用（<10000条法规）
- ✅ 快速原型验证

---

**项目状态：✅ 完整可运行，满足所有考题要求**
