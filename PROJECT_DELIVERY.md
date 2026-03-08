# 保险营销内容智能审核系统 - 项目提交说明

## 📦 项目概述

本项目是基于阿里云百炼大模型 API 构建的保险营销内容智能审核系统，使用 RAG（检索增强生成）技术，自动判断金融营销内容是否违反监管规定。

## 📂 项目结构

```
aliyun_rag/
├── 核心代码 (15个.py文件)
│   ├── __init__.py                 # 模块导出
│   ├── config.py                   # 配置管理
│   ├── bailian_client.py           # 百炼API客户端（含多模态）
│   ├── extractors.py               # 文件文本提取
│   ├── knowledge_base.py           # RAG知识库（NumPy）
│   ├── knowledge_base_milvus.py    # RAG知识库（Milvus）
│   ├── hybrid_retriever.py         # 混合检索（BM25 + 向量）
│   ├── auditor.py                  # 基础审核器
│   ├── enhanced_auditor.py         # 增强审核器（意图识别）
│   ├── multimodal_auditor.py       # 多模态审核器（图片/图文）
│   ├── confidence_calculator.py    # 置信度计算模块
│   ├── evaluate.py                  # 效果评估模块
│   ├── main.py                     # CLI命令入口
│   └── batch_audit.py              # 批量审核脚本
│
├── Demo应用
│   └── demo/
│       ├── app.py                  # Gradio Web应用
│       ├── cli_demo.py             # CLI交互式Demo
│       ├── cli_demo_auto.py        # CLI自动修复Demo
│       ├── run.py                  # Web启动脚本
│       └── start.sh                # Shell启动脚本
│
├── 配置与文档
│   ├── requirements.txt            # Python依赖
│   ├── .env.example               # 配置模板（无密钥）
│   ├── .gitignore                 # Git忽略规则
│   ├── README.md                   # 项目说明文档
│   ├── PROJECT_SUMMARY.md          # 完整技术文档
│   ├── LICENSE                     # 许可证
│   └── quickstart.sh               # 快速启动脚本
│
├── 测试数据
│   ├── demo_cases.jsonl           # 评估测试集
│   ├── test_data/
│   │   ├── images/                # 测试图片（3张）
│   │   ├── marketing_texts/        # 测试文案
│   │   └── create_test_images.py   # 图片生成脚本
│   └── cleanup.sh                 # 项目清理脚本
│
└── 参考资料
    └── references/
        ├── 保险销售行为管理办法.pdf
        ├── 互联网保险业务监管办法.docx
        └── 实操考题_候选人.docx
```

## ✅ 已完成的考题要求

| 要求 | 状态 | 说明 |
|------|------|------|
| 支持文本/图文输入 | ✅ | 文本、图片、图文混合三种输入模式 |
| 是否合规输出 | ✅ | is_compliant: yes/no |
| 违规类型识别 | ✅ | 7种违规类型自动识别 |
| 条文编号与原文 | ✅ | clause_id + clause_text |
| 置信度输出 | ✅ | 数学公式 + LLM 输出 |
| 百炼大模型API | ✅ | 通义千问（qwen-plus/qwen-vl-plus） |
| RAG技术增强 | ✅ | 向量检索 + BM25混合检索 |
| 评估模块 | ✅ | evaluate.py，准确率100% |

## 🚀 快速启动

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 配置API Key
```bash
# 复制配置模板
cp .env.example .env

# 编辑 .env，填入API密钥
# DASHSCOPE_API_KEY=sk-你的密钥
```

### 3. 构建知识库
```bash
# 方法1：使用脚本
bash quickstart.sh

# 方法2：手动命令
python -m aliyun_rag.main build-kb \
  --kb-dir kb \
  --vector-db numpy \
  --docs "references/保险销售行为管理办法.pdf" "references/互联网保险业务监管办法.docx"
```

### 4. 运行Demo

#### CLI Demo（推荐云服务器）
```bash
python demo/cli_demo.py
```

#### Web Demo
```bash
cd demo && bash start.sh
# 访问 http://localhost:7860
```

## 📊 核心技术栈

| 组件 | 技术选型 |
|------|---------|
| 大模型 | 阿里云百炼（通义千问） |
| 聊天模型 | qwen-plus |
| 视觉模型 | qwen-vl-plus |
| 嵌入模型 | text-embedding-v3 (1024维) |
| 向量数据库 | Milvus Lite / NumPy |
| 稀疏检索 | BM25 + jieba |
| 混合检索 | RRF融合（0.3 BM25 + 0.7 向量）|
| Web框架 | Gradio 4.0+ |

## 📈 评估结果

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

## 📝 提交清单

- [x] 已删除敏感配置文件 (.env)
- [x] 已清理临时文件（缓存、数据库等）
- [x] 已更新 README.md
- [x] 已更新 PROJECT_SUMMARY.md
- [x] 包含完整Demo代码
- [x] 包含评估模块
- [x] 包含测试数据
- [x] 包含参考资料

## 📦 打包文件

`aliyun_rag_clean.tar.gz` (510KB)

---

**项目状态**: ✅ 完整可运行，满足所有考题要求
