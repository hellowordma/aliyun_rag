# 快速可跑版：保险营销内容智能审核 Demo

本 Demo 满足你当前“先跑起来”的目标：
- 使用百炼兼容模式 API（`DASHSCOPE_API_KEY`）
- 使用 Qwen 文本模型做合规判定
- 支持 Qwen 多模态 OCR 提取 PDF（`--pdf-mode vl`）
- 输出结构化结果：是否合规、违规类型、条文编号与原文、置信度

## 1. 环境准备

在 `aliyun_project` 目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r aliyun_rag\requirements.txt
```

设置百炼 API Key：

```powershell
$env:DASHSCOPE_API_KEY="你的百炼Key"
```

可选模型变量（不设则使用默认值）：

```powershell
$env:QWEN_CHAT_MODEL="qwen-plus"
$env:QWEN_VL_MODEL="qwen-vl-plus"
$env:QWEN_EMBEDDING_MODEL="text-embedding-v3"
```

## 2. 构建法规知识库（含 PDF OCR）

默认会读取当前目录下三份监管文件：
- `互联网保险业务监管办法.docx`
- `保险销售行为管理办法.pdf`
- `金融产品网络营销管理办法（征求意见稿）.doc`

执行：

```powershell
python -m aliyun_rag.main build-kb --pdf-mode vl --kb-dir aliyun_rag/kb
```

说明：
- `--pdf-mode vl`：使用 Qwen-VL 做 PDF OCR（推荐）
- `--pdf-mode native`：用本地 PDF 文本解析（更快，但扫描件效果差）

`.doc` 文件处理说明：
- 程序会尝试调用本机 Word COM 自动转换 `.doc -> .docx`
- 若本机无 Word，建议手动另存为 `.docx` 后重跑

## 3. 审核文本

```powershell
python -m aliyun_rag.main audit-text --kb-dir aliyun_rag/kb --text "本保险保证年化收益8%，稳赚不赔。"
```

## 4. 审核文件（含 PDF/图片 OCR）

```powershell
python -m aliyun_rag.main audit-file --kb-dir aliyun_rag/kb --file "D:\path\to\marketing.pdf" --pdf-mode vl
```

## 5. 简单评测

```powershell
python -m aliyun_rag.main evaluate --kb-dir aliyun_rag/kb --dataset aliyun_rag/demo_cases.jsonl
```

## 6. 输出示例（字段）

```json
{
  "is_compliant": "no",
  "violations": [
    {
      "type": "夸大收益",
      "clause_id": "第XX条",
      "clause_text": "......",
      "reason": "......",
      "confidence": 0.91
    }
  ],
  "overall_confidence": 0.9,
  "summary": "存在夸大收益表述"
}
```

