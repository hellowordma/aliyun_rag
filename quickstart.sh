#!/bin/bash
# 保险营销内容智能审核系统 - 快速启动脚本

set -e

echo "========================================"
echo "保险营销内容智能审核系统"
echo "========================================"
echo ""

# 检查 API Key
if [ ! -f ".env" ]; then
    echo "❌ 未找到 .env 文件！"
    echo "请先复制 .env.example 为 .env 并填入你的 DASHSCOPE_API_KEY"
    exit 1
fi

source .env

if [ "$DASHSCOPE_API_KEY" = "your_api_key_here" ] || [ -z "$DASHSCOPE_API_KEY" ]; then
    echo "❌ 请在 .env 文件中设置你的 DASHSCOPE_API_KEY"
    echo "获取地址: https://bailian.console.aliyun.com/"
    exit 1
fi

echo "✅ API Key 配置已找到"
echo ""

# 检查依赖
echo "📦 检查依赖..."
pip install -r requirements.txt -q
echo "✅ 依赖已就绪"
echo ""

# 构建知识库
echo "📚 构建法规知识库..."
python -m aliyun_rag.main build-kb --kb-dir kb --pdf-mode vl
echo ""

# 运行测试
echo "🧪 运行测试审核..."
python -m aliyun_rag.main audit-text --kb-dir kb --text "本保险保证年化收益8%，稳赚不赔。"
echo ""

echo "========================================"
echo "✅ 系统配置完成！"
echo "========================================"
echo ""
echo "常用命令："
echo "  # 审核文本"
echo "  python -m aliyun_rag.main audit-text --kb-dir kb --text \"你的营销文案\""
echo ""
echo "  # 审核文件（支持 PDF/图片）"
echo "  python -m aliyun_rag.main audit-file --kb-dir kb --file /path/to/file.pdf"
echo ""
echo "  # 运行评估"
echo "  python -m aliyun_rag.main evaluate --kb-dir kb --dataset demo_cases.jsonl"
echo ""
