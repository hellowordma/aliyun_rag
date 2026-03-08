#!/bin/bash
# Gradio Demo 启动脚本

cd "$(dirname "$0")"

echo "========================================"
echo "保险营销内容智能审核系统 - Web Demo"
echo "========================================"
echo ""
echo "🚀 启动 Gradio 服务..."
echo ""

# 检查依赖
echo "📦 检查依赖..."
python3 -c "
import gradio
import rank_bm25
import jieba
print('✅ 所有依赖已安装')
" 2>/dev/null || {
    echo "❌ 缺少依赖，正在安装..."
    pip install -r ../requirements.txt
    pip install rank-bm25 jieba gradio
}

echo ""
echo "🌐 Web界面: http://localhost:7860"
echo "📚 API文档: http://localhost:7860/docs"
echo ""
echo "按 Ctrl+C 停止服务"
echo ""
echo "========================================"

# 启动 Gradio
python3 run.py
