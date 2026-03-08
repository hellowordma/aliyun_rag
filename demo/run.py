#!/usr/bin/env python3
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

if __name__ == "__main__":
    import gradio as gr
    from aliyun_rag.demo.app import create_ui

    print("=" * 70)
    print("保险营销内容智能审核系统 - Web Demo")
    print("=" * 70)
    print()
    print(f"📂 项目路径: {project_root}")
    print(f"🐍 Python版本: {sys.version.split()[0]}")
    print(f"✅ Gradio版本: {gr.__version__}")
    print()
    print("🌐 Web界面: http://localhost:7860")
    print("📚 API文档: http://localhost:7860/docs")
    print()
    print("按 Ctrl+C 停止服务")
    print("=" * 70)
    print()

    # 创建并启动 UI
    ui = create_ui()
    ui.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
