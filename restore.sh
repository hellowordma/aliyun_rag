#!/bin/bash
# 项目恢复脚本 - 恢复被删除的文件和目录

echo "======================================"
echo "保险营销审核系统 - 项目恢复"
echo "======================================"
echo ""

# 1. 创建目录结构
echo "[1/6] 创建目录结构..."
mkdir -p kb kb_milvus test_data/audit_results __pycache__ demo/__pycache__
echo "  ✓ 目录结构已创建"

# 2. 恢复 .env 文件
echo ""
echo "[2/6] 恢复 .env 配置文件..."
if [ ! -f ".env" ]; then
    cat > .env << 'EOF'
# 阿里云百炼 API 配置
DASHSCOPE_API_KEY=sk-your-api-key-here

# 模型配置（可选，使用默认值）
# QWEN_CHAT_MODEL=qwen-plus
# QWEN_VL_MODEL=qwen-vl-plus
# QWEN_EMBEDDING_MODEL=text-embedding-v3

# 向量数据库类型（numpy 或 milvus）
# VECTOR_DB_TYPE=numpy
EOF
    echo "  ✓ .env 文件已创建（请填入您的API密钥）"
else
    echo "  - .env 文件已存在，跳过"
fi

# 3. 恢复知识库
echo ""
echo "[3/6] 恢复知识库..."
if [ -f ".env" ]; then
    # 加载环境变量
    export $(cat .env | grep -v '^#' | xargs)

    # 检查API密钥
    if grep -q "sk-your-api-key-here" .env 2>/dev/null; then
        echo "  ⚠️  请先在 .env 文件中填入您的 API 密钥"
        echo "  然后运行以下命令构建知识库："
        echo "      python -m aliyun_rag.main build-kb --kb-dir kb --vector-db numpy \\"
        echo "        --docs 'references/保险销售行为管理办法.pdf' 'references/互联网保险业务监管办法.docx'"
    else
        echo "  正在构建知识库..."
        python -m aliyun_rag.main build-kb \
            --kb-dir kb \
            --vector_db numpy \
            --docs "references/保险销售行为管理办法.pdf" "references/互联网保险业务监管办法.docx" 2>/dev/null
        if [ $? -eq 0 ]; then
            echo "  ✓ 知识库构建完成"
        else
            echo "  ⚠️  知识库构建失败，请手动执行"
        fi
    fi
else
    echo "  - .env 文件不存在，已创建模板"
fi

# 4. Python缓存会在运行时自动生成
echo ""
echo "[4/6] Python缓存..."
echo "  - 缓存文件会在运行Python时自动生成"
echo "  - 可手动生成: python -m py_compile *.py"

# 5. 数据库会在使用Milvus时自动创建
echo ""
echo "[5/6] 数据库..."
echo "  - Milvus数据库会在使用Milvus知识库时自动创建"

# 6. 审核结果目录
echo ""
echo "[6/6] 审核结果目录..."
mkdir -p test_data/audit_results
echo "  ✓ 目录已就绪"

echo ""
echo "======================================"
echo "恢复完成！"
echo ""
echo "下一步："
echo "  1. 编辑 .env 文件，填入您的 DASHSCOPE_API_KEY"
echo "  2. 运行以下命令构建知识库："
echo "     python -m aliyun_rag.main build-kb --kb-dir kb --vector-db numpy \\"
echo "       --docs 'references/保险销售行为管理办法.pdf' 'references/互联网保险业务监管办法.docx'"
echo "  3. 运行 Demo："
echo "     python demo/cli_demo.py"
echo "======================================"
