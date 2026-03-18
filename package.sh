#!/bin/bash
# 项目打包脚本 - 创建提交给HR的压缩包（包含数据库）

set -e

echo "======================================"
echo "保险营销审核系统 - 项目打包"
echo "======================================"
echo ""

# 配置
PROJECT_NAME="aliyun_rag"
PACKAGE_NAME="${PROJECT_NAME}_submission.tar.gz"
TEMP_DIR="/tmp/${PROJECT_NAME}_pack"

# 清理并创建临时目录
echo "[1/5] 准备打包环境..."
rm -rf "$TEMP_DIR"
mkdir -p "$TEMP_DIR"
echo "  ✓ 临时目录已创建"

# 复制项目文件
echo ""
echo "[2/5] 复制项目文件..."
mkdir -p "$TEMP_DIR/$PROJECT_NAME"

# 复制所有文件，只排除敏感和临时文件
for item in *; do
    case "$item" in
        __pycache__|*.pyc|.git|audit_logs|.DS_Store|*.log|venv*|.idea|.vscode)
            # 跳过这些文件/目录
            ;;
        *)
            if [ -f "$item" ]; then
                cp "$item" "$TEMP_DIR/$PROJECT_NAME/"
            elif [ -d "$item" ]; then
                cp -r "$item" "$TEMP_DIR/$PROJECT_NAME/"
            fi
            ;;
    esac
done

# 清理复制后不需要的内容
rm -rf "$TEMP_DIR/$PROJECT_NAME/__pycache__"
rm -rf "$TEMP_DIR/$PROJECT_NAME/demo/__pycache__"
find "$TEMP_DIR/$PROJECT_NAME" -name "*.pyc" -delete
rm -rf "$TEMP_DIR/$PROJECT_NAME/.git"

# 清理审核结果目录（保留知识库）
rm -rf "$TEMP_DIR/$PROJECT_NAME/test_data/audit_results"

echo "  ✓ 核心文件已复制"

# 处理 .env 文件（替换为模板）
echo ""
echo "[3/5] 处理敏感配置..."
if [ -f "$TEMP_DIR/$PROJECT_NAME/.env" ]; then
    rm -f "$TEMP_DIR/$PROJECT_NAME/.env"
    echo "  ✓ 已删除 .env（包含API密钥）"
fi

# 确保 .env.example 存在
if [ ! -f "$TEMP_DIR/$PROJECT_NAME/.env.example" ]; then
    cat > "$TEMP_DIR/$PROJECT_NAME/.env.example" << 'EOF'
# 阿里云百炼 API 配置
DASHSCOPE_API_KEY=sk-your-api-key-here

# 模型配置（可选，使用默认值）
# QWEN_CHAT_MODEL=qwen-plus
# QWEN_VL_MODEL=qwen-vl-plus
# QWEN_EMBEDDING_MODEL=text-embedding-v3

# 向量数据库类型（numpy 或 milvus）
# VECTOR_DB_TYPE=numpy
EOF
    echo "  ✓ 已创建 .env.example 模板"
fi

# 统计包含的内容
echo ""
echo "[4/5] 统计打包内容..."

# 检查 Milvus 数据库
if [ -f "$TEMP_DIR/$PROJECT_NAME/milvus_demo.db" ]; then
    DB_SIZE=$(du -h "$TEMP_DIR/$PROJECT_NAME/milvus_demo.db" | cut -f1)
    echo "  ✓ Milvus 数据库: milvus_demo.db ($DB_SIZE)"
else
    echo "  - Milvus 数据库: 未找到"
fi

# 检查知识库元数据
if [ -d "$TEMP_DIR/$PROJECT_NAME/kb_milvus" ]; then
    CHUNKS=$(wc -l < "$TEMP_DIR/$PROJECT_NAME/kb_milvus/chunks.jsonl" 2>/dev/null || echo "0")
    echo "  ✓ 知识库元数据: kb_milvus/ ($CHUNKS chunks)"
else
    echo "  - 知识库元数据: 未找到"
fi

# 创建压缩包
echo ""
echo "[5/5] 创建压缩包..."
cd "$TEMP_DIR"
tar -czf "$PACKAGE_NAME" "$PROJECT_NAME"
mv "$PACKAGE_NAME" "/mnt/workspace/"
rm -rf "$TEMP_DIR"

# 获取文件大小
PACKAGE_SIZE=$(du -h "/mnt/workspace/$PACKAGE_NAME" | cut -f1)
PACKAGE_PATH="/mnt/workspace/$PACKAGE_NAME"

echo "  ✓ 压缩完成"
echo ""

# 显示打包结果
echo "======================================"
echo "打包完成！"
echo "======================================"
echo ""
echo "文件信息:"
echo "  名称: $PACKAGE_NAME"
echo "  路径: $PACKAGE_PATH"
echo "  大小: $PACKAGE_SIZE"
echo ""
echo "包含内容:"
echo "  - 核心代码 (15个 .py 文件)"
echo "  - Demo应用 (demo/)"
echo "  - 配置文件 (requirements.txt, .env.example)"
echo "  - 文档 (README.md, PROJECT_SUMMARY.md, etc.)"
echo "  - 参考资料 (references/)"
echo "  - 测试数据 (test_data/)"
echo "  - Milvus 数据库 (milvus_demo.db)"
echo "  - 知识库元数据 (kb_milvus/)"
echo ""
echo "已删除:"
echo "  - .env (敏感信息)"
echo "  - __pycache__, *.pyc (缓存)"
echo "  - test_data/audit_results/ (审核结果)"
echo ""
echo "解压命令:"
echo "  tar -xzf $PACKAGE_NAME"
echo ""
echo "后续步骤 (HR接收后):"
echo "  1. 解压文件"
echo "  2. 复制 .env.example 为 .env"
echo "  3. 填入 DASHSCOPE_API_KEY"
echo "  4. 直接运行: python demo/cli_demo.py"
echo "  (知识库已构建，无需重建)"
echo "======================================"
