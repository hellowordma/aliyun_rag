#!/bin/bash
# 项目清理脚本 - 删除临时文件和敏感信息

echo "======================================"
echo "保险营销审核系统 - 项目清理"
echo "======================================"
echo ""

# 清理Python缓存
echo "[1/6] 清理Python缓存..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete 2>/dev/null
echo "  ✓ 已删除 .pyc 文件和 __pycache__ 目录"

# 清理敏感配置（保留.example）
echo "[2/6] 清理敏感配置..."
if [ -f ".env" ]; then
    rm -f .env
    echo "  ✓ 已删除 .env 文件（包含API密钥）"
else
    echo "  - .env 文件不存在，跳过"
fi

# 清理生成的数据库
echo "[3/6] 清理生成的数据库..."
rm -f milvus_demo.db
echo "  ✓ 已删除 milvus_demo.db"

# 清理知识库（可重新生成）
echo "[4/6] 清理知识库..."
rm -rf kb/ kb_milvus/
echo "  ✓ 已删除 kb/ 和 kb_milvus/ 目录"

# 清理审核结果
echo "[5/6] 清理审核结果..."
rm -rf test_data/audit_results/
echo "  ✓ 已删除 test_data/audit_results/"

# 清理日志
echo "[6/6] 清理日志..."
rm -rf audit_logs/ 2>/dev/null
echo "  ✓ 已删除 audit_logs/ 目录"

echo ""
echo "======================================"
echo "清理完成！"
echo ""
echo "提交前请确保："
echo "  1. 已删除 .env 文件"
echo "  2. 已更新 .env.example（如需要）"
echo "  3. 已更新 README.md"
echo "  4. 已更新 PROJECT_SUMMARY.md"
echo ""
echo "打包命令："
echo "  cd /mnt/workspace"
echo "  tar czf aliyun_rag_clean.tar.gz --exclude='.git' aliyun_rag/"
echo "======================================"
