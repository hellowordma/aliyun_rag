# 项目文件恢复指南

## 已删除的文件说明

### 1. .env (敏感配置)
**状态**: 需要手动创建  
**原因**: 包含API密钥，无法自动恢复  
**恢复方法**:
```bash
# 方法1: 使用恢复脚本
bash restore.sh

# 方法2: 手动创建
cat > .env << 'EOF'
DASHSCOPE_API_KEY=sk-your-api-key-here
