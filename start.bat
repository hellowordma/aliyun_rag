@echo off
chcp 65001 >nul
title 保险营销内容智能审核系统

echo ======================================
echo 保险营销内容智能审核系统
echo ======================================
echo.

REM 检查Python是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未检测到Python，请先安装Python 3.9或更高版本
    echo 下载地址: https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

echo [√] Python已安装
echo.

REM 检查.env文件
if not exist ".env" (
    echo [提示] 未找到.env配置文件
    echo.
    if exist ".env.example" (
        echo 正在创建配置文件...
        copy ".env.example" ".env" >nul
        echo [√] 已创建.env文件
        echo.
        echo ======================================
        echo 重要提示: 请编辑.env文件
        echo ======================================
        echo 1. 用记事本打开.env文件
        echo 2. 将 DASHSCOPE_API_KEY=sk-your-api-key-here
        echo 3. 改为你的实际API密钥
        echo 4. 保存后重新运行此脚本
        echo.
        echo 获取API密钥: https://bailian.console.aliyun.com/
        echo ======================================
        echo.
        notepad .env
        echo 配置完成后，请重新运行此脚本
        pause
        exit /b 0
    ) else (
        echo [错误] 找不到配置模板文件
        pause
        exit /b 1
    )
)

REM 检查是否配置了API密钥
findstr "sk-your-api-key-here" .env >nul
if not errorlevel 1 (
    echo [错误] 请先在.env文件中填入你的API密钥
    echo.
    notepad .env
    echo 配置完成后，请重新运行此脚本
    pause
    exit /b 1
)

echo [√] 配置文件已就绪
echo.

REM 检查知识库
if not exist "kb\chunks.jsonl" (
    echo [提示] 知识库不存在，正在构建...
    echo 这可能需要2-3分钟，请耐心等待...
    echo.
    python -m aliyun_rag.main build-kb --kb-dir kb --vector-db numpy --docs "references/保险销售行为管理办法.pdf" "references/互联网保险业务监管办法.docx"
    if errorlevel 1 (
        echo.
        echo [错误] 知识库构建失败
        pause
        exit /b 1
    )
    echo.
    echo [√] 知识库构建完成
    echo.
)

REM 启动CLI Demo
echo ======================================
echo 正在启动审核系统...
echo ======================================
echo.

python demo\cli_demo.py

pause
