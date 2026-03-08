#!/usr/bin/env python3
"""
创建测试图片

生成用于测试多模态审核功能的图片
"""

from PIL import Image, ImageDraw, ImageFont
import os

# 创建输出目录
output_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(output_dir, "images")
os.makedirs(images_dir, exist_ok=True)

def create_text_image(text, filename, color=(0, 0, 0), bg_color=(255, 255, 255)):
    """创建带文字的图片"""
    # 图片尺寸
    width, height = 800, 600

    # 创建图片
    img = Image.new('RGB', (width, height), bg_color)
    draw = ImageDraw.Draw(img)

    # 尝试使用中文字体，如果失败则使用默认字体
    try:
        # 尝试使用常见的中文字体
        font_paths = [
            '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
            '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
            '/System/Library/Fonts/PingFang.ttc',
        ]
        font = None
        for font_path in font_paths:
            if os.path.exists(font_path):
                font = ImageFont.truetype(font_path, 40)
                break

        if font is None:
            font = ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    # 计算文字位置（居中）
    # 将文字按行分割
    lines = text.split('\n')
    line_height = 60
    total_height = len(lines) * line_height
    y_start = (height - total_height) // 2

    # 绘制文字
    for i, line in enumerate(lines):
        # 获取文字边界框
        bbox = draw.textbbox((0, 0), line, font=font)
        text_width = bbox[2] - bbox[0]
        x = (width - text_width) // 2
        y = y_start + i * line_height

        # 绘制文字
        draw.text((x, y), line, fill=color, font=font)

    # 保存图片
    output_path = os.path.join(images_dir, filename)
    img.save(output_path)
    print(f"Created: {output_path}")
    return output_path


def create_violation_image_1():
    """创建违规图片1：夸大收益"""
    text = """保险产品
年化收益率保证8%
保本保收益 零风险
稳赚不赔！
限时抢购！"""

    return create_text_image(
        text,
        "violation_exaggerated_return.png",
        color=(220, 20, 60),  # 深红色
        bg_color=(255, 250, 240)
    )


def create_violation_image_2():
    """创建违规图片2：误导宣传"""
    text = """明星同款！
跟着名人买保险
收益翻倍！
错过等一年！"""

    return create_text_image(
        text,
        "violation_misleading.png",
        color=(255, 140, 0),  # 橙色
        bg_color=(255, 255, 240)
    )


def create_compliant_image():
    """创建合规图片"""
    text = """保险产品提示
过往业绩不代表未来表现
具体以合同条款为准
投资有风险 投保需谨慎
请仔细阅读保险条款"""

    return create_text_image(
        text,
        "compliant_risk_warning.png",
        color=(0, 100, 0),  # 深绿色
        bg_color=(240, 255, 240)
    )


if __name__ == "__main__":
    print("Creating test images...")

    create_violation_image_1()
    create_violation_image_2()
    create_compliant_image()

    print(f"\nTest images created in: {images_dir}")
    print("\nYou can now test the image audit feature with:")
    print(f"  - {images_dir}/violation_exaggerated_return.png")
    print(f"  - {images_dir}/violation_misleading.png")
    print(f"  - {images_dir}/compliant_risk_warning.png")
