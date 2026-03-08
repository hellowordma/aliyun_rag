import base64
from typing import Any

import numpy as np
from openai import OpenAI

from .config import Settings


class BailianClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = OpenAI(
            api_key=settings.dashscope_api_key,
            base_url=settings.api_base,
        )

    def chat(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        temperature: float = 0.1,
        max_tokens: int | None = None,
    ) -> str:
        completion = self.client.chat.completions.create(
            model=model or self.settings.chat_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens or self.settings.max_tokens,
        )
        return (completion.choices[0].message.content or "").strip()

    def embed_texts(self, texts: list[str], batch_size: int = 10) -> np.ndarray:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            response = self.client.embeddings.create(
                model=self.settings.embedding_model,
                input=batch,
            )
            vectors.extend(item.embedding for item in response.data)

        return np.asarray(vectors, dtype=np.float32)

    def ocr_image_bytes(
        self,
        image_bytes: bytes,
        image_mime: str = "image/png",
        prompt: str | None = None,
    ) -> str:
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        image_url = f"data:{image_mime};base64,{image_base64}"

        user_prompt = prompt or (
            "请对图片做高保真OCR，输出完整文本。"
            "保留数字、标点、标题和条款编号，不要解释，不要摘要。"
        )

        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": "你是专业OCR助手，只返回识别到的文本内容。",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            },
        ]

        return self.chat(
            messages=messages,
            model=self.settings.vl_model,
            temperature=0.0,
            max_tokens=self.settings.max_tokens,
        )

    def ocr_image_path(self, image_path: str, prompt: str | None = None) -> str:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        return self.ocr_image_bytes(image_bytes=image_bytes, prompt=prompt)

    def analyze_marketing_image(
        self,
        image_bytes: bytes,
        image_mime: str = "image/png",
        text_context: str | None = None,
    ) -> str:
        """分析营销图片内容，提取文字和视觉元素用于合规审核

        Args:
            image_bytes: 图片字节数据
            image_mime: 图片MIME类型 (image/png, image/jpeg等)
            text_context: 可选的文字说明，与图片一起分析

        Returns:
            营销内容分析结果（JSON格式字符串）
        """
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        image_url = f"data:{image_mime};base64,{image_base64}"

        # 构建多模态分析提示词
        if text_context:
            user_prompt = f"""请分析这张保险营销图片和相关文字说明。

文字说明：
{text_context}

请输出以下内容：
1. 图片中识别到的所有文字内容（包括标题、标语、说明文字等）
2. 图片中的视觉元素描述（如人物代言、图表、Logo等）
3. 综合图片和文字说明，提取完整的营销内容

输出格式为JSON：
{{
  "extracted_text": "图片中提取的所有文字",
  "visual_elements": ["视觉元素1", "视觉元素2"],
  "marketing_content": "综合的营销内容描述",
  "detected_issues": ["可能存在的问题1", "可能存在的问题2"]
}}"""
        else:
            user_prompt = """请分析这张保险营销图片的内容。

请输出以下内容：
1. 图片中识别到的所有文字内容（包括标题、标语、说明文字等）
2. 图片中的视觉元素描述（如人物代言、图表、Logo等）
3. 综合图片内容，提取完整的营销内容

输出格式为JSON：
{
  "extracted_text": "图片中提取的所有文字",
  "visual_elements": ["视觉元素1", "视觉元素2"],
  "marketing_content": "综合的营销内容描述",
  "detected_issues": ["可能存在的问题1", "可能存在的问题2"]
}"""

        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": "你是保险营销合规审核专家，擅长分析图片中的营销内容并识别潜在的合规问题。",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            },
        ]

        return self.chat(
            messages=messages,
            model=self.settings.vl_model,
            temperature=0.0,
            max_tokens=self.settings.max_tokens,
        )
