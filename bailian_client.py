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

    def embed_texts(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
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
