from dataclasses import dataclass
import os


@dataclass
class Settings:
    dashscope_api_key: str
    api_base: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    chat_model: str = "qwen-plus"
    vl_model: str = "qwen-vl-plus"
    embedding_model: str = "text-embedding-v3"
    max_tokens: int = 1800

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            dashscope_api_key=os.getenv("DASHSCOPE_API_KEY", "").strip(),
            api_base=os.getenv("BAILIAN_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1").strip(),
            chat_model=os.getenv("QWEN_CHAT_MODEL", "qwen-plus").strip(),
            vl_model=os.getenv("QWEN_VL_MODEL", "qwen-vl-plus").strip(),
            embedding_model=os.getenv("QWEN_EMBEDDING_MODEL", "text-embedding-v3").strip(),
            max_tokens=int(os.getenv("QWEN_MAX_TOKENS", "1800")),
        )

    def validate(self) -> None:
        if not self.dashscope_api_key:
            raise RuntimeError(
                "Missing DASHSCOPE_API_KEY. Please set it before running the demo."
            )
