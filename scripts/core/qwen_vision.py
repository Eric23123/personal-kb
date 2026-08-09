"""DashScope Qwen-VL client for Personal-KB media preprocessing."""

from __future__ import annotations

import os
from typing import Any

from .common_client import HttpClientError, JsonHttpClient, load_api_key

DEFAULT_QWEN_VISION_API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
DEFAULT_QWEN_VISION_MODEL = "qwen-vl-plus"


class QwenVisionClient:
    """Small OpenAI-compatible Qwen-VL client with no import-time network work."""

    def __init__(
        self,
        api_key: str | None = None,
        api_url: str | None = None,
        *,
        http_client: JsonHttpClient | None = None,
        timeout: float = 180,
    ) -> None:
        self.api_key = api_key or load_api_key("DASHSCOPE_API_KEY")
        self.api_url = api_url or os.environ.get("PERSONAL_KB_QWEN_VISION_API_URL", DEFAULT_QWEN_VISION_API_URL)
        self.http_client = http_client or JsonHttpClient(timeout=timeout, retries=2)
        self.timeout = timeout

    def complete_image(self, prompt: str, image_b64: str, *, model: str = DEFAULT_QWEN_VISION_MODEL) -> str:
        if not self.api_key:
            raise HttpClientError("DASHSCOPE_API_KEY is not set for Qwen-VL image processing")
        response = self.http_client.post_json(
            self.api_url,
            {
                "model": model,
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                ]}],
                "temperature": 0.1,
                "max_tokens": 2048,
            },
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=self.timeout,
        )
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise HttpClientError("Qwen-VL response did not contain choices[0].message.content") from error
        if not isinstance(content, str):
            raise HttpClientError("Qwen-VL response content was not text")
        return content
