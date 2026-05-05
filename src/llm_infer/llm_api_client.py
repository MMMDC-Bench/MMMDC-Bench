import base64
import json
import mimetypes
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import unquote, urlparse

from loguru import logger


def _extract_text_from_openai_choice(choice: Dict[str, Any]) -> str:
    message = choice.get("message", {}) if isinstance(choice, dict) else {}
    content = message.get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
        return "\n".join(parts).strip()
    return str(content).strip()


def _extract_text_from_anthropic(response_json: Dict[str, Any]) -> str:
    content = response_json.get("content", [])
    if not isinstance(content, list):
        return str(content).strip()
    parts: List[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text" and isinstance(block.get("text"), str):
            parts.append(block["text"].strip())
    return "\n".join([p for p in parts if p]).strip()


class LlmApiClient:
    """
    轻量级LLM API客户端（无框架依赖）。
    支持协议:
      - openai: /chat/completions
      - anthropic: /messages
      - dashscope: /compatible-mode/v1/chat/completions
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str,
        api_protocol: str = "openai",
        max_retries: int = 3,
        retry_delay: float = 2.0,
        timeout: float = 120.0,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.api_protocol = api_protocol.strip().lower()
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens

        self._lock = threading.Lock()
        self.total_requests = 0
        self.success_requests = 0
        self.failed_requests = 0
        self.latencies_ms: List[float] = []

    def _resolve_image(self, image_path_or_url: str) -> Optional[Dict[str, str]]:
        value = (image_path_or_url or "").strip()
        if not value:
            return None

        # 远程URL原样透传（主要用于OpenAI兼容格式）
        if value.startswith("http://") or value.startswith("https://"):
            return {"kind": "url", "value": value}

        if value.startswith("file://"):
            parsed = urlparse(value)
            local_path = unquote(parsed.path or "")
            path = Path(local_path)
        else:
            path = Path(value)
        if not path.is_absolute():
            path = Path.cwd() / path

        if not path.exists() or not path.is_file():
            logger.warning(f"Image not found, skipping: {value}")
            return None

        mime_type, _ = mimetypes.guess_type(str(path))
        if not mime_type:
            mime_type = "image/png"

        with open(path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        return {"kind": "base64", "value": encoded, "mime_type": mime_type}

    def _build_openai_payload(self, prompt: str, images: Optional[List[str]]) -> Dict[str, Any]:
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for image in images or []:
            resolved = self._resolve_image(image)
            if not resolved:
                continue
            if resolved["kind"] == "url":
                content.append(
                    {"type": "image_url", "image_url": {"url": resolved["value"]}}
                )
            else:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{resolved['mime_type']};base64,{resolved['value']}"
                        },
                    }
                )

        return {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

    def _build_anthropic_payload(self, prompt: str, images: Optional[List[str]]) -> Dict[str, Any]:
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for image in images or []:
            resolved = self._resolve_image(image)
            if not resolved:
                continue
            if resolved["kind"] == "url":
                content.append(
                    {
                        "type": "image",
                        "source": {"type": "url", "url": resolved["value"]},
                    }
                )
            else:
                content.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": resolved["mime_type"],
                            "data": resolved["value"],
                        },
                    }
                )

        return {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

    def _build_request(self, prompt: str, images: Optional[List[str]]) -> Tuple[str, Dict[str, str], Dict[str, Any]]:
        if self.api_protocol == "openai":
            url = f"{self.base_url}/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }
            payload = self._build_openai_payload(prompt, images)
            return url, headers, payload

        if self.api_protocol == "dashscope":
            # DashScope 标准兼容模式（OpenAI-compatible）
            url = f"{self.base_url}/compatible-mode/v1/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }
            payload = self._build_openai_payload(prompt, images)
            return url, headers, payload

        if self.api_protocol == "anthropic":
            url = f"{self.base_url}/messages"
            headers = {
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            }
            payload = self._build_anthropic_payload(prompt, images)
            return url, headers, payload

        raise ValueError(
            f"Unsupported api_protocol: {self.api_protocol}. Supported: openai, anthropic, dashscope"
        )

    def _parse_response_text(self, response_json: Dict[str, Any]) -> str:
        if self.api_protocol in {"openai", "dashscope"}:
            choices = response_json.get("choices", [])
            if isinstance(choices, list) and choices:
                return _extract_text_from_openai_choice(choices[0])
            return ""

        if self.api_protocol == "anthropic":
            return _extract_text_from_anthropic(response_json)

        return ""

    def _record_stats(self, success: bool, latency_ms: float) -> None:
        with self._lock:
            self.total_requests += 1
            if success:
                self.success_requests += 1
            else:
                self.failed_requests += 1
            self.latencies_ms.append(latency_ms)

    def chat_with_images(self, prompt: str, images: Optional[List[str]] = None) -> str:
        url, headers, payload = self._build_request(prompt, images)
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        last_error = None
        for attempt in range(self.max_retries):
            start = time.time()
            try:
                req = urllib.request.Request(url=url, data=body, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    raw = resp.read().decode("utf-8")
                    response_json = json.loads(raw)

                text = self._parse_response_text(response_json).strip()
                self._record_stats(success=True, latency_ms=(time.time() - start) * 1000.0)
                return text
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
                last_error = e
                self._record_stats(success=False, latency_ms=(time.time() - start) * 1000.0)
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                continue

        logger.error(f"Request failed after {self.max_retries} retries: {last_error}")
        return ""

    def batch_chat_with_images(
        self,
        requests: List[Tuple[str, Optional[List[str]]]],
        max_workers: int = 4,
        show_progress: bool = False,
    ) -> List[str]:
        _ = show_progress
        if not requests:
            return []

        responses: List[str] = [""] * len(requests)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(self.chat_with_images, prompt, images): idx
                for idx, (prompt, images) in enumerate(requests)
            }
            for future in as_completed(future_map):
                idx = future_map[future]
                try:
                    responses[idx] = future.result()
                except Exception as e:
                    logger.error(f"Batch request failed at index={idx}: {e}")
                    responses[idx] = ""
        return responses

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            avg_latency = sum(self.latencies_ms) / len(self.latencies_ms) if self.latencies_ms else 0.0
            success_rate = (
                f"{(self.success_requests / self.total_requests) * 100:.2f}%"
                if self.total_requests > 0
                else "0.00%"
            )
            return {
                "total_requests": self.total_requests,
                "success_requests": self.success_requests,
                "failed_requests": self.failed_requests,
                "success_rate": success_rate,
                "avg_latency_ms": round(avg_latency, 2),
            }

    def print_stats(self) -> None:
        stats = self.get_stats()
        logger.info(f"Total Requests: {stats['total_requests']}")
        logger.info(f"Success Requests: {stats['success_requests']}")
        logger.info(f"Failed Requests: {stats['failed_requests']}")
        logger.info(f"Success Rate: {stats['success_rate']}")
        logger.info(f"Avg Latency: {stats['avg_latency_ms']} ms")
