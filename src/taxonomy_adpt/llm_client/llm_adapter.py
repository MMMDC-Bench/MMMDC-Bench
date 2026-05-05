"""
LLM 适配层 - 用于 TaxoAdapt 项目
使用标准大模型调用协议（openai/anthropic/dashscope）。
"""

import os
from typing import List, Any, Optional

# 尝试导入 loguru，如果不可用则使用标准 logging
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

from src.llm_infer.llm_api_client import LlmApiClient


def _append_json_guard(prompt: str) -> str:
    if "JSON" in prompt:
        return prompt
    return prompt + "\n\n请以标准JSON格式输出，不要包含markdown代码块标记。"


class LLMAdapter:
    """Taxonomy 任务的统一 LLM 适配器。"""

    def __init__(
        self,
        model_name: str,
        api_protocol: str,
        base_url: str,
        api_key: str,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        timeout: float = 120.0,
        temperature: float = 0.1,
        max_new_tokens: int = 3000,
    ):
        self.model_name = model_name
        self.client = LlmApiClient(
            model=model_name,
            base_url=base_url,
            api_key=api_key,
            api_protocol=api_protocol,
            max_retries=max_retries,
            retry_delay=retry_delay,
            timeout=timeout,
            temperature=temperature,
            max_tokens=max_new_tokens,
        )
        logger.info(
            f"[LLMAdapter] initialized | model={model_name} protocol={api_protocol} base_url={base_url}"
        )

    def chat(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_new_tokens: int = 3000,
        json_mode: bool = False,
        schema: Any = None,
        image_base64: Optional[str] = None,
    ) -> str:
        _ = temperature
        _ = max_new_tokens
        query = _append_json_guard(prompt) if (json_mode or schema is not None) else prompt
        images = [image_base64] if image_base64 else None
        response = self.client.chat_with_images(query, images)
        if response.startswith("```"):
            lines = response.split("\n")
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            response = "\n".join(lines)
        return response

    def batch_chat(
        self,
        prompts: List[str],
        temperature: float = 0.1,
        max_new_tokens: int = 3000,
        json_mode: bool = False,
        schema: Any = None,
        max_workers: int = 3,
        show_progress: bool = True,
        images_base64: Optional[List[Optional[str]]] = None,
        timeout_per_request: float = 120.0,
        max_batch_retries: int = 2,
    ) -> List[str]:
        _ = temperature
        _ = max_new_tokens
        _ = show_progress
        _ = max_batch_retries
        self.client.timeout = timeout_per_request

        processed_prompts = []
        for p in prompts:
            processed_prompts.append(_append_json_guard(p) if (json_mode or schema is not None) else p)

        if images_base64 and len(images_base64) < len(processed_prompts):
            images_base64 = images_base64 + [None] * (len(processed_prompts) - len(images_base64))

        requests = []
        for idx, prompt in enumerate(processed_prompts):
            image = images_base64[idx] if images_base64 else None
            requests.append((prompt, [image] if image else None))

        responses = self.client.batch_chat_with_images(
            requests=requests,
            max_workers=max_workers,
            show_progress=False,
        )

        cleaned = []
        for response in responses:
            if response.startswith("```"):
                lines = response.split("\n")
                if lines and lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                response = "\n".join(lines)
            cleaned.append(response)
        return cleaned

    def get_stats(self):
        return self.client.get_stats()

    def print_stats(self):
        self.client.print_stats()


_global_adapter: Optional[LLMAdapter] = None


def _resolve_protocol_from_args(args) -> str:
    protocol = getattr(args, "api_protocol", "") or os.environ.get("API_PROTOCOL", "")
    if protocol:
        return protocol.strip().lower()

    llm = getattr(args, "llm", "gpt")
    if llm == "claude":
        return "anthropic"
    return "openai"


def _build_adapter_from_args(args) -> LLMAdapter:
    model_name = getattr(args, "model_name", "gpt-4o")
    api_protocol = _resolve_protocol_from_args(args)
    base_url = getattr(args, "base_url", "") or os.environ.get("BASE_URL", "")
    api_key = getattr(args, "api_key", "") or os.environ.get("API_KEY", "")
    max_retries = int(getattr(args, "max_retries", 3))
    retry_delay = float(getattr(args, "retry_delay", 2.0))
    timeout = float(getattr(args, "timeout_per_request", 120.0))
    temperature = float(getattr(args, "temperature", 0.1))

    if not base_url:
        raise ValueError("缺少 base_url，请通过 --base_url 或环境变量 BASE_URL 指定")
    if not api_key:
        raise ValueError("缺少 api_key，请通过 --api_key 或环境变量 API_KEY 指定")

    return LLMAdapter(
        model_name=model_name,
        api_protocol=api_protocol,
        base_url=base_url,
        api_key=api_key,
        max_retries=max_retries,
        retry_delay=retry_delay,
        timeout=timeout,
        temperature=temperature,
        max_new_tokens=int(getattr(args, "max_new_tokens", 3000)),
    )


def get_global_adapter(args=None) -> LLMAdapter:
    global _global_adapter
    if _global_adapter is None:
        if args is None:
            raise ValueError("首次初始化全局LLMAdapter时需要传入args")
        _global_adapter = _build_adapter_from_args(args)
    return _global_adapter


def promptLLM(
    args,
    prompts: List[str],
    schema: Any = None,
    max_new_tokens: int = 3000,
    json_mode: bool = False,
    temperature: float = 0.1,
    top_p: float = 1.0,
    images_base64: Optional[List[Optional[str]]] = None,
    timeout_per_request: float = 120.0,
    **kwargs
) -> List[str]:
    _ = top_p
    _ = kwargs
    adapter = get_global_adapter(args)

    max_workers = int(getattr(args, "max_workers", 3))
    max_batch_retries = int(getattr(args, "max_batch_retries", 2))

    if len(prompts) == 1:
        image = images_base64[0] if images_base64 else None
        response = adapter.chat(
            prompts[0],
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            json_mode=json_mode,
            schema=schema,
            image_base64=image,
        )
        return [response]

    return adapter.batch_chat(
        prompts,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        json_mode=json_mode,
        schema=schema,
        max_workers=max_workers,
        show_progress=getattr(args, "show_progress", True),
        images_base64=images_base64,
        timeout_per_request=timeout_per_request,
        max_batch_retries=max_batch_retries,
    )


def constructPrompt(args, system_instruction: str, main_prompt: str) -> str:
    _ = args
    return f"{system_instruction}\n\n{main_prompt}"


def initializeLLM(args):
    get_global_adapter(args)
    return args
