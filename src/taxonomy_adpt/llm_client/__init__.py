"""
Taxonomy Adapt 的 LLM 客户端入口。
仅保留标准协议调用适配层（openai / anthropic / dashscope）。
"""

from .llm_adapter import (
    LLMAdapter,
    initializeLLM,
    constructPrompt,
    promptLLM,
    get_global_adapter,
)

__all__ = [
    "LLMAdapter",
    "initializeLLM",
    "constructPrompt",
    "promptLLM",
    "get_global_adapter",
]

