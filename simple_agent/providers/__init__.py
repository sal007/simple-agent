"""Providers turn the agent's neutral conversation into real API calls.

To add a new backend, write a class with `name`, `model` and a `chat()`
method (see base.Provider) and register it in `create_provider` below.
"""

from __future__ import annotations

from .base import Message, Provider, Reply, ToolCall, ToolSpec

PROVIDERS = ("openai", "anthropic")


def create_provider(name: str, model: str, base_url: str | None = None, api_key: str | None = None) -> Provider:
    # Imports live inside the branches so you only need the SDK you actually use.
    if name == "openai":
        from .openai_compat import OpenAICompatibleProvider

        return OpenAICompatibleProvider(model=model, base_url=base_url, api_key=api_key)
    if name == "anthropic":
        from .anthropic_provider import AnthropicProvider

        return AnthropicProvider(model=model, api_key=api_key)
    raise ValueError(f"Unknown provider {name!r}. Choose one of: {', '.join(PROVIDERS)}")


__all__ = ["Message", "Provider", "Reply", "ToolCall", "ToolSpec", "PROVIDERS", "create_provider"]
