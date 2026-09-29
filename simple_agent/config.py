"""Settings: built-in defaults, overridden by a TOML file, overridden by CLI flags."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

from .agent import DEFAULT_SYSTEM_PROMPT

# Sensible starting points for each provider. LM Studio is the default because
# it runs locally and needs no account.
PROVIDER_DEFAULTS = {
    "openai": {"base_url": "http://localhost:1234/v1", "model": "local-model", "api_key_env": "OPENAI_API_KEY"},
    "anthropic": {"base_url": None, "model": "claude-opus-5-5", "api_key_env": "ANTHROPIC_API_KEY"},
}

CONFIG_LOCATIONS = [Path("config.toml"), Path.home() / ".config" / "simple-agent" / "config.toml"]


@dataclass
class Settings:
    provider: str
    model: str
    base_url: str | None
    api_key: str | None
    system_prompt: str
    max_steps: int
    trace: bool = False
    trace_dir: str = "traces"


def load_file(path: str | None) -> dict:
    """Read the first config file that exists (or the one given with --config)."""
    candidates = [Path(path)] if path else CONFIG_LOCATIONS
    for candidate in candidates:
        if candidate.is_file():
            with candidate.open("rb") as f:
                return tomllib.load(f)
    if path:
        raise FileNotFoundError(f"Config file not found: {path}")
    return {}


def resolve(
    file_config: dict,
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
) -> Settings:
    """Merge the layers. Anything passed in explicitly (from CLI flags) wins."""
    provider = provider or file_config.get("provider", "openai")
    if provider not in PROVIDER_DEFAULTS:
        raise ValueError(f"Unknown provider {provider!r}. Choose one of: {', '.join(PROVIDER_DEFAULTS)}")

    defaults = PROVIDER_DEFAULTS[provider]
    section = file_config.get("providers", {}).get(provider, {})

    return Settings(
        provider=provider,
        model=model or section.get("model") or defaults["model"],
        base_url=base_url or section.get("base_url") or defaults["base_url"],
        # Prefer the environment for secrets so they don't end up in a file.
        api_key=os.environ.get(defaults["api_key_env"]) or section.get("api_key"),
        system_prompt=file_config.get("system_prompt", DEFAULT_SYSTEM_PROMPT),
        max_steps=int(file_config.get("max_steps", 10)),
        trace=bool(file_config.get("trace", False)),
        trace_dir=file_config.get("trace_dir", "traces"),
    )
