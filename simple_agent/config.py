"""Settings: built-in defaults, overridden by a TOML file, overridden by CLI flags."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .agent import DEFAULT_SYSTEM_PROMPT
from .context import ContextManager
from .mcp import McpServerConfig
from .usage import DEFAULT_PRICES, prices_from_config

# Sensible starting points for each provider. LM Studio is the default because
# it runs locally and needs no account.
PROVIDER_DEFAULTS = {
    "openai": {"base_url": "http://localhost:1234/v1", "model": "local-model", "api_key_env": "OPENAI_API_KEY"},
    "anthropic": {"base_url": None, "model": "claude-opus-5-5", "api_key_env": "ANTHROPIC_API_KEY"},
}

# Folders searched for plugin .py files (see plugins.py).
DEFAULT_PLUGIN_DIRS = ["plugins", "~/.config/simple-agent/plugins"]

CONFIG_LOCATIONS = [Path("config.toml"), Path.home() / ".config" / "simple-agent" / "config.toml"]


@dataclass
class Settings:
    provider: str
    model: str
    base_url: str | None
    api_key: str | None
    system_prompt: str
    max_steps: int
    stream: bool = True
    trace: bool = False
    trace_dir: str = "traces"
    sessions_dir: str = "sessions"
    context: ContextManager | None = field(default_factory=ContextManager)
    mcp_servers: list[McpServerConfig] = field(default_factory=list)
    plugin_dirs: list[str] = field(default_factory=lambda: list(DEFAULT_PLUGIN_DIRS))
    subagents: bool = True  # Offer the delegate tool (see subagents.py).
    planning: bool = True  # Offer the update_plan tool (see planning.py).
    plugin_settings: dict[str, dict] = field(default_factory=dict)  # [plugins.<name>] tables.
    project_instructions: bool = True  # Read AGENTS.md files (see instructions.py).
    show_usage: bool = True  # Print tokens (and cost) after each answer (see usage.py).
    prices: dict[str, tuple[float, float]] = field(default_factory=lambda: dict(DEFAULT_PRICES))


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
        stream=bool(file_config.get("stream", True)),
        trace=bool(file_config.get("trace", False)),
        trace_dir=file_config.get("trace_dir", "traces"),
        sessions_dir=file_config.get("sessions_dir", "sessions"),
        context=_context(file_config.get("context", {})),
        mcp_servers=_mcp_servers(file_config.get("mcp_servers", {})),
        plugin_dirs=list(file_config.get("plugin_dirs", DEFAULT_PLUGIN_DIRS)),
        subagents=bool(file_config.get("subagents", True)),
        planning=bool(file_config.get("planning", True)),
        plugin_settings=dict(file_config.get("plugins", {})),
        project_instructions=bool(file_config.get("project_instructions", True)),
        show_usage=bool(file_config.get("show_usage", True)),
        prices=prices_from_config(file_config.get("prices", {})),
    )


def _context(section: dict) -> ContextManager | None:
    """The [context] table. enabled = false turns context management off."""
    if not section.get("enabled", True):
        return None
    known = ContextManager.__dataclass_fields__
    unknown = set(section) - set(known) - {"enabled"}
    if unknown:
        raise ValueError(f"Unknown [context] setting(s): {', '.join(sorted(unknown))}")
    return ContextManager(**{k: v for k, v in section.items() if k in known})


def _mcp_servers(section: dict) -> list[McpServerConfig]:
    """The [mcp_servers.<name>] tables, one per server."""
    servers = []
    for name, table in section.items():
        if "command" not in table:
            raise ValueError(f"[mcp_servers.{name}] needs a command")
        known = McpServerConfig.__dataclass_fields__
        unknown = set(table) - set(known)
        if unknown:
            raise ValueError(f"Unknown [mcp_servers.{name}] setting(s): {', '.join(sorted(unknown))}")
        servers.append(McpServerConfig(name=name, **table))
    return servers
