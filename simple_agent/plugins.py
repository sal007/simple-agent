"""Plugins: add tools by dropping a .py file in a folder.

A plugin is an ordinary Python file in one of the plugin folders (by default
./plugins and ~/.config/simple-agent/plugins). It registers tools with the
same decorator as tools.py, imported from here:

    from simple_agent.plugins import tool

    @tool("Count the words in a piece of text.",
          {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]})
    def word_count(text: str) -> str:
        return str(len(text.split()))

When the agent starts, every *.py file in those folders is run and its tools
are added next to the built-in ones. Edit a plugin and type /reload to load
it again without restarting. Files whose name starts with "_" are skipped,
so helpers can live there too.

A plugin is code you chose to install and it runs with your permissions,
like any Python script, so only use plugins you trust or wrote yourself.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .tools import ToolRegistry

# While a plugin file is being run, @tool registers into this registry.
_loading: ToolRegistry | None = None


def tool(description: str, parameters: dict[str, Any] | None = None, confirm: bool = False):
    """Decorator for plugin tools. Same arguments as ToolRegistry.tool in tools.py."""
    if _loading is None:
        raise RuntimeError("simple_agent.plugins.tool can only be used in a plugin file the agent is loading")
    return _loading.tool(description, parameters, confirm)


@dataclass
class Plugin:
    path: Path
    tools: list[str] = field(default_factory=list)
    error: str | None = None  # Set if the file failed to load.


def find(dirs: list[str | Path]) -> list[Path]:
    """All plugin files in the given folders, in name order. Missing folders are fine."""
    files = []
    for d in dirs:
        folder = Path(d).expanduser()
        if folder.is_dir():
            files += sorted(p for p in folder.glob("*.py") if not p.name.startswith("_"))
    return files


def load(dirs: list[str | Path], registry: ToolRegistry) -> list[Plugin]:
    """Run every plugin file and add its tools to `registry`.

    A plugin that fails (a syntax error, a missing import, ...) is reported
    and skipped, so one broken file doesn't stop the agent. A plugin tool
    with the same name as an existing tool replaces it, which is handy for
    trying out a different description or implementation of a built-in.
    """
    global _loading
    plugins = []
    for path in find(dirs):
        plugin = Plugin(path)
        plugins.append(plugin)
        collected = ToolRegistry()
        _loading = collected
        try:
            _run_file(path)
        except Exception as exc:  # noqa: BLE001 - report any failure and carry on
            plugin.error = f"{type(exc).__name__}: {exc}"
            continue
        finally:
            _loading = None
        registry.update(collected)
        plugin.tools = collected.names()
    return plugins


def _run_file(path: Path) -> None:
    """Import a .py file by path, as a fresh module each time (so /reload sees edits)."""
    name = f"simple_agent_plugin_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # Lets dataclasses and the like inside the plugin work.
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise


class PluginLoader:
    """Loads plugins on top of a base set of tools, and can do it again for /reload."""

    def __init__(self, dirs: list[str | Path], base: ToolRegistry):
        self.dirs = dirs
        self.base = base  # The built-in tools (and any MCP tools), never changed.
        self.plugins: list[Plugin] = []

    def load(self) -> ToolRegistry:
        """A new registry: the base tools plus freshly loaded plugin tools."""
        registry = self.base.copy()
        self.plugins = load(self.dirs, registry)
        return registry

    def describe(self) -> list[str]:
        """One line per plugin file, for the CLI."""
        lines = []
        for plugin in self.plugins:
            if plugin.error:
                lines.append(f"plugin {plugin.path} failed to load: {plugin.error}")
                continue
            names = [f"{n} (replaces the built-in)" if n in self.base.names() else n for n in plugin.tools]
            lines.append(f"plugin {plugin.path}: {', '.join(names) or 'no tools'}")
        return lines
