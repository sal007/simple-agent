"""Tests for loading tools from plugin files."""

import textwrap
from pathlib import Path

import pytest

from simple_agent.cli import _command
from simple_agent.agent import Agent
from simple_agent.config import resolve
from simple_agent.plugins import PluginLoader, load, tool
from simple_agent.tools import ToolRegistry, default_tools

REPO_PLUGINS = Path(__file__).parent.parent / "plugins"


def write(folder, name, code):
    (folder / name).write_text(textwrap.dedent(code))


GREET = """
    from simple_agent.plugins import tool

    @tool("Say hello.", {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]})
    def greet(name):
        return f"Hello, {name}!"
"""


def test_loads_tools_from_a_folder(tmp_path):
    write(tmp_path, "greet.py", GREET)
    write(tmp_path, "_helper.py", "raise RuntimeError('should be skipped')")
    registry = ToolRegistry()
    plugins = load([tmp_path, tmp_path / "missing"], registry)
    assert [p.path.name for p in plugins] == ["greet.py"]
    assert plugins[0].tools == ["greet"]
    assert registry.run("greet", {"name": "Stem"}) == "Hello, Stem!"


def test_a_broken_plugin_is_reported_and_skipped(tmp_path):
    write(tmp_path, "a_broken.py", "def oops(:\n")
    write(tmp_path, "b_greet.py", GREET)
    registry = ToolRegistry()
    broken, ok = load([tmp_path], registry)
    assert broken.error.startswith("SyntaxError") and broken.tools == []
    assert ok.error is None and registry.names() == ["greet"]


def test_plugin_tools_can_ask_first_and_replace_built_ins(tmp_path):
    write(tmp_path, "tools.py", """
        from simple_agent.plugins import tool

        @tool("Delete everything.", confirm=True)
        def nuke():
            return "boom"

        @tool("A better calculator.")
        def calculator():
            return "plugin version"
    """)
    loader = PluginLoader([tmp_path], default_tools)
    registry = loader.load()
    assert registry.asks_first("nuke") and "declined" in registry.run("nuke", {})
    assert registry.run("calculator", {}) == "plugin version"
    assert default_tools.run("calculator", {"expression": "1+1"}) == "2"  # The built-in list is untouched.
    assert "calculator (replaces the built-in)" in loader.describe()[0]


def test_reload_picks_up_edits_and_removed_files(tmp_path, capsys):
    write(tmp_path, "greet.py", GREET)
    loader = PluginLoader([tmp_path], ToolRegistry())
    agent = Agent(provider=None, tools=loader.load())
    assert agent.tools.run("greet", {"name": "a"}) == "Hello, a!"

    write(tmp_path, "greet.py", GREET.replace("Hello", "Hi"))
    _command(agent, "/reload", loader=loader)
    assert agent.tools.run("greet", {"name": "a"}) == "Hi, a!"

    (tmp_path / "greet.py").unlink()
    _command(agent, "/reload", loader=loader)
    assert "greet" not in agent.tools.names()
    assert "(no plugins found)" in capsys.readouterr().out


def test_decorator_only_works_while_loading():
    with pytest.raises(RuntimeError):
        tool("Not in a plugin.")


def test_the_example_plugin():
    registry = ToolRegistry()
    (plugin,) = load([REPO_PLUGINS], registry)
    assert plugin.tools == ["word_count"]
    assert registry.run("word_count", {"text": "the cat and the hat", "top": 1}) == "5 words. Most common: the (2)"


def test_config():
    assert resolve({}).plugin_dirs == ["plugins", "~/.config/simple-agent/plugins"]
    assert resolve({"plugin_dirs": ["my_tools"]}).plugin_dirs == ["my_tools"]
