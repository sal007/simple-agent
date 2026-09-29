"""Tests for MCP support, against the tiny stdio server in fake_mcp_server.py."""

import sys
from pathlib import Path

import pytest

from simple_agent.agent import Agent
from simple_agent.config import resolve
from simple_agent.mcp import McpError, McpServer, McpServerConfig, connect
from simple_agent.providers.base import Message, Reply, ToolCall
from simple_agent.tools import ToolRegistry, default_tools

FAKE_SERVER = str(Path(__file__).with_name("fake_mcp_server.py"))


def fake_config(**kwargs):
    return McpServerConfig(name="fake", command=sys.executable, args=[FAKE_SERVER], **kwargs)


@pytest.fixture
def server():
    s = McpServer(fake_config())
    s.start()
    yield s
    s.close()


def test_handshake_and_tool_list(server):
    assert server.server_info["name"] == "fake"
    assert [t["name"] for t in server.list_tools()] == ["add", "fail"]  # Both pages.


def test_call_tool(server):
    assert server.call_tool("add", {"a": 2, "b": 3}) == "5"
    assert server.call_tool("fail", {}) == "Error: something broke"


def test_protocol_errors_are_raised(server):
    with pytest.raises(McpError, match="no method"):
        server.request("resources/list")


def test_tools_join_the_registry_and_ask_first():
    registry = default_tools.copy()
    servers, messages = connect([fake_config()], registry)
    try:
        assert "fake__add" in registry.names() and "fake__add" not in default_tools.names()
        spec = next(s for s in registry.specs() if s.name == "fake__add")
        assert spec.parameters["required"] == ["a", "b"] and "MCP server 'fake'" in spec.description
        assert registry.asks_first("fake__add")
        # No approver means declined, like write_file.
        assert "declined" in registry.run("fake__add", {"a": 1, "b": 1})
        registry.approve = lambda name, args: True
        assert registry.run("fake__add", {"a": 1, "b": 1}) == "2"
        assert messages == ["MCP server 'fake': 2 tools (asks before each call)"]
    finally:
        for s in servers:
            s.close()


def test_trusted_server_runs_without_asking_in_the_agent_loop():
    class Scripted:
        name, model = "fake", "fake-model"

        def __init__(self):
            self.replies = [
                Reply(Message(role="assistant", tool_calls=[ToolCall("c1", "fake__add", {"a": 40, "b": 2})])),
                Reply(Message(role="assistant", content="It's 42.")),
            ]

        def chat(self, system, messages, tools, on_text=None):
            return self.replies.pop(0)

    registry = ToolRegistry()
    servers, _ = connect([fake_config(trusted=True)], registry)
    try:
        agent = Agent(provider=Scripted(), tools=registry, stream=False)
        assert agent.ask("40 + 2?") == "It's 42."
        assert agent.history[2].content == "42"
    finally:
        servers[0].close()


def test_a_broken_server_is_skipped():
    registry = ToolRegistry()
    bad = McpServerConfig(name="bad", command=sys.executable, args=["-c", "pass"], timeout=5)
    missing = McpServerConfig(name="missing", command="no-such-command-xyz")
    servers, messages = connect([bad, missing], registry)
    assert servers == [] and registry.names() == []
    assert "server exited" in messages[0] and "could not start MCP server 'missing'" in messages[1]


def test_config_section():
    settings = resolve({"mcp_servers": {"files": {"command": "npx", "args": ["-y", "server"], "trusted": True}}})
    assert settings.mcp_servers == [McpServerConfig(name="files", command="npx", args=["-y", "server"], trusted=True)]
    with pytest.raises(ValueError, match="needs a command"):
        resolve({"mcp_servers": {"files": {"args": []}}})
