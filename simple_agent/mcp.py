"""MCP support: use tools from outside servers as if they were built in.

MCP (Model Context Protocol) is a standard way for a program to offer tools
to an agent. A server can be written in any language; the agent starts it as
a subprocess and talks to it over its stdin and stdout. Each message is one
line of JSON-RPC 2.0:

    agent  -> server   {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
    server -> agent    {"jsonrpc": "2.0", "id": 1, "result": {"tools": [...]}}

Only four messages are needed to use a server's tools, and this file is a
small hand-written client for exactly those (no MCP library, so you can read
the whole protocol here):

    1. initialize                 - say hello and agree on a protocol version
    2. notifications/initialized  - tell the server we're ready
    3. tools/list                 - get each tool's name, description and JSON Schema
    4. tools/call                 - run a tool and get its result

The tool descriptions have the same shape our own tools use (a name, a
description and a JSON Schema), so they drop straight into the ToolRegistry.
The model can't tell them apart from the built-in ones.

This client only speaks the stdio transport. MCP servers can also run over
HTTP; see the README for ideas on adding that.
"""

from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import threading
from dataclasses import dataclass, field
from typing import Any

from .providers.base import ToolSpec
from .tools import ToolRegistry

PROTOCOL_VERSION = "2025-06-18"


@dataclass
class McpServerConfig:
    """One [mcp_servers.<name>] section of config.toml."""

    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)  # Added to the current environment.
    # Trusted servers' tools run without asking. Otherwise every call asks y/N
    # first, like write_file and run_shell, since we don't know what they do.
    trusted: bool = False
    timeout: float = 60  # Seconds to wait for any one reply.


class McpError(Exception):
    pass


class McpServer:
    """A running MCP server and the JSON-RPC conversation with it."""

    def __init__(self, config: McpServerConfig):
        self.config = config
        self.process: subprocess.Popen | None = None
        self.server_info: dict = {}
        self._next_id = 1
        self._lines: queue.Queue[str | None] = queue.Queue()
        self._lock = threading.Lock()  # One request at a time.

    # --- starting and stopping -------------------------------------------------

    def start(self) -> None:
        """Launch the server and do the initialize handshake."""
        self.process = subprocess.Popen(
            [self.config.command, *self.config.args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,  # Servers log to stderr; keep it out of the chat.
            env={**os.environ, **self.config.env},
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        # A background thread reads stdout line by line, so a request can wait
        # for its reply with a timeout instead of blocking forever.
        threading.Thread(target=self._read_stdout, daemon=True).start()

        result = self.request(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},  # We only use tools, so we offer the server nothing extra.
                "clientInfo": {"name": "simple-agent", "version": "0.1"},
            },
        )
        self.server_info = result.get("serverInfo", {})
        self.notify("notifications/initialized")

    def close(self) -> None:
        if self.process and self.process.poll() is None:
            self.process.stdin.close()  # Closing stdin is how stdio servers are asked to exit.
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()

    # --- the two MCP calls we need -------------------------------------------

    def list_tools(self) -> list[dict]:
        """Every tool the server offers (the list can come in pages)."""
        tools, cursor = [], None
        while True:
            result = self.request("tools/list", {"cursor": cursor} if cursor else {})
            tools += result.get("tools", [])
            cursor = result.get("nextCursor")
            if not cursor:
                return tools

    def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Run a tool and turn its result into text for the model."""
        result = self.request("tools/call", {"name": name, "arguments": arguments})
        parts = []
        for item in result.get("content", []):
            if item.get("type") == "text":
                parts.append(item["text"])
            elif item.get("type") == "resource" and "text" in item.get("resource", {}):
                parts.append(item["resource"]["text"])
            else:  # Images, audio, links: our providers only send text, so describe them.
                parts.append(f"[{item.get('type')} content: {item.get('mimeType') or item.get('uri') or 'not shown'}]")
        if not parts and "structuredContent" in result:
            parts.append(json.dumps(result["structuredContent"]))
        text = "\n".join(parts) or "(no output)"
        # isError means the tool ran but failed; the model should see why.
        return f"Error: {text}" if result.get("isError") else text

    # --- JSON-RPC --------------------------------------------------------------

    def request(self, method: str, params: dict | None = None) -> dict:
        """Send a request and wait for the reply with the same id."""
        with self._lock:
            request_id = self._next_id
            self._next_id += 1
            self._send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params or {}})
            while True:
                try:
                    line = self._lines.get(timeout=self.config.timeout)
                except queue.Empty:
                    raise McpError(f"{self.config.name}: no reply to {method} after {self.config.timeout}s") from None
                if line is None:
                    raise McpError(f"{self.config.name}: the server exited (is the command right?)")
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    continue  # Not JSON-RPC (some servers print banners); skip it.
                if message.get("id") == request_id and "method" not in message:
                    if "error" in message:
                        raise McpError(f"{self.config.name}: {message['error'].get('message', message['error'])}")
                    return message.get("result", {})
                if "method" in message and "id" in message:
                    self._answer_server_request(message)
                # Anything else is a notification (e.g. a log message); ignore it.

    def notify(self, method: str, params: dict | None = None) -> None:
        """A notification is a message without an id: no reply is expected."""
        self._send({"jsonrpc": "2.0", "method": method, **({"params": params} if params else {})})

    def _answer_server_request(self, message: dict) -> None:
        # Servers may ask the client things too. We only answer "ping";
        # everything else (sampling, roots, ...) we say we don't support.
        if message["method"] == "ping":
            self._send({"jsonrpc": "2.0", "id": message["id"], "result": {}})
        else:
            error = {"code": -32601, "message": f"simple-agent does not support {message['method']}"}
            self._send({"jsonrpc": "2.0", "id": message["id"], "error": error})

    def _send(self, message: dict) -> None:
        if self.process is None or self.process.poll() is not None:
            raise McpError(f"{self.config.name}: the server is not running")
        self.process.stdin.write(json.dumps(message) + "\n")
        self.process.stdin.flush()

    def _read_stdout(self) -> None:
        for line in self.process.stdout:
            if line.strip():
                self._lines.put(line)
        self._lines.put(None)  # End of output: the server exited.


def register_tools(server: McpServer, registry: ToolRegistry) -> list[str]:
    """Add all of a server's tools to the registry. Returns the names used.

    Each tool is named "<server>__<tool>" so two servers (or a server and a
    built-in tool) can't clash. Tool names may only use letters, digits, _ and
    - and be at most 64 characters, which both APIs require.
    """
    names = []
    for tool in server.list_tools():
        name = re.sub(r"[^A-Za-z0-9_-]", "_", f"{server.config.name}__{tool['name']}")[:64]
        schema = tool.get("inputSchema") or {"type": "object", "properties": {}}
        description = tool.get("description") or tool.get("title") or tool["name"]
        spec = ToolSpec(name, f"{description} (from MCP server '{server.config.name}')", schema)
        registry.add(spec, _caller(server, tool["name"]), confirm=not server.config.trusted)
        names.append(name)
    return names


def _caller(server: McpServer, tool_name: str):
    """A plain function the registry can call like any built-in tool."""

    def run(**arguments):
        return server.call_tool(tool_name, arguments)

    return run


def connect(configs: list[McpServerConfig], registry: ToolRegistry) -> tuple[list[McpServer], list[str]]:
    """Start every configured server and register its tools.

    A server that fails to start is skipped (and reported) rather than
    stopping the agent. Returns the running servers and one message per server.
    """
    servers, messages = [], []
    for config in configs:
        server = McpServer(config)
        try:
            server.start()
            names = register_tools(server, registry)
        except (OSError, McpError) as exc:
            server.close()
            messages.append(f"could not start MCP server '{config.name}': {exc}")
            continue
        servers.append(server)
        trust = "runs without asking" if config.trusted else "asks before each call"
        messages.append(f"MCP server '{config.name}': {len(names)} tools ({trust})")
    return servers, messages
