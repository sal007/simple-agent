"""Tools: plain Python functions the model is allowed to call.

A tool is registered with the @tool decorator, which records its name,
description and a JSON Schema for its arguments. The agent sends those
descriptions to the model; when the model asks for a tool, the registry
runs the function and returns its result as text.

Tools that change things (writing files, running commands) are registered
with confirm=True. Before running one, the registry asks the `approve`
callback, which the CLI wires to a y/N prompt. Nothing runs without a yes.

Adding a tool is the easiest way to extend the agent. Copy one below.
"""

from __future__ import annotations

import ast
import datetime as _dt
import operator
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .providers.base import ToolSpec


@dataclass
class Tool:
    spec: ToolSpec
    func: Callable[..., Any]
    confirm: bool = False  # Ask the user before every call.


# Given a tool name and its arguments, return True to allow the call.
Approver = Callable[[str, dict[str, Any]], bool]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        # Who decides whether a confirm=True tool may run. With no approver
        # (e.g. in tests or scripts), those tools are always declined.
        self.approve: Approver | None = None

    def tool(self, description: str, parameters: dict[str, Any] | None = None, confirm: bool = False):
        """Decorator: register `func` as a tool the model can call.

        `parameters` is a JSON Schema object describing the keyword arguments.
        `confirm=True` means the user must approve each call.
        """

        def register(func: Callable[..., Any]) -> Callable[..., Any]:
            schema = parameters or {"type": "object", "properties": {}}
            self._tools[func.__name__] = Tool(ToolSpec(func.__name__, description, schema), func, confirm)
            return func

        return register

    def asks_first(self, name: str) -> bool:
        return name in self._tools and self._tools[name].confirm

    def specs(self) -> list[ToolSpec]:
        return [t.spec for t in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools)

    def run(self, name: str, arguments: dict[str, Any]) -> str:
        """Run a tool and always return a string, even on failure.

        Errors are returned to the model as text rather than raised, so the
        model can see what went wrong and try again.
        """
        tool = self._tools.get(name)
        if tool is None:
            return f"Error: unknown tool {name!r}. Available tools: {', '.join(self._tools)}"
        if "__invalid_json__" in arguments:
            return f"Error: arguments were not valid JSON: {arguments['__invalid_json__']}"
        if tool.confirm and not (self.approve and self.approve(name, arguments)):
            return "The user declined this tool call. Don't retry it; ask the user what they would like instead."
        try:
            return str(tool.func(**arguments))
        except Exception as exc:  # noqa: BLE001 - any failure goes back to the model
            return f"Error: {type(exc).__name__}: {exc}"


# The default set of tools. The first four only read; the last two change
# things, so they ask first.
default_tools = ToolRegistry()


@default_tools.tool("Get the current local date and time.")
def get_current_time() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


@default_tools.tool(
    "Evaluate an arithmetic expression, e.g. '(2 + 3) * 4 ** 2 / 7'. "
    "Supports + - * / // % ** and parentheses.",
    {
        "type": "object",
        "properties": {"expression": {"type": "string", "description": "The expression to evaluate."}},
        "required": ["expression"],
    },
)
def calculator(expression: str) -> str:
    return str(_safe_eval(ast.parse(expression, mode="eval").body))


@default_tools.tool(
    "List the files and folders in a directory.",
    {
        "type": "object",
        "properties": {"path": {"type": "string", "description": "Directory path. Defaults to '.'."}},
    },
)
def list_files(path: str = ".") -> str:
    entries = sorted(Path(path).iterdir())
    lines = [f"{p.name}/" if p.is_dir() else p.name for p in entries]
    return "\n".join(lines) or "(empty directory)"


@default_tools.tool(
    "Read a text file and return its contents (truncated to max_chars).",
    {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file."},
            "max_chars": {"type": "integer", "description": "Maximum characters to return. Defaults to 10000."},
        },
        "required": ["path"],
    },
)
def read_file(path: str, max_chars: int = 10_000) -> str:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    if len(text) > max_chars:
        return text[:max_chars] + f"\n... [truncated, {len(text) - max_chars} more characters]"
    return text


@default_tools.tool(
    "Write text to a file, creating it (and any missing folders) or replacing its contents.",
    {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file."},
            "content": {"type": "string", "description": "The full text to write."},
        },
        "required": ["path", "content"],
    },
    confirm=True,
)
def write_file(path: str, content: str) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} characters to {target}"


@default_tools.tool(
    "Run a shell command in the current directory and return its exit code and output.",
    {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The command to run."},
            "timeout": {"type": "integer", "description": "Seconds before the command is stopped. Defaults to 60."},
        },
        "required": ["command"],
    },
    confirm=True,
)
def run_shell(command: str, timeout: int = 60) -> str:
    try:
        done = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return f"Error: the command did not finish within {timeout} seconds"
    output = f"exit code: {done.returncode}\nstdout:\n{done.stdout}\nstderr:\n{done.stderr}"
    return output if len(output) <= 10_000 else output[:10_000] + "\n... [truncated]"


# --- helpers ---------------------------------------------------------------

_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node: ast.AST) -> float | int:
    """Evaluate a parsed arithmetic expression without using eval()."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
        left, right = _safe_eval(node.left), _safe_eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 1000:
            raise ValueError("exponent too large")
        return _OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"unsupported expression: {ast.dump(node)}")
