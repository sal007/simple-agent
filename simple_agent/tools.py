"""Tools: plain Python functions the model is allowed to call.

A tool is registered with the @tool decorator, which records its name,
description and a JSON Schema for its arguments. The agent sends those
descriptions to the model; when the model asks for a tool, the registry
runs the function and returns its result as text.

Adding a tool is the easiest way to extend the agent. Copy one below.
"""

from __future__ import annotations

import ast
import datetime as _dt
import operator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .providers.base import ToolSpec


@dataclass
class Tool:
    spec: ToolSpec
    func: Callable[..., Any]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def tool(self, description: str, parameters: dict[str, Any] | None = None):
        """Decorator: register `func` as a tool the model can call.

        `parameters` is a JSON Schema object describing the keyword arguments.
        """

        def register(func: Callable[..., Any]) -> Callable[..., Any]:
            schema = parameters or {"type": "object", "properties": {}}
            self._tools[func.__name__] = Tool(ToolSpec(func.__name__, description, schema), func)
            return func

        return register

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
        try:
            return str(tool.func(**arguments))
        except Exception as exc:  # noqa: BLE001 - any failure goes back to the model
            return f"Error: {type(exc).__name__}: {exc}"


# The default set of tools. They are read-only on purpose: a safe base to build on.
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
