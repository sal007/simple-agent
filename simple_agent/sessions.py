"""Saved sessions: write a conversation to a JSON file and load it back later.

A session file is the agent's history plus a little context (provider,
model, token usage). Because the history is provider-neutral, a session
saved with one provider can be resumed with another.
"""

from __future__ import annotations

import datetime as _dt
import json
import re
from pathlib import Path
from typing import Any

from .agent import Agent
from .providers.base import Message, ToolCall

FORMAT_VERSION = 1


def session_path(sessions_dir: str | Path, name: str) -> Path:
    """Turn a session name like "tea-research" into sessions/tea-research.json.

    Anything that already looks like a path (has a slash or ends in .json)
    is used as it is.
    """
    if "/" in name or "\\" in name or name.endswith(".json"):
        return Path(name)
    if not re.fullmatch(r"[A-Za-z0-9._-]+", name):
        raise ValueError(f"Session names can only use letters, digits, '.', '_' and '-': {name!r}")
    return Path(sessions_dir) / f"{name}.json"


def save(agent: Agent, path: Path) -> Path:
    data = {
        "version": FORMAT_VERSION,
        "saved_at": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "provider": agent.provider.name,
        "model": agent.provider.model,
        "usage": agent.usage,
        "history": [_message_to_dict(m) for m in agent.history],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load(agent: Agent, path: Path) -> dict[str, Any]:
    """Replace the agent's conversation with the one saved in `path`."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("version") != FORMAT_VERSION:
        raise ValueError(f"{path} is not a simple-agent session file (version {data.get('version')!r})")
    agent.reset()
    agent.history.extend(_message_from_dict(d) for d in data["history"])
    agent.usage = dict(data.get("usage", {}))
    return data


def list_sessions(sessions_dir: str | Path) -> list[tuple[str, dict[str, Any]]]:
    """(name, summary) for each saved session, newest first."""
    found = []
    for path in sorted(Path(sessions_dir).glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        first = next((m["content"] for m in data.get("history", []) if m.get("role") == "user"), "")
        found.append((path.stem, {
            "saved_at": data.get("saved_at", ""),
            "model": f"{data.get('provider', '?')} · {data.get('model', '?')}",
            "messages": len(data.get("history", [])),
            "first_message": first,
        }))
    return found


def _message_to_dict(m: Message) -> dict[str, Any]:
    d: dict[str, Any] = {"role": m.role, "content": m.content}
    if m.tool_calls:
        d["tool_calls"] = [{"id": c.id, "name": c.name, "arguments": c.arguments} for c in m.tool_calls]
    if m.tool_call_id:
        d["tool_call_id"] = m.tool_call_id
    raw = _raw_to_json(m.raw)
    if raw is not None:
        # The provider's own copy of the reply (Anthropic keeps thinking blocks
        # here), so a resumed Claude conversation replays it exactly.
        d["raw"] = raw
    return d


def _message_from_dict(d: dict[str, Any]) -> Message:
    return Message(
        role=d["role"],
        content=d.get("content", ""),
        tool_calls=[ToolCall(c["id"], c["name"], c["arguments"]) for c in d.get("tool_calls", [])],
        tool_call_id=d.get("tool_call_id"),
        raw=d.get("raw"),
    )


def _raw_to_json(raw: Any) -> Any:
    """Provider replies are SDK objects; turn them into plain JSON if we can."""
    if raw is None:
        return None
    if isinstance(raw, list):
        items = [_raw_to_json(item) for item in raw]
        return None if any(item is None for item in items) else items
    if hasattr(raw, "model_dump"):  # Pydantic models, like the Anthropic SDK's content blocks.
        return raw.model_dump(mode="json", exclude_unset=True)
    if isinstance(raw, (dict, str, int, float, bool)):
        return raw
    return None  # Unknown type: skip it; the provider rebuilds the turn from the plain fields.
