"""The provider interface: the one thing every LLM backend has to implement.

The agent never talks to an SDK directly. It keeps the conversation in the
small, provider-neutral format below and hands it to a Provider, which
translates it into whatever its API expects and translates the reply back.
That is what lets the same agent run against LM Studio, OpenAI or Claude.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ToolCall:
    """The model asking us to run one tool."""

    id: str  # Opaque id from the API; we echo it back with the result.
    name: str
    arguments: dict[str, Any]


@dataclass
class Message:
    """One entry in the conversation history.

    role is one of:
      "user"      - something the person typed
      "assistant" - a model reply (text and/or tool calls)
      "tool"      - the result of running a tool (tool_call_id says which call)
    """

    role: str
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    # A provider may stash its native reply here so it can replay it exactly
    # on the next request (Anthropic needs this for its thinking blocks).
    # Other providers ignore it.
    raw: Any = None


@dataclass
class ToolSpec:
    """A tool as described to the model: a name, what it does, and a JSON Schema."""

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass
class Reply:
    """What a provider returns for one model call."""

    message: Message  # Always role="assistant".
    stop_reason: str = ""  # Provider's own reason string, handy for debugging.
    usage: dict[str, int] = field(default_factory=dict)


class Provider(Protocol):
    """Anything with a name, a model and a chat() method is a provider."""

    name: str
    model: str

    def chat(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> Reply:
        """Send the conversation so far and return the model's next message."""
        ...
