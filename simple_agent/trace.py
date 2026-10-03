"""Trace mode: watch every step of the agent loop, and keep a log of it.

With --trace the tracer does two things:

  1. Prints each step to the terminal: what is sent to the model, what comes
     back (text, tool calls, stop reason, tokens, time), and every tool result.
  2. Appends the same events to a JSONL log file (one JSON object per line) in
     traces/, so runs can be read back or compared across models later.

Every request resends the whole history, so each "request" event only lists
the messages added since the previous request. Together with the "run_start"
event (system prompt and tool list) that is enough to rebuild any request.
"""

from __future__ import annotations

import datetime as _dt
import json
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import TextIO

from .agent import AgentEvents
from .providers.base import Message, Provider, Reply, ToolCall, ToolSpec


class Tracer:
    def __init__(self, log_dir: str | Path = "traces", out: TextIO = sys.stdout):
        self.log_dir = Path(log_dir)
        self.out = out
        self.log_path: Path | None = None
        self._sent = 0  # How many history messages the model has already seen.
        self._turn_base = 0  # History length when the current turn started.
        self._started = 0.0  # When the current model call or tool call started.

    def start(self, provider: Provider, system_prompt: str, tools: list[ToolSpec]) -> Path:
        """Open a new log file for this run and record the setup."""
        stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        model = re.sub(r"[^A-Za-z0-9._-]+", "_", provider.model)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / f"{stamp}-{provider.name}-{model}.jsonl"
        self._log(
            "run_start",
            provider=provider.name,
            model=provider.model,
            system_prompt=system_prompt,
            tools=[asdict(t) for t in tools],
        )
        self._print(f"logging to {self.log_path}")
        return self.log_path

    def events(self) -> AgentEvents:
        """The callbacks to hand to Agent(events=...)."""
        return AgentEvents(
            on_turn_start=self.turn_start,
            on_model_request=self.model_request,
            on_model_reply=self.model_reply,
            on_tool_call=self.tool_call,
            on_tool_result=self.tool_result,
            on_turn_end=self.turn_end,
            on_error=self.error,
            on_reset=self.reset,
            on_context=self.context,
            on_subagent=self.subagent,
            on_plan=self.plan,
        )

    # --- the callbacks, in the order the agent calls them ---------------------

    def turn_start(self, user_input: str) -> None:
        self._turn_base = self._sent
        self._turn_usage: dict[str, int] = {}  # Tokens and model calls this turn, logged at turn_end.
        self._log("turn_start", user_input=user_input)

    def model_request(self, step: int, messages: list[Message]) -> None:
        new = messages[self._sent :]
        self._sent = len(messages)
        self._print(f"step {step}: sending {_count(len(messages), 'message')} to the model ({len(new)} new)")
        for m in new:
            self._print(f"  + {_describe(m)}")
        self._log("request", step=step, history_length=len(messages), new_messages=[_message_dict(m) for m in new])
        self._started = time.perf_counter()

    def model_reply(self, step: int, reply: Reply) -> None:
        seconds = time.perf_counter() - self._started
        self._sent += 1  # The reply joins the history the model has seen.
        tokens = ", ".join(f"{k.replace('_tokens', '')} {v}" for k, v in reply.usage.items()) or "no usage reported"
        self._print(f"step {step}: reply in {seconds:.2f}s, stop={reply.stop_reason or '?'}, tokens: {tokens}")
        if reply.message.content:
            self._print(f"  text: {_short(reply.message.content, 300)}")
        self._add_usage({**reply.usage, "model_calls": 1})
        self._log(
            "reply",
            step=step,
            seconds=round(seconds, 3),
            stop_reason=reply.stop_reason,
            usage=reply.usage,
            message=_message_dict(reply.message),
        )

    def tool_call(self, call: ToolCall) -> None:
        self._print(f"  tool call: {call.name}({json.dumps(call.arguments)})")
        self._started = time.perf_counter()

    def tool_result(self, call: ToolCall, result: str) -> None:
        seconds = time.perf_counter() - self._started
        self._print(f"  tool result ({seconds * 1000:.1f} ms): {_short(result, 300)}")
        self._log("tool", name=call.name, arguments=call.arguments, result=result, seconds=round(seconds, 4))

    def turn_end(self, answer: str) -> None:
        self._log("turn_end", answer=answer, usage=getattr(self, "_turn_usage", {}))

    def _add_usage(self, usage: dict) -> None:
        turn = self.__dict__.setdefault("_turn_usage", {})
        for key in ("input_tokens", "output_tokens", "model_calls"):
            if key in usage:
                turn[key] = turn.get(key, 0) + usage[key]

    def error(self, error: BaseException) -> None:
        self._sent = self._turn_base  # The agent drops a failed turn from its history.
        self._log("error", error=f"{type(error).__name__}: {error}")

    def reset(self) -> None:
        self._sent = self._turn_base = 0
        self._log("reset")

    def context(self, kind: str, details: dict) -> None:
        self._print(f"context: {describe_context(kind, details)}")
        self._log("context", kind=kind, **details)
        if kind in ("cleared", "compacted"):
            # Earlier messages changed, so list the whole history again on the
            # next request: that is exactly what the model will now see.
            self._sent = 0

    def subagent(self, kind: str, details: dict) -> None:
        # The helper's own model calls aren't listed step by step (its history
        # never joins this one), but its tool calls and its answer are.
        self._print(f"  sub-agent {describe_subagent(kind, details, 300)}")
        self._log("subagent", kind=kind, **details)
        if kind == "end":
            self._add_usage(details)  # The helper's tokens count toward this turn.

    def plan(self, steps: list[dict]) -> None:
        # The plan is also added to the system prompt from now on.
        self._print("  plan updated:")
        for line in render_plan(steps).splitlines():
            self._print(f"    {line}")
        self._log("plan", steps=steps)

    # --- helpers --------------------------------------------------------------

    def _print(self, text: str) -> None:
        print(f"  trace> {text}", file=self.out)

    def _log(self, event: str, **fields) -> None:
        if self.log_path is None:
            return
        record = {"time": _dt.datetime.now().astimezone().isoformat(timespec="milliseconds"), "event": event, **fields}
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def describe_context(kind: str, details: dict) -> str:
    """One line about a context-management event (shared with the CLI)."""
    before, after = details.get("tokens_before"), details.get("tokens_after")
    if kind == "cleared":
        return f"cleared {_count(details['tool_results'], 'old tool result')} (~{before} -> ~{after} tokens)"
    if kind == "compacting":
        return f"history is ~{before} tokens; summarizing {_count(details['messages'], 'older message')}..."
    if kind == "compacted":
        return f"replaced {_count(details['messages'], 'older message')} with a summary (~{before} -> ~{after} tokens)"
    if kind == "compact_failed":
        return f"could not summarize, keeping the full history ({details['error']})"
    return f"{kind} {details}"


def render_plan(steps: list[dict]) -> str:
    marks = {"pending": "[ ]", "in_progress": "[>]", "done": "[x]"}
    return "\n".join(f"{marks.get(s['status'], '[?]')} {s['step']}" for s in steps) or "(empty plan)"


def describe_subagent(kind: str, details: dict, limit: int = 120) -> str:
    """One line about what a sub-agent is doing (shared with the CLI)."""
    if kind == "start":
        return f"started: {_short(details['task'], limit)}"
    if kind == "tool_call":
        return f"tool call: {details['name']}({json.dumps(details['arguments'])})"
    if kind == "tool_result":
        return f"tool result: {_short(details['result'], limit)}"
    if kind == "end":
        tokens = details.get("input_tokens", 0) + details.get("output_tokens", 0)
        return f"finished in {_count(details['steps'], 'step')} ({tokens} tokens): {_short(details['answer'], limit)}"
    return f"{kind} {details}"


def _message_dict(m: Message) -> dict:
    """A Message as plain JSON, without the provider's raw reply object."""
    d = {"role": m.role, "content": m.content}
    if m.tool_calls:
        d["tool_calls"] = [asdict(tc) for tc in m.tool_calls]
    if m.tool_call_id:
        d["tool_call_id"] = m.tool_call_id
    return d


def _describe(m: Message) -> str:
    if m.role == "tool":
        return f"[tool result] {_short(m.content)}"
    calls = "".join(f" [calls {c.name}]" for c in m.tool_calls)
    return f"[{m.role}] {_short(m.content)}{calls}"


def _count(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def _short(text: str, limit: int = 120) -> str:
    text = text.replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 3] + "..."
