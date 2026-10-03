"""Context management: keep the history small enough for the model.

Every request resends the whole conversation, so it grows with each turn and
each tool result. Past the model's context window the request fails (or a
local server silently cuts off the start). This module tries two ways to
stay under a token limit, cheapest first:

  1. Clear old tool results. Tool output (a file, a directory listing) is
     often the biggest thing in the history and is rarely needed again once
     the model has used it. We keep the most recent few and replace the rest
     with a short placeholder. The model still sees that the call happened.

  2. Compact (summarize). If the history is still too big, ask the model to
     summarize the older turns, then replace them with that summary. The
     most recent turns are kept word for word.

Clearing is checked before every model call. Compaction only runs at the
start of a turn, never in the middle of a tool round, so it never has to cut
a tool call apart from its result.

Token counts here are estimates (about 4 characters per token), which is
good enough to decide when to act. The real numbers are in /usage.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

from .providers.base import Message

if TYPE_CHECKING:
    from .agent import Agent

CLEARED_PREFIX = "[old tool result cleared to save space"
SUMMARY_HEADER = "Here is a summary of our conversation so far (older messages were removed to save space):"

# Anthropic's recommended prompt for summarizing a conversation so a fresh
# context window can carry on with it.
SUMMARY_PROMPT = (
    "Summarize the transcript inside <summary> tags. Include relevant information in the summary such that "
    "this conversation will be continued by a new context window without needing to redo work or be "
    "reprovided with relevant constraints or context. Be sure to preserve: (1) any difficulties or problems "
    "that came up, and how they were handled or resolved; (2) any possibilities, options, or approaches that "
    "were raised, tried, or set aside, and why; (3) anything that was asked for, decided, agreed, ruled out, "
    "or established as a preference, constraint, or boundary - stated exactly; (4) exactly where things stand "
    "now - what has been covered, settled, or completed so far; (5) anything still open, unresolved, promised, "
    "or expected to happen next; (6) specific details that would be hard to reconstruct - names, numbers, "
    "dates, exact wording, links or references - kept exactly. Be complete on these even at the cost of "
    "length; keep everything else concise. Weight the two voices differently: keep what the user said, asked "
    "for, shared, or established carefully and close to their own words; your own explanations and reasoning "
    "can be condensed much further, to what they concluded or produced - as long as nothing in the six items "
    "above is dropped. Do not call any tools while writing this summary; respond with text only."
)


@dataclass
class ContextManager:
    max_tokens: int = 8000  # Act when the estimated request size goes over this.
    clear_tool_results: bool = True  # Strategy 1.
    keep_tool_results: int = 3  # How many of the newest tool results to leave alone.
    compact: bool = True  # Strategy 2.
    keep_recent_turns: int = 2  # User turns kept word for word when compacting.

    def estimate(self, agent: Agent) -> int:
        """Roughly how many tokens the next request will be (about 4 characters per token)."""
        chars = len(agent.system()) + len(json.dumps([asdict(t) for t in agent.tools.specs()]))
        for m in agent.history:
            chars += len(m.content) + sum(len(json.dumps(c.arguments)) + len(c.name) for c in m.tool_calls)
        return chars // 4

    # --- the two hooks the agent calls ----------------------------------------

    def before_turn(self, agent: Agent) -> None:
        """At the start of a turn: clear old tool results, then compact if still too big."""
        self.before_model_call(agent)
        if self.compact and self.estimate(agent) > self.max_tokens:
            self.compact_history(agent)

    def before_model_call(self, agent: Agent) -> None:
        """Before each model call: clear old tool results if the history is too big."""
        if self.clear_tool_results and self.estimate(agent) > self.max_tokens:
            self.clear_old_tool_results(agent)

    # --- strategy 1: clear old tool results ----------------------------------

    def clear_old_tool_results(self, agent: Agent) -> int:
        """Replace all but the newest `keep_tool_results` tool results with a placeholder.

        Returns how many results were cleared. Everything is cleared in one go
        (rather than one at a time) so this happens rarely, in batches.
        """
        results = [m for m in agent.history if m.role == "tool"]
        old = results[: max(len(results) - self.keep_tool_results, 0)]
        before = self.estimate(agent)
        cleared = 0
        for m in old:
            if m.content.startswith(CLEARED_PREFIX):
                continue
            m.content = f"{CLEARED_PREFIX}: it was {len(m.content)} characters]"
            cleared += 1
        if cleared:
            agent.events.on_context(
                "cleared", {"tool_results": cleared, "tokens_before": before, "tokens_after": self.estimate(agent)}
            )
        return cleared

    # --- strategy 2: summarize older turns -----------------------------------

    def compact_history(self, agent: Agent) -> bool:
        """Summarize everything before the last `keep_recent_turns` user turns.

        The older messages are replaced by two: a user message holding the
        summary and a short assistant reply, so the history still alternates
        user, assistant, user... Returns True if the history was compacted.
        """
        user_turns = [i for i, m in enumerate(agent.history) if m.role == "user"]
        if len(user_turns) <= self.keep_recent_turns:
            return False  # Nothing old enough to summarize.
        cut = user_turns[-self.keep_recent_turns] if self.keep_recent_turns > 0 else len(agent.history)
        older, recent = agent.history[:cut], agent.history[cut:]
        if len(older) <= 2 and older[0].content.startswith(SUMMARY_HEADER):
            return False  # Only the previous summary is left; summarizing it again won't help.

        before = self.estimate(agent)
        agent.events.on_context("compacting", {"messages": len(older), "tokens_before": before})
        try:
            # The same system prompt and tools as a normal request, with one
            # extra user message at the end asking for the summary.
            reply = agent.provider.chat(
                agent.system(), older + [Message(role="user", content=SUMMARY_PROMPT)], agent.tools.specs()
            )
        except Exception as exc:  # noqa: BLE001 - a failed summary shouldn't lose the user's turn
            agent.events.on_context("compact_failed", {"error": f"{type(exc).__name__}: {exc}"})
            return False
        for key, value in reply.usage.items():
            agent.usage[key] = agent.usage.get(key, 0) + value
        agent.usage["model_calls"] = agent.usage.get("model_calls", 0) + 1

        match = re.search(r"<summary>(.*?)</summary>", reply.message.content, re.DOTALL)
        summary = (match.group(1) if match else reply.message.content).strip()
        if not summary:
            agent.events.on_context("compact_failed", {"error": "the model returned no summary"})
            return False

        agent.history[:] = [
            Message(role="user", content=f"{SUMMARY_HEADER}\n\n<summary>\n{summary}\n</summary>"),
            Message(role="assistant", content="Thanks, I have the summary and will continue from there."),
            *recent,
        ]
        agent.events.on_context(
            "compacted",
            {"messages": len(older), "tokens_before": before, "tokens_after": self.estimate(agent), "summary": summary},
        )
        return True

