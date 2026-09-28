"""The agent loop. This is the heart of the project.

An "agent" here is just this loop:

    1. Add the user's message to the history.
    2. Send the history (plus tool descriptions) to the model.
    3. If the model asked for tools, run them, add the results to the
       history, and go back to step 2.
    4. Otherwise the model has answered: return its text.

Everything else (providers, tools, the CLI) plugs into this.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .providers.base import Message, Provider, Reply, ToolCall
from .tools import ToolRegistry, default_tools

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant running in a terminal. "
    "Use the available tools when they help you give a correct answer, "
    "and answer directly when they don't. Keep answers concise."
)


@dataclass
class AgentEvents:
    """Optional callbacks so a UI (or the tracer in trace.py) can see what the agent is doing.

    They fire in this order for one turn:
      on_turn_start -> (on_model_request -> on_model_reply -> [on_tool_call -> on_tool_result]...)...
      -> on_turn_end, or on_error if something failed.
    """

    on_turn_start: Callable[[str], None] = lambda user_input: None
    on_model_request: Callable[[int, list[Message]], None] = lambda step, messages: None
    on_model_reply: Callable[[int, Reply], None] = lambda step, reply: None
    on_tool_call: Callable[[ToolCall], None] = lambda call: None
    on_tool_result: Callable[[ToolCall, str], None] = lambda call, result: None
    on_turn_end: Callable[[str], None] = lambda answer: None
    on_error: Callable[[BaseException], None] = lambda error: None
    on_reset: Callable[[], None] = lambda: None


@dataclass
class Agent:
    provider: Provider
    tools: ToolRegistry = default_tools
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    max_steps: int = 10  # Most model calls per user message, so a confused model can't loop forever.
    history: list[Message] = field(default_factory=list)
    events: AgentEvents = field(default_factory=AgentEvents)
    usage: dict[str, int] = field(default_factory=lambda: {"input_tokens": 0, "output_tokens": 0})

    def ask(self, user_input: str) -> str:
        """Run one full turn: the user's message in, the model's final answer out."""
        start = len(self.history)
        self.history.append(Message(role="user", content=user_input))
        self.events.on_turn_start(user_input)
        try:
            answer = self._loop()
        except BaseException as exc:
            # If the API call fails (or you press Ctrl+C), drop the half-finished
            # turn so the history stays valid for the next message.
            del self.history[start:]
            self.events.on_error(exc)
            raise
        self.events.on_turn_end(answer)
        return answer

    def _loop(self) -> str:
        for step in range(1, self.max_steps + 1):
            self.events.on_model_request(step, self.history)
            reply = self.provider.chat(self.system_prompt, self.history, self.tools.specs())
            self.events.on_model_reply(step, reply)
            for key, value in reply.usage.items():
                self.usage[key] = self.usage.get(key, 0) + value
            self.history.append(reply.message)

            if not reply.message.tool_calls:
                return reply.message.content

            for call in reply.message.tool_calls:
                self.events.on_tool_call(call)
                result = self.tools.run(call.name, call.arguments)
                self.events.on_tool_result(call, result)
                self.history.append(Message(role="tool", content=result, tool_call_id=call.id))

        return f"(Stopped after {self.max_steps} steps without a final answer.)"

    def reset(self) -> None:
        """Forget the conversation (the system prompt and tools stay)."""
        self.history.clear()
        self.events.on_reset()
