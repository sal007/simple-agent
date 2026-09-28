"""Tests for the agent loop, using a scripted fake provider (no network)."""

from simple_agent.agent import Agent
from simple_agent.providers.base import Message, Reply, ToolCall


class ScriptedProvider:
    """Returns pre-written replies in order and records what it was sent."""

    name = "fake"
    model = "fake-model"

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def chat(self, system, messages, tools):
        self.calls.append(list(messages))
        return self.replies.pop(0)


def assistant(text="", tool_calls=()):
    return Reply(Message(role="assistant", content=text, tool_calls=list(tool_calls)), usage={"input_tokens": 1})


def test_plain_answer():
    agent = Agent(provider=ScriptedProvider([assistant("hello")]))
    assert agent.ask("hi") == "hello"
    assert [m.role for m in agent.history] == ["user", "assistant"]


def test_tool_call_then_answer():
    provider = ScriptedProvider([
        assistant(tool_calls=[ToolCall("c1", "calculator", {"expression": "6*7"})]),
        assistant("It's 42."),
    ])
    agent = Agent(provider=provider)
    assert agent.ask("what is 6*7?") == "It's 42."
    tool_msg = agent.history[2]
    assert tool_msg.role == "tool" and tool_msg.content == "42" and tool_msg.tool_call_id == "c1"
    # The second model call saw the tool result.
    assert provider.calls[1][-1].content == "42"
    assert agent.usage["input_tokens"] == 2


def test_max_steps_stops_a_looping_model():
    looping = [assistant(tool_calls=[ToolCall(str(i), "get_current_time", {})]) for i in range(3)]
    agent = Agent(provider=ScriptedProvider(looping), max_steps=3)
    assert "Stopped after 3 steps" in agent.ask("loop")


def test_failed_call_rolls_back_history():
    class Broken:
        name, model = "broken", "x"

        def chat(self, *args):
            raise RuntimeError("server down")

    agent = Agent(provider=Broken())
    try:
        agent.ask("hi")
    except RuntimeError:
        pass
    assert agent.history == []
