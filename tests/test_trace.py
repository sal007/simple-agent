import io
import json

import pytest

from simple_agent.agent import Agent
from simple_agent.providers.base import Message, Reply, ToolCall
from simple_agent.trace import Tracer


class ScriptedProvider:
    name, model = "fake", "fake/model:1"

    def __init__(self, replies):
        self.replies = list(replies)

    def chat(self, system, messages, tools, on_text=None):
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def assistant(text="", tool_calls=(), stop="end_turn"):
    message = Message(role="assistant", content=text, tool_calls=list(tool_calls))
    return Reply(message, stop_reason=stop, usage={"input_tokens": 10, "output_tokens": 3})


def traced_agent(tmp_path, replies):
    out = io.StringIO()
    tracer = Tracer(tmp_path, out=out)
    agent = Agent(provider=ScriptedProvider(replies))
    log_path = tracer.start(agent.provider, agent.system_prompt, agent.tools.specs())
    agent.events = tracer.events()
    return agent, out, log_path


def read_log(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_trace_prints_and_logs_every_step(tmp_path):
    agent, out, log_path = traced_agent(tmp_path, [
        assistant(tool_calls=[ToolCall("c1", "calculator", {"expression": "6*7"})], stop="tool_use"),
        assistant("It's 42."),
    ])
    assert agent.ask("what is 6*7?") == "It's 42."

    assert log_path.name.endswith("-fake-fake_model_1.jsonl")
    events = read_log(log_path)
    assert [e["event"] for e in events] == [
        "run_start", "turn_start", "request", "reply", "tool", "request", "reply", "turn_end",
    ]
    assert {t["name"] for t in events[0]["tools"]} >= {"calculator"}
    # Each request lists only what's new since the last one.
    assert [m["role"] for m in events[2]["new_messages"]] == ["user"]
    assert [m["role"] for m in events[5]["new_messages"]] == ["tool"]
    assert events[5]["history_length"] == 3
    assert events[3]["message"]["tool_calls"][0]["name"] == "calculator"
    assert events[4]["result"] == "42"
    assert events[6]["usage"] == {"input_tokens": 10, "output_tokens": 3}

    printed = out.getvalue()
    assert "step 1: sending 1 message to the model" in printed
    assert "stop=tool_use, tokens: input 10, output 3" in printed
    assert "tool result" in printed and "42" in printed


def test_trace_logs_errors_and_recovers(tmp_path):
    agent, out, log_path = traced_agent(tmp_path, [RuntimeError("server down"), assistant("hi")])
    with pytest.raises(RuntimeError):
        agent.ask("first")
    assert agent.ask("second") == "hi"

    events = read_log(log_path)
    assert events[3] == {**events[3], "event": "error", "error": "RuntimeError: server down"}
    # The failed turn was dropped, so the retry sends one message and it counts as new.
    retry = [e for e in events if e["event"] == "request"][-1]
    assert retry["history_length"] == 1 and len(retry["new_messages"]) == 1


def test_reset_starts_a_fresh_history_in_the_log(tmp_path):
    agent, out, log_path = traced_agent(tmp_path, [assistant("a"), assistant("b")])
    agent.ask("one")
    agent.reset()
    agent.ask("two")

    events = read_log(log_path)
    assert "reset" in [e["event"] for e in events]
    last = [e for e in events if e["event"] == "request"][-1]
    assert [m["content"] for m in last["new_messages"]] == ["two"]
