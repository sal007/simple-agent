"""Tests for the token and cost counter (usage.py) and where it shows up."""

import io
import json

import pytest

from simple_agent import usage
from simple_agent.agent import Agent
from simple_agent.cli import Printer, _ask, _command
from simple_agent.config import resolve
from simple_agent.evals import Task, run_task
from simple_agent.providers.base import Message, Reply, ToolCall
from simple_agent.subagents import enable as enable_subagents
from simple_agent.trace import Tracer
from simple_agent.usage import UsageMeter


class Scripted:
    name = "fake"

    def __init__(self, *replies, model="claude-opus-5-5"):
        self.model = model
        self.replies = list(replies)

    def chat(self, system, messages, tools, on_text=None):
        return self.replies.pop(0)


def say(text="", calls=(), tokens=(1000, 50)):
    message = Message(role="assistant", content=text, tool_calls=list(calls))
    return Reply(message, usage={"input_tokens": tokens[0], "output_tokens": tokens[1]})


def tool_round():
    return say(calls=[ToolCall("c1", "calculator", {"expression": "6*7"})])


def test_cost_and_the_line():
    turn = {"input_tokens": 4210, "output_tokens": 180, "model_calls": 3}
    total = {"input_tokens": 9876, "output_tokens": 512, "model_calls": 7}
    line = usage.turn_line(turn, total, "claude-opus-5-5", usage.DEFAULT_PRICES)
    # Opus 5.5: $4 in, $20 out per million: 4210*4/1e6 + 180*20/1e6 = 0.02044.
    assert line == ("(this turn: 3 model calls, 4,210 in + 180 out tokens, $0.0204 · "
                    "session: 9,876 in + 512 out tokens, $0.0497)")
    assert usage.cost(turn, "qwen2.5-7b-instruct", usage.DEFAULT_PRICES) is None
    local = usage.turn_line(turn, total, "qwen2.5-7b-instruct", usage.DEFAULT_PRICES)
    assert "$" not in local and "1 model call" not in local
    assert usage.format_dollars(12.345) == "$12.35"


def test_prices_from_config():
    prices = usage.prices_from_config({"my-model": {"input": 1, "output": 2}, "claude-opus-5-5": {"input": 3, "output": 9}})
    assert prices["my-model"] == (1.0, 2.0) and prices["claude-opus-5-5"] == (3.0, 9.0)
    assert prices["claude-haiku-4-5"] == usage.DEFAULT_PRICES["claude-haiku-4-5"]
    with pytest.raises(ValueError, match="input and output"):
        usage.prices_from_config({"my-model": {"input": 1}})


def test_settings():
    settings = resolve({"show_usage": False, "prices": {"m": {"input": 1, "output": 1}}})
    assert settings.show_usage is False and settings.prices["m"] == (1.0, 1.0)
    assert resolve({}).show_usage is True


def test_the_line_after_each_answer_counts_every_call_in_the_turn(capsys):
    agent = Agent(provider=Scripted(tool_round(), say("42"), say("Hi again")), stream=False, events=Printer().events())
    meter = UsageMeter("claude-opus-5-5", dict(usage.DEFAULT_PRICES))

    assert _ask(agent, Printer(), "what is 6*7?", meter)
    out = capsys.readouterr().out
    assert "agent> 42" in out
    assert "(this turn: 2 model calls, 2,000 in + 100 out tokens, $0.0100 · session: 2,000 in + 100 out" in out

    assert _ask(agent, Printer(), "hi", meter)
    assert "(this turn: 1 model call, 1,000 in + 50 out tokens, $0.0050 · session: 3,000 in + 150 out" in capsys.readouterr().out


def test_sub_agent_calls_count_toward_the_turn(capsys):
    delegate = say(calls=[ToolCall("d1", "delegate", {"task": "add up"})])
    agent = Agent(provider=Scripted(delegate, tool_round(), say("helper: 42"), say("It is 42.")), stream=False)
    enable_subagents(agent)
    meter = UsageMeter("claude-opus-5-5")
    assert _ask(agent, Printer(), "go", meter)
    assert "(this turn: 4 model calls, 4,000 in + 200 out tokens" in capsys.readouterr().out


def test_switching_it_off_and_on(capsys):
    agent = Agent(provider=Scripted(say("one"), say("two"), model="qwen2.5-7b-instruct"), stream=False)
    meter = UsageMeter("qwen2.5-7b-instruct", show=False)
    _ask(agent, Printer(), "1", meter)
    assert "this turn" not in capsys.readouterr().out

    _command(agent, "/usage on", meter=meter)
    _ask(agent, Printer(), "2", meter)
    out = capsys.readouterr().out
    assert "is on" in out and "(this turn: 1 model call, 1,000 in + 50 out tokens · session" in out

    _command(agent, "/usage", meter=meter)
    out = capsys.readouterr().out
    assert "this session: 2 model calls, 2,000 in + 100 out tokens" in out and "no price for 'qwen2.5-7b-instruct'" in out
    _command(agent, "/usage off", meter=meter)
    assert not meter.show


def test_trace_logs_the_turn_total(tmp_path):
    tracer = Tracer(tmp_path, out=io.StringIO())
    agent = Agent(provider=Scripted(tool_round(), say("42")), stream=False)
    log_path = tracer.start(agent.provider, agent.system_prompt, agent.tools.specs())
    agent.events = tracer.events()
    agent.ask("6*7?")
    records = [json.loads(line) for line in log_path.read_text().splitlines()]
    (end,) = [r for r in records if r["event"] == "turn_end"]
    assert end["usage"] == {"input_tokens": 2000, "output_tokens": 100, "model_calls": 2}


def test_eval_results_have_the_cost():
    task = Task("t/x", "6*7?", {"answer_number": 42})
    result = run_task(task, lambda: Agent(provider=Scripted(say("42")), stream=False), log=lambda line: None)
    assert result["model_calls"] == 1 and result["cost_usd"] == pytest.approx(0.005)
    local = run_task(task, lambda: Agent(provider=Scripted(say("42"), model="local"), stream=False), log=lambda line: None)
    assert local["cost_usd"] is None
