"""Tests for sub-agents: the delegate tool runs a subtask in a fresh agent."""

import io
from pathlib import Path

from simple_agent import evals, subagents
from simple_agent.agent import Agent
from simple_agent.config import resolve
from simple_agent.evals import Task, run_task
from simple_agent.providers.base import Message, Reply, ToolCall
from simple_agent.tools import default_tools
from simple_agent.trace import Tracer


class Scripted:
    """Plays back replies in order, and records the system prompt and tools of each call."""

    name, model = "fake", "fake-model"

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def chat(self, system, messages, tools, on_text=None):
        self.calls.append({"system": system, "messages": list(messages), "tools": [t.name for t in tools]})
        return self.replies.pop(0)


def say(text="", calls=()):
    return Reply(Message(role="assistant", content=text, tool_calls=list(calls)), usage={"input_tokens": 100, "output_tokens": 10})


def call(name, **arguments):
    return ToolCall(f"c-{name}", name, arguments)


def delegating_provider(helper_calls=None):
    """Main agent delegates; the helper reads a file and answers; the main agent answers."""
    return Scripted(
        say(calls=[call("delegate", task="Read big.txt and tell me the secret word.")]),  # main
        say(calls=helper_calls or [call("read_file", path="big.txt")]),  # helper
        say("The secret word is 'pineapple'."),  # helper's answer
        say("It's pineapple."),  # main's answer
    )


def test_delegate_runs_a_fresh_helper_and_returns_only_its_answer(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "big.txt").write_text("filler " * 1000 + "secret: pineapple")
    provider = delegating_provider()
    events = []
    agent = Agent(provider=provider, stream=False)
    subagents.enable(agent)
    agent.events.on_subagent = lambda kind, details: events.append((kind, details))

    assert agent.ask("What's the secret word in big.txt?") == "It's pineapple."

    main_first, helper_first, helper_second, main_second = provider.calls
    # The helper started empty, with its own prompt and no delegate tool.
    assert [m.content for m in helper_first["messages"]] == ["Read big.txt and tell me the secret word."]
    assert helper_first["system"] == subagents.SUBAGENT_PROMPT
    assert "delegate" in main_first["tools"] and "delegate" not in helper_first["tools"]
    # The file only ever went into the helper's history...
    assert "filler" in helper_second["messages"][-1].content
    # ...while the main agent only got the short answer back.
    assert main_second["messages"][-1].content == "The secret word is 'pineapple'."
    assert all("filler" not in m.content for m in agent.history)
    # The helper's tokens count toward the session.
    assert agent.usage == {"input_tokens": 400, "output_tokens": 40, "model_calls": 4}
    assert [kind for kind, _ in events] == ["start", "tool_call", "tool_result", "end"]
    assert events[-1][1]["steps"] == 2


def test_enabling_does_not_change_the_shared_tool_list():
    agent = Agent(provider=Scripted())
    subagents.enable(agent)
    assert "delegate" in agent.tools.names() and "delegate" not in default_tools.names()


def test_a_helper_cannot_delegate_again(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    provider = delegating_provider(helper_calls=[call("delegate", task="go deeper")])
    agent = Agent(provider=provider, stream=False)
    subagents.enable(agent)
    agent.ask("start")
    helper_tool_result = provider.calls[2]["messages"][-1].content
    assert "unknown tool 'delegate'" in helper_tool_result


def test_trace_shows_the_helper(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "big.txt").write_text("secret: pineapple")
    out = io.StringIO()
    tracer = Tracer(tmp_path / "traces", out=out)
    agent = Agent(provider=delegating_provider(), stream=False)
    subagents.enable(agent)
    agent.events = tracer.events()
    tracer.start(agent.provider, agent.system_prompt, agent.tools.specs())
    agent.ask("go")
    text = out.getvalue()
    assert "sub-agent started: Read big.txt" in text
    assert "sub-agent tool call: read_file" in text
    assert "sub-agent finished in 2 steps (220 tokens)" in text
    assert '"event": "subagent"' in tracer.log_path.read_text()


def test_eval_counts_the_helpers_tools_and_records_it(tmp_path):
    task = Task("t/secret", "Find the secret", {"answer_contains": "pineapple", "tool_used": "read_file"},
                files={"big.txt": "secret: pineapple"})

    def make_agent():
        agent = Agent(provider=delegating_provider(), tools=default_tools.copy(), stream=False)
        subagents.enable(agent)
        return agent

    lines = []
    result = run_task(task, make_agent, log=lines.append)
    assert result["passed"]
    assert result["tools_called"] == ["delegate", "read_file"]
    assert result["subagents"][0]["answer"] == "The secret word is 'pineapple'."
    assert "1 sub-agents" in lines[0]


def test_switching_it_off():
    assert resolve({}).subagents is True
    assert resolve({"subagents": False}).subagents is False


def test_the_many_files_tasks_load():
    tasks = evals.load_tasks([Path(__file__).parent.parent / "evals" / "many_files.toml"])
    assert [t.id for t in tasks] == ["many_files/total-across-reports", "many_files/find-the-error"]
