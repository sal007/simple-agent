"""Tests for the planning tool: the plan is state, shown to the model at every step."""

import io
from pathlib import Path

from simple_agent import planning, subagents
from simple_agent.agent import Agent
from simple_agent.cli import Printer, _command
from simple_agent.config import resolve
from simple_agent.context import ContextManager
from simple_agent.evals import Task, load_tasks, run_task
from simple_agent.providers.base import Message, Reply, ToolCall
from simple_agent.tools import default_tools
from simple_agent.trace import Tracer


class Scripted:
    name, model = "fake", "fake-model"

    def __init__(self, *replies):
        self.replies = list(replies)
        self.systems = []
        self.calls = []

    def chat(self, system, messages, tools, on_text=None):
        self.systems.append(system)
        self.calls.append({"messages": list(messages), "tools": [t.name for t in tools]})
        return self.replies.pop(0)


def say(text="", calls=()):
    return Reply(Message(role="assistant", content=text, tool_calls=list(calls)), usage={"input_tokens": 1})


def plan(*steps):
    items = [{"step": text, "status": status} for text, status in steps]
    return ToolCall("p", "update_plan", {"steps": items})


def planned_run():
    return Scripted(
        say(calls=[plan(("Read the file", "in_progress"), ("Answer", "pending"))]),
        say(calls=[plan(("Read the file", "done"), ("Answer", "in_progress"))]),
        say("Done."),
    )


def test_the_plan_is_added_to_the_system_prompt_at_every_step():
    provider = planned_run()
    agent = Agent(provider=provider, stream=False)
    planner = planning.enable(agent)
    agent.ask("go")

    first, second, third = provider.systems
    assert first == agent.system_prompt  # No plan yet.
    assert second.startswith(agent.system_prompt)
    assert "[>] Read the file\n[ ] Answer" in second
    assert "[x] Read the file\n[>] Answer" in third
    assert planner.steps[1] == {"step": "Answer", "status": "in_progress"}
    assert not planner.done()


def test_the_plan_survives_compaction():
    provider = Scripted(
        say(calls=[plan(("Step one", "done"), ("Step two", "pending"))]),
        say("a" * 4000),
        say("b" * 4000),
        say("<summary>earlier stuff</summary>"),
        say("ok"),
    )
    agent = Agent(provider=provider, stream=False, context=ContextManager(max_tokens=900, keep_recent_turns=1))
    planning.enable(agent)
    agent.ask("one")
    agent.ask("two")
    agent.ask("three")  # Compacts first: the plan's tool call is summarized away...
    assert all("update_plan" not in str(m.tool_calls) for m in agent.history)
    assert "[ ] Step two" in provider.systems[-1]  # ...but the model still sees the plan.


def test_bad_plans_are_rejected_without_changing_the_plan():
    agent = Agent(provider=Scripted(), stream=False)
    planner = planning.enable(agent)
    planner.update_plan([{"step": "Keep me", "status": "pending"}])
    assert planner.update_plan([{"step": "x", "status": "maybe"}]).startswith("Error: item 1 has status 'maybe'")
    assert planner.update_plan([{"status": "done"}]).startswith("Error: item 1 needs a 'step'")
    assert planner.steps == [{"step": "Keep me", "status": "pending"}]


def test_reset_clears_the_plan_and_enable_does_not_touch_shared_tools():
    agent = Agent(provider=Scripted(), stream=False)
    planner = planning.enable(agent)
    planner.update_plan([{"step": "x", "status": "done"}])
    assert planner.done()
    agent.reset()
    assert planner.steps == [] and agent.system() == agent.system_prompt
    assert "update_plan" in agent.tools.names() and "update_plan" not in default_tools.names()


def test_cli_shows_the_plan(capsys):
    agent = Agent(provider=planned_run(), stream=False, events=Printer().events())
    planning.enable(agent)
    agent.ask("go")
    out = capsys.readouterr().out
    assert "  plan> [>] Read the file\n        [ ] Answer" in out
    _command(agent, "/plan")
    assert capsys.readouterr().out == "  [x] Read the file\n  [>] Answer\n"


def test_trace_logs_the_plan(tmp_path):
    out = io.StringIO()
    tracer = Tracer(tmp_path, out=out)
    agent = Agent(provider=planned_run(), stream=False)
    planning.enable(agent)
    agent.events = tracer.events()
    tracer.start(agent.provider, agent.system_prompt, agent.tools.specs())
    agent.ask("go")
    assert "plan updated:" in out.getvalue()
    assert tracer.log_path.read_text().count('"event": "plan"') == 2


def test_a_sub_agent_gets_its_own_plan(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    provider = Scripted(
        say(calls=[plan(("Delegate", "in_progress"))]),
        say(calls=[ToolCall("d", "delegate", {"task": "do it"})]),
        say(calls=[plan(("Helper step", "in_progress"))]),  # the helper plans
        say("helper done"),
        say("all done"),
    )
    agent = Agent(provider=provider, stream=False)
    subagents.enable(agent)
    planner = planning.enable(agent)
    agent.ask("go")
    assert planner.steps == [{"step": "Delegate", "status": "in_progress"}]  # Untouched by the helper.
    assert "[>] Helper step" in provider.systems[3]  # The helper saw its own plan...
    assert "[>] Delegate" not in provider.systems[3]  # ...not the parent's.


def test_eval_records_and_checks_the_plan():
    task = Task("t/plan", "go", {"plan_made": True, "plan_completed": True, "answer_contains": "Done"})

    def make_agent():
        agent = Agent(provider=planned_run(), tools=default_tools.copy(), stream=False)
        planning.enable(agent)
        return agent

    result = run_task(task, make_agent, log=lambda line: None)
    assert result["plan"][0] == {"step": "Read the file", "status": "done"}
    assert result["checks"]["plan_made"]["passed"]
    assert result["checks"]["plan_completed"] == {"passed": False, "detail": "steps not done: ['Answer']"}


def test_switching_it_off_and_the_planning_tasks():
    assert resolve({}).planning is True and resolve({"planning": False}).planning is False
    tasks = load_tasks([Path(__file__).parent.parent / "evals" / "planning.toml"])
    assert [t.id for t in tasks] == ["planning/orders-summary", "planning/temperature-report"]
