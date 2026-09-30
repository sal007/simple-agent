"""Tests for the eval runner, with scripted providers standing in for a model."""

import json
import os
from pathlib import Path

import pytest

from simple_agent import evals
from simple_agent.agent import Agent
from simple_agent.evals import Run, Task, load_tasks, run_task
from simple_agent.providers.base import Message, Reply, ToolCall
from simple_agent.tools import default_tools

REPO_EVALS = Path(__file__).parent.parent / "evals"


class Scripted:
    """Plays back replies in order; each can be a Reply or an exception to raise."""

    name, model = "fake", "fake-model"

    def __init__(self, *replies):
        self.replies = list(replies)

    def chat(self, system, messages, tools, on_text=None):
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def say(text="", calls=()):
    return Reply(Message(role="assistant", content=text, tool_calls=list(calls)), usage={"input_tokens": 10, "output_tokens": 2})


def call(name, **arguments):
    return ToolCall(f"c-{name}", name, arguments)


def agent_for(provider):
    return lambda: Agent(provider=provider, tools=default_tools.copy(), stream=False)


def quiet(line):
    pass


def test_a_passing_task_with_a_tool_call():
    task = Task("t/sum", "Add up numbers.txt", {"answer_number": 205, "tool_used": "read_file", "max_steps": 2},
                files={"numbers.txt": "200\n5\n"})
    provider = Scripted(say(calls=[call("read_file", path="numbers.txt")]), say("The total is 205."))
    home = os.getcwd()
    result = run_task(task, agent_for(provider), log=quiet)

    assert result["passed"] and result["error"] is None
    assert result["steps"] == 2 and result["tools_called"] == ["read_file"]
    assert result["input_tokens"] == 20 and result["output_tokens"] == 4
    assert result["messages"][2] == {"role": "tool", "content": "200\n5\n", "tool_call_id": "c-read_file"}
    assert os.getcwd() == home  # Back where we started.


def test_write_file_is_allowed_but_shell_is_declined():
    task = Task("t/write", "Write hello.txt", {"file_contains": {"hello.txt": "hi"}})
    provider = Scripted(
        say(calls=[call("run_shell", command="echo hi > hello.txt")]),
        say(calls=[call("write_file", path="hello.txt", content="hi there")]),
        say("Done."),
    )
    result = run_task(task, agent_for(provider), log=quiet)
    assert result["passed"]
    assert "declined" in result["messages"][2]["content"]


def test_failures_say_why():
    lines = []
    task = Task("t/fail", "What is 2+2?", {"answer_number": 4, "answer_contains": "four", "max_steps": 1})
    result = run_task(task, agent_for(Scripted(say("It is 5."))), log=lines.append)
    assert not result["passed"]
    assert result["checks"]["answer_number"] == {"passed": False, "detail": "4 not found in the answer"}
    assert result["checks"]["max_steps"]["passed"]
    assert "FAIL  t/fail" in lines[0] and "answer is missing ['four']" in lines[0]


def test_an_api_error_fails_only_that_task():
    tasks = [Task("t/a", "a", {"answer_contains": "x"}), Task("t/b", "b", {"answer_contains": "yes"})]
    provider = Scripted(RuntimeError("server down"), say("yes"))
    run = evals.run_tasks(tasks, agent_for(provider), log=quiet)
    assert run["results"][0]["error"] == "RuntimeError: server down"
    assert run["results"][1]["passed"]
    assert run["summary"]["passed"] == 1 and run["summary"]["score"] == 0.5


def test_answer_number_handles_formatting():
    run = lambda answer: Run(answer, Path("."), [], 1)  # noqa: E731
    assert evals.check_answer_number(run("That's 7,006,652."), 7006652)[0]
    assert evals.check_answer_number(run("$64.80 in total"), 64.8)[0]
    assert evals.check_answer_number(run("-3 degrees"), -3)[0]
    assert not evals.check_answer_number(run("about 64"), 64.8)[0]


def test_task_files_are_checked(tmp_path):
    (tmp_path / "bad.toml").write_text('[[task]]\nid = "x"\nprompt = "p"\ncheck.answer_is = "y"\n')
    with pytest.raises(ValueError, match="unknown check"):
        load_tasks([tmp_path])


def test_the_starter_tasks_load():
    tasks = load_tasks([REPO_EVALS])
    groups = {t.id.split("/")[0] for t in tasks}
    assert {"math", "files", "multi_step", "many_files"} <= groups and len(tasks) >= 10


def test_eval_command_saves_and_compares(tmp_path, monkeypatch, capsys):
    (tmp_path / "t.toml").write_text('[[task]]\nid = "hi"\nprompt = "Say yes"\ncheck.answer_contains = "yes"\n')
    monkeypatch.setattr(evals, "create_provider", lambda *args: Scripted(say("yes"), say("no")))
    out = tmp_path / "results"

    assert evals.main([str(tmp_path / "t.toml"), "--repeat", "2", "--out", str(out), "--no-plugins"]) == 0
    (saved,) = out.glob("*.json")
    data = json.loads(saved.read_text())
    assert data["summary"]["tasks"] == 2 and data["summary"]["passed"] == 1
    assert data["simple_agent_version"] and "calculator" in data["tools"]

    capsys.readouterr()
    assert evals.main(["--compare", str(saved), str(saved)]) == 0
    table = capsys.readouterr().out
    assert "t/hi" in table and "1/2" in table and "1/2 passed" in table


def test_cli_routes_eval(monkeypatch):
    seen = []
    monkeypatch.setattr(evals, "main", lambda argv: seen.append(argv) or 0)
    from simple_agent.cli import main

    assert main(["eval", "--only", "math/"]) == 0
    assert seen == [["--only", "math/"]]
