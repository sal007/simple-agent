"""Tests for project instructions (AGENTS.md files, instructions.py)."""

import os

from simple_agent import instructions
from simple_agent.agent import Agent
from simple_agent.cli import _command, main
from simple_agent.evals import Task, load_tasks, run_task
from simple_agent.instructions import ProjectInstructions, find_files
from simple_agent.providers.base import Message, Reply, ToolCall
from simple_agent.subagents import enable as enable_subagents


class Recording:
    """Answers from a script and keeps every system prompt it was sent."""

    name, model = "fake", "fake-model"

    def __init__(self, *replies):
        self.replies = list(replies)
        self.systems = []

    def chat(self, system, messages, tools, on_text=None):
        self.systems.append(system)
        return self.replies.pop(0)


def say(text="", calls=()):
    return Reply(Message(role="assistant", content=text, tool_calls=list(calls)), usage={})


def test_files_from_the_repo_top_down_to_here(tmp_path):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    (repo / "app" / "src").mkdir(parents=True)
    (tmp_path / "AGENTS.md").write_text("outside the repo")
    (repo / "AGENTS.md").write_text("repo rules")
    (repo / "app" / "src" / "AGENTS.md").write_text("src rules")
    user = tmp_path / "me.md"
    user.write_text("my rules")

    found = find_files(repo / "app" / "src", user_file=user)
    assert found == [user, repo / "AGENTS.md", repo / "app" / "src" / "AGENTS.md"]
    assert find_files(repo / "app", user_file=None) == [repo / "AGENTS.md"]


def test_outside_a_repo_only_the_start_folder(tmp_path):
    (tmp_path / "AGENTS.md").write_text("parent")
    (tmp_path / "here").mkdir()
    assert find_files(tmp_path / "here", user_file=None) == []
    (tmp_path / "here" / "AGENTS.md").write_text("here")
    assert find_files(tmp_path / "here", user_file=None) == [tmp_path / "here" / "AGENTS.md"]


def test_the_model_sees_them_and_edits_count_from_the_next_call(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "AGENTS.md").write_text("Use tabs, not spaces.")
    provider = Recording(say("ok"), say("ok"))
    agent = Agent(provider=provider, system_prompt="Base prompt.", stream=False)
    instructions.enable(agent, ProjectInstructions(user_file=None))

    agent.ask("hi")
    assert provider.systems[0].startswith("Base prompt.\n\nProject instructions from AGENTS.md.")
    assert provider.systems[0].endswith("Use tabs, not spaces.")

    (tmp_path / "AGENTS.md").write_text("Use spaces after all.")
    os.utime(tmp_path / "AGENTS.md", (1, 1))  # A different modified time, however fast the test runs.
    agent.ask("again")
    assert provider.systems[1].endswith("Use spaces after all.")


def test_long_files_are_cut_and_instructions_come_before_the_plan(tmp_path, monkeypatch):
    from simple_agent import planning

    monkeypatch.chdir(tmp_path)
    (tmp_path / "AGENTS.md").write_text("x" * (instructions.MAX_CHARS + 50))
    agent = Agent(provider=Recording(), stream=False)
    planning.enable(agent).steps = [{"step": "do it", "status": "pending"}]
    instructions.enable(agent, ProjectInstructions(user_file=None))
    system = agent.system()
    assert "[... cut off: AGENTS.md is longer than" in system
    assert system.index("Project instructions") < system.index("Your current plan")


def test_sub_agents_follow_them_too(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "AGENTS.md").write_text("Answer in French.")
    provider = Recording(say(calls=[ToolCall("d1", "delegate", {"task": "say hi"})]), say("bonjour"), say("done"))
    agent = Agent(provider=provider, stream=False)
    enable_subagents(agent)
    instructions.enable(agent, ProjectInstructions(user_file=None))
    agent.ask("go")
    assert all(system.endswith("Answer in French.") for system in provider.systems)
    assert len(provider.systems) == 3


def test_cli_start_up_line_flag_and_command(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(instructions, "USER_FILE", tmp_path / "nobody.md")
    (tmp_path / "AGENTS.md").write_text("Be brief.")
    lines = iter(["/instructions", "/exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(lines))

    assert main(["--no-plugins", "--no-usage"]) == 0
    out = capsys.readouterr().out
    assert "(project instructions: AGENTS.md (9 characters))" in out
    assert "--- " in out and "Be brief." in out

    lines = iter(["/instructions", "/exit"])
    assert main(["--no-plugins", "--no-instructions"]) == 0
    out = capsys.readouterr().out
    assert "project instructions:" not in out and "project instructions are off" in out

    (tmp_path / "config.toml").write_text("project_instructions = false\n")
    lines = iter(["/exit"])
    assert main(["--no-plugins"]) == 0
    assert "project instructions:" not in capsys.readouterr().out


def test_no_file_says_how_to_add_one(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    agent = Agent(provider=Recording(), stream=False)
    instructions.enable(agent, ProjectInstructions(user_file=None))
    _command(agent, "/instructions")
    assert "no AGENTS.md here" in capsys.readouterr().out


def test_eval_tasks_read_the_agents_md_in_their_folder():
    task = Task("t/rules", "What's the codename?", {"answer_contains": "heron"}, files={"AGENTS.md": "Codename: HERON"})
    provider = Recording(say("HERON"))

    def make_agent():
        agent = Agent(provider=provider, stream=False)
        instructions.enable(agent, ProjectInstructions(user_file=None))
        return agent

    assert run_task(task, make_agent, log=lambda line: None)["passed"]
    assert provider.systems[0].endswith("Codename: HERON")


def test_the_instruction_eval_tasks_load():
    tasks = load_tasks(["evals/instructions.toml"])
    assert len(tasks) == 3 and all("AGENTS.md" in t.files for t in tasks)
