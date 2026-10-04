"""Tests for long-term memory (memory.py): notes in a folder, the index in the system prompt."""

import os

from simple_agent import memory
from simple_agent.agent import Agent
from simple_agent.cli import _command, main
from simple_agent.config import resolve
from simple_agent.memory import Memory, parse
from simple_agent.providers.base import Message, Reply, ToolCall
from simple_agent.subagents import enable as enable_subagents


class Recording:
    """Answers from a script and keeps every system prompt and tool list it was sent."""

    name, model = "fake", "fake-model"

    def __init__(self, *replies):
        self.replies = list(replies)
        self.systems, self.tools = [], []

    def chat(self, system, messages, tools, on_text=None):
        self.systems.append(system)
        self.tools.append([t.name for t in tools])
        return self.replies.pop(0)


def say(text="", calls=()):
    return Reply(Message(role="assistant", content=text, tool_calls=list(calls)), usage={})


def test_save_read_update_delete(tmp_path):
    changes = []
    notes = Memory(tmp_path / "memory", on_change=lambda kind, title, path: changes.append((kind, title, path.name)))
    assert notes.notes() == []  # The folder doesn't exist until the first save.

    assert notes.save("Preferred units", "Metric  units\nplease", "Use km and kg.") == "Saved the note 'Preferred units'."
    path = tmp_path / "memory" / "preferred-units.md"
    note = parse(path.read_text(), "x")
    assert (note.title, note.description, note.text) == ("Preferred units", "Metric units please", "Use km and kg.")
    assert note.updated  # Today's date.

    assert "Use km and kg." in notes.read("preferred units")  # Titles match whatever the case.
    assert notes.save("Preferred units", "Metric", "Use km, kg and °C.").startswith("Updated")
    assert "°C" in notes.read("Preferred units") and len(list(path.parent.iterdir())) == 1

    assert "no note titled 'Nope'" in notes.read("Nope") and "'Preferred units'" in notes.read("Nope")
    assert notes.delete("Preferred units") == "Deleted the note 'Preferred units'."
    assert not path.exists() and "Error" in notes.delete("Preferred units")
    assert changes == [("saved", "Preferred units", "preferred-units.md"),
                       ("updated", "Preferred units", "preferred-units.md"),
                       ("deleted", "Preferred units", "preferred-units.md")]


def test_bad_notes_are_refused(tmp_path):
    notes = Memory(tmp_path)
    assert notes.save("", "d", "text").startswith("Error")
    assert notes.save("Title", "d", "  ").startswith("Error")
    assert "limited to" in notes.save("Title", "d", "x" * (memory.MAX_NOTE_CHARS + 1))
    assert notes.notes() == []


def test_hand_written_files_work_too(tmp_path):
    (tmp_path / "server.md").write_text("The build server is ci.example.com.\n")
    (tmp_path / "x.md").write_text("---\ntitle: Release day\ndescription: When we ship\n---\nFridays.\n")
    notes = Memory(tmp_path)
    assert "ci.example.com" in notes.read("server")
    assert "Fridays." in notes.read("Release day")  # Found by the title inside, not the file name.
    assert notes.save("Release day", "When we ship", "Thursdays.").startswith("Updated")
    assert "Thursdays." in (tmp_path / "x.md").read_text()


def test_the_index_is_in_the_system_prompt_at_every_step(tmp_path):
    folder = tmp_path / "memory"
    provider = Recording(
        say(calls=[ToolCall("s1", "save_memory", {"title": "Pet", "description": "The user's cat", "text": "Her name is Mia."})]),
        say("Noted."),
        say(calls=[ToolCall("r1", "read_memory", {"title": "Pet"})]),
        say("Mia."),
    )
    agent = Agent(provider=provider, system_prompt="Base prompt.", stream=False)
    memory.enable(agent, Memory(folder))

    agent.ask("My cat is called Mia, remember that.")
    assert "no saved notes yet" in provider.systems[0]
    assert "- Pet: The user's cat" in provider.systems[1]
    assert "Her name is Mia" not in provider.systems[1]  # Only titles and descriptions, not the text.
    assert {"save_memory", "read_memory", "delete_memory"} <= set(provider.tools[0])

    agent.reset()  # A new conversation keeps the notes.
    assert agent.ask("What's my cat called?") == "Mia."
    assert "- Pet: The user's cat" in provider.systems[2]
    assert "Her name is Mia." in agent.history[-2].content  # The read_memory result.


def test_the_index_has_a_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "MAX_INDEX_NOTES", 2)
    notes = Memory(tmp_path)
    for i in range(4):
        notes.save(f"Note {i}", f"number {i}", "text")
        os.utime(tmp_path / f"note-{i}.md", (i, i))  # Note 3 is the newest.
    index = notes.system_note()
    assert "- Note 3: number 3\n- Note 2: number 2\n- ... and 2 older notes" in index


def test_deleting_asks_first_and_saving_does_not(tmp_path):
    agent = Agent(provider=Recording(), stream=False)
    memory.enable(agent, Memory(tmp_path))
    assert agent.tools.asks_first("delete_memory")
    assert not agent.tools.asks_first("save_memory") and not agent.tools.asks_first("read_memory")
    Memory(tmp_path).save("Keep", "d", "text")
    agent.tools.approve = lambda name, arguments: False
    assert "declined" in agent.tools.run("delete_memory", {"title": "Keep"})
    assert (tmp_path / "keep.md").exists()


def test_sub_agents_can_read_but_not_change_notes(tmp_path):
    Memory(tmp_path).save("Pet", "The user's cat", "Mia.")
    provider = Recording(say(calls=[ToolCall("d1", "delegate", {"task": "find the cat's name"})]), say("Mia."), say("Mia."))
    agent = Agent(provider=provider, stream=False)
    memory.enable(agent, Memory(tmp_path))
    enable_subagents(agent)
    agent.ask("go")

    parent, helper = provider.tools[0], provider.tools[1]
    assert {"save_memory", "delete_memory", "read_memory"} <= set(parent)
    assert "read_memory" in helper and "save_memory" not in helper and "delete_memory" not in helper
    assert "- Pet: The user's cat" in provider.systems[1]


def test_cli_flag_setting_and_command(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    Memory("memory").save("Pet", "The user's cat", "Mia.")
    lines = iter(["/tools", "/memory", "/memory forget Pet", "/memory", "/exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(lines))

    assert main(["--no-plugins", "--no-usage"]) == 0
    out = capsys.readouterr().out
    assert "save_memory:" in out and "delete_memory (asks first):" in out
    assert "Pet: The user's cat (" in out
    assert "memory> deleted 'Pet' (memory/pet.md)" in out and "no notes yet in memory/" in out

    lines = iter(["/tools", "/memory", "/exit"])
    assert main(["--no-plugins", "--no-memory"]) == 0
    out = capsys.readouterr().out
    assert "save_memory" not in out and "long-term memory is off" in out

    (tmp_path / "config.toml").write_text('memory = false\n')
    lines = iter(["/tools", "/exit"])
    assert main(["--no-plugins"]) == 0
    assert "save_memory" not in capsys.readouterr().out
    assert resolve({"memory": False}).memory is False
    assert resolve({"memory_dir": "~/notes"}).memory_dir == "~/notes"
    assert resolve({}).memory is True and resolve({}).memory_dir == "memory"


def test_memory_command_without_memory(capsys):
    agent = Agent(provider=Recording(), stream=False)
    _command(agent, "/memory")
    assert "long-term memory is off" in capsys.readouterr().out
