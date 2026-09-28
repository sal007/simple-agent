import json

import pytest
from anthropic.types.beta import BetaTextBlock, BetaThinkingBlock, BetaToolUseBlock

from simple_agent import sessions
from simple_agent.agent import Agent
from simple_agent.cli import main
from simple_agent.providers.anthropic_provider import _to_anthropic
from simple_agent.providers.base import Message, Reply, ToolCall


class ScriptedProvider:
    name, model = "fake", "fake-model"

    def __init__(self, replies=()):
        self.replies = list(replies)
        self.seen = []

    def chat(self, system, messages, tools, on_text=None):
        self.seen.append(list(messages))
        return self.replies.pop(0)


def sample_history():
    raw = [
        BetaThinkingBlock(type="thinking", thinking="", signature="sig"),
        BetaToolUseBlock(type="tool_use", id="toolu_1", name="calculator", input={"expression": "6*7"}),
    ]
    return [
        Message(role="user", content="what is 6*7?"),
        Message(role="assistant", tool_calls=[ToolCall("toolu_1", "calculator", {"expression": "6*7"})], raw=raw),
        Message(role="tool", content="42", tool_call_id="toolu_1"),
        Message(role="assistant", content="42.", raw=[BetaTextBlock(type="text", text="42.")]),
    ]


def test_save_and_load_round_trip(tmp_path):
    agent = Agent(provider=ScriptedProvider())
    agent.history.extend(sample_history())
    agent.usage = {"input_tokens": 7, "output_tokens": 3}
    path = sessions.save(agent, sessions.session_path(tmp_path, "math"))
    assert path == tmp_path / "math.json"

    resumed = Agent(provider=ScriptedProvider())
    data = sessions.load(resumed, path)
    assert data["provider"] == "fake" and resumed.usage == agent.usage
    assert [(m.role, m.content, m.tool_call_id) for m in resumed.history] == [
        (m.role, m.content, m.tool_call_id) for m in agent.history
    ]
    assert resumed.history[1].tool_calls == agent.history[1].tool_calls
    # Claude's raw reply (thinking block included) survives, so it can be replayed exactly.
    assert _to_anthropic(resumed.history)[1]["content"] == [
        {"type": "thinking", "thinking": "", "signature": "sig"},
        {"type": "tool_use", "id": "toolu_1", "name": "calculator", "input": {"expression": "6*7"}},
    ]


def test_resumed_conversation_is_sent_to_the_model(tmp_path):
    first = Agent(provider=ScriptedProvider())
    first.history.extend(sample_history())
    path = sessions.save(first, tmp_path / "s.json")

    provider = ScriptedProvider([Reply(Message(role="assistant", content="84"))])
    agent = Agent(provider=provider)
    sessions.load(agent, path)
    assert agent.ask("and doubled?") == "84"
    assert len(provider.seen[0]) == 5


def test_session_names(tmp_path):
    assert sessions.session_path(tmp_path, "a-b_1.2") == tmp_path / "a-b_1.2.json"
    assert str(sessions.session_path(tmp_path, "other/place.json")) == "other/place.json"
    with pytest.raises(ValueError):
        sessions.session_path(tmp_path, "bad name!")


def test_list_sessions(tmp_path):
    agent = Agent(provider=ScriptedProvider())
    agent.history.extend(sample_history())
    sessions.save(agent, tmp_path / "one.json")
    (tmp_path / "junk.json").write_text("not json")
    [(name, info)] = sessions.list_sessions(tmp_path)
    assert name == "one" and info["messages"] == 4 and info["first_message"] == "what is 6*7?"


def test_load_rejects_other_files(tmp_path):
    path = tmp_path / "x.json"
    path.write_text(json.dumps({"hello": "world"}))
    with pytest.raises(ValueError):
        sessions.load(Agent(provider=ScriptedProvider()), path)


def test_cli_save_list_and_autosave(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.toml").write_text('sessions_dir = "saved"\n')
    lines = iter(["/save first", "/sessions", "/load first", "/exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(lines))
    first = Agent(provider=ScriptedProvider())
    first.history.extend(sample_history())
    sessions.save(first, tmp_path / "saved" / "seed.json")

    assert main(["--resume", "seed"]) == 0
    out = capsys.readouterr().out
    assert "(resumed seed: 4 messages" in out
    assert "first: 4 messages" in out
    assert (tmp_path / "saved" / "first.json").exists()
    assert (tmp_path / "saved" / "last.json").exists()
