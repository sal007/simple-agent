"""Tests for context management: clearing old tool results and compaction."""

import io
import json

from simple_agent.agent import Agent
from simple_agent.config import resolve
from simple_agent.context import CLEARED_PREFIX, SUMMARY_HEADER, SUMMARY_PROMPT, ContextManager
from simple_agent.providers.base import Message, Reply, ToolCall
from simple_agent.tools import ToolRegistry
from simple_agent.trace import Tracer

tools = ToolRegistry()


@tools.tool("Return a lot of text.")
def dump() -> str:
    return "x" * 2000  # About 500 tokens.


class ScriptedProvider:
    name, model = "fake", "fake-model"

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def chat(self, system, messages, tools, on_text=None):
        self.calls.append(list(messages))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def say(text="", tool_calls=()):
    return Reply(Message(role="assistant", content=text, tool_calls=list(tool_calls)), usage={"input_tokens": 1})


def dump_call(n):
    return say(tool_calls=[ToolCall(f"c{n}", "dump", {})])


def make_agent(replies, **settings):
    events = []
    agent = Agent(provider=ScriptedProvider(replies), tools=tools, stream=False, context=ContextManager(**settings))
    agent.events.on_context = lambda kind, details: events.append((kind, details))
    return agent, events


def test_small_history_is_left_alone():
    agent, events = make_agent([dump_call(1), say("done")])
    agent.ask("go")
    assert agent.history[2].content == "x" * 2000
    assert events == []


def test_clears_all_but_the_newest_tool_results():
    # Four tool calls in one turn; each result is ~500 tokens, the limit is 1000.
    replies = [dump_call(i) for i in range(4)] + [say("done")]
    agent, events = make_agent(replies, max_tokens=1000, keep_tool_results=1, compact=False)
    agent.ask("go")

    results = [m.content for m in agent.history if m.role == "tool"]
    assert results[-1] == "x" * 2000  # The newest is kept...
    assert all(r.startswith(CLEARED_PREFIX) for r in results[:-1])  # ...older ones cleared.
    assert "2000 characters" in results[0]
    # Every model call stayed near the limit: the one after the 4th result saw 3 cleared.
    last_call = agent.provider.calls[-1]
    assert sum(m.content.startswith(CLEARED_PREFIX) for m in last_call) == 3
    assert events[0][0] == "cleared" and events[0][1]["tokens_after"] < events[0][1]["tokens_before"]


def test_clearing_can_be_turned_off():
    replies = [dump_call(i) for i in range(3)] + [say("done")]
    agent, events = make_agent(replies, max_tokens=100, clear_tool_results=False, compact=False)
    agent.ask("go")
    assert all(m.content == "x" * 2000 for m in agent.history if m.role == "tool")
    assert events == []


def test_compacts_older_turns_into_a_summary():
    replies = [say("a" * 1600), say("b" * 1600), say("c" * 1600),
               say("<summary>We talked about a, b and c.</summary>"), say("fine")]
    agent, events = make_agent(replies, max_tokens=900, keep_recent_turns=1)
    for question in ("one", "two", "three"):
        agent.ask(question)
    agent.ask("four")  # Starts over the limit, so it compacts first.

    summary_request = agent.provider.calls[3]
    assert [m.content for m in summary_request[:-1]] == ["one", "a" * 1600, "two", "b" * 1600]
    assert summary_request[-1].content == SUMMARY_PROMPT
    # Older turns became a summary; the last turn and the new one are word for word.
    assert [m.role for m in agent.history] == ["user", "assistant"] * 3
    assert agent.history[0].content.startswith(SUMMARY_HEADER)
    assert "We talked about a, b and c." in agent.history[0].content
    assert [m.content for m in agent.history[2:]] == ["three", "c" * 1600, "four", "fine"]
    assert [kind for kind, _ in events] == ["compacting", "compacted"]
    assert agent.usage["input_tokens"] == 5  # The summary call counts too.


def test_failed_summary_keeps_the_history_and_the_turn():
    replies = [say("a" * 4000), say("b" * 4000), RuntimeError("server down"), say("still here")]
    agent, events = make_agent(replies, max_tokens=900, keep_recent_turns=1)
    agent.ask("one")
    agent.ask("two")
    assert agent.ask("three") == "still here"
    assert len(agent.history) == 6
    assert events[-1][0] == "compact_failed" and "server down" in events[-1][1]["error"]


def test_nothing_to_compact_with_only_recent_turns():
    agent, events = make_agent([say("a" * 4000), say("ok")], max_tokens=100, keep_recent_turns=2)
    agent.ask("one")
    agent.ask("two")  # Over the limit, but the only older turn is one of the two to keep.
    assert [kind for kind, _ in events] == []


def test_config_section():
    ctx = resolve({"context": {"max_tokens": 3000, "compact": False}}).context
    assert ctx.max_tokens == 3000 and ctx.compact is False and ctx.clear_tool_results is True
    assert resolve({"context": {"enabled": False}}).context is None
    assert resolve({}).context == ContextManager()


def test_trace_shows_the_whole_history_after_it_changes(tmp_path):
    replies = [say("a" * 4000), say("b" * 4000), say("<summary>s</summary>"), say("fine")]
    out = io.StringIO()
    tracer = Tracer(tmp_path, out=out)
    agent = Agent(provider=ScriptedProvider(replies), tools=tools, stream=False,
                  context=ContextManager(max_tokens=900, keep_recent_turns=1), events=tracer.events())
    tracer.start(agent.provider, agent.system_prompt, agent.tools.specs())
    for question in ("one", "two", "three"):
        agent.ask(question)

    text = out.getvalue()
    assert "context: replaced 2 older messages with a summary" in text
    # After compaction the next request lists all 5 messages again, not just the new one.
    assert "sending 5 messages to the model (5 new)" in text
    records = [json.loads(line) for line in tracer.log_path.read_text().splitlines()]
    compacted = next(r for r in records if r["event"] == "context" and r["kind"] == "compacted")
    assert compacted["summary"] == "s"
