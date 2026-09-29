"""Run both real providers (real SDKs, real HTTP) against tiny fake servers.

Each fake server first asks for the calculator tool, then answers with text,
so these tests check the full round trip: request format, tool call parsing,
and how tool results are sent back.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from simple_agent.agent import Agent
from simple_agent.providers import create_provider


@pytest.fixture
def fake_server():
    """Start a server whose responses come from `handler(path, body) -> dict`."""
    servers = []

    def start(handler):
        requests = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                requests.append({"path": self.path, "headers": dict(self.headers), "body": body})
                response = handler(body, len(requests))
                if isinstance(response, list):  # A streamed reply: Server-Sent Events.
                    payload = "".join(response).encode()
                    content_type = "text/event-stream"
                else:
                    payload = json.dumps(response).encode()
                    content_type = "application/json"
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return f"http://127.0.0.1:{server.server_port}", requests

    yield start
    for server in servers:
        server.shutdown()


def openai_reply(body, n):
    usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    if n == 1:
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "calculator", "arguments": '{"expression": "6*7"}'}}
            ],
        }
        finish = "tool_calls"
    else:
        message = {"role": "assistant", "content": "The answer is 42."}
        finish = "stop"
    return {
        "id": f"chatcmpl-{n}",
        "object": "chat.completion",
        "created": 0,
        "model": body["model"],
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": usage,
    }


def openai_stream(body, n):
    """The same two replies as openai_reply, sent the way a streaming server sends them."""
    assert body["stream"] is True

    def chunk(delta, finish=None, usage=None):
        choices = [{"index": 0, "delta": delta, "finish_reason": finish}] if delta is not None else []
        data = {"id": f"c{n}", "object": "chat.completion.chunk", "created": 0, "model": "m", "choices": choices}
        if usage:
            data["usage"] = usage
        return f"data: {json.dumps(data)}\n\n"

    usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    if n == 1:
        # The tool call arrives in pieces: id and name first, then the arguments.
        events = [
            chunk({"role": "assistant", "tool_calls": [
                {"index": 0, "id": "call_1", "type": "function", "function": {"name": "calculator", "arguments": ""}}]}),
            chunk({"tool_calls": [{"index": 0, "function": {"arguments": '{"expression": '}}]}),
            chunk({"tool_calls": [{"index": 0, "function": {"arguments": '"6*7"}'}}]}),
            chunk({}, finish="tool_calls"),
        ]
    else:
        events = [chunk({"role": "assistant", "content": "The answer "}), chunk({"content": "is 42."}), chunk({}, finish="stop")]
    return events + [chunk(None, usage=usage), "data: [DONE]\n\n"]


@pytest.mark.parametrize("stream", [False, True])
def test_openai_compatible_round_trip(fake_server, stream):
    url, requests = fake_server(openai_stream if stream else openai_reply)
    streamed = []
    agent = Agent(provider=create_provider("openai", "local-model", base_url=url + "/v1"), stream=stream)
    agent.events.on_text = streamed.append

    assert agent.ask("what is 6*7?") == "The answer is 42."

    first, second = (r["body"] for r in requests)
    assert requests[0]["path"] == "/v1/chat/completions"
    assert first["messages"][0]["role"] == "system"
    assert first["tools"][0]["type"] == "function"
    # The tool result goes back as a "tool" message tied to the call id.
    assert second["messages"][-2]["tool_calls"][0]["id"] == "call_1"
    assert second["messages"][-1] == {"role": "tool", "tool_call_id": "call_1", "content": "42"}
    assert agent.usage == {"input_tokens": 20, "output_tokens": 10}
    assert streamed == (["The answer ", "is 42."] if stream else [])


def anthropic_reply(body, n):
    base = {"id": f"msg_{n}", "type": "message", "role": "assistant", "model": body["model"],
            "stop_sequence": None, "usage": {"input_tokens": 10, "output_tokens": 5}}
    if n == 1:
        return base | {
            "stop_reason": "tool_use",
            "content": [
                {"type": "thinking", "thinking": "", "signature": "sig-abc"},
                {"type": "tool_use", "id": "toolu_1", "name": "calculator", "input": {"expression": "6*7"}},
            ],
        }
    return base | {"stop_reason": "end_turn", "content": [{"type": "text", "text": "The answer is 42."}]}


def anthropic_stream(body, n):
    """The same two replies as anthropic_reply, as a stream of Server-Sent Events."""
    assert body["stream"] is True
    message = anthropic_reply(body, n)

    def event(name, data):
        return f"event: {name}\ndata: {json.dumps({'type': name, **data})}\n\n"

    events = [event("message_start", {"message": message | {"content": [], "stop_reason": None}})]
    for i, block in enumerate(message["content"]):
        if block["type"] == "text":
            events.append(event("content_block_start", {"index": i, "content_block": {"type": "text", "text": ""}}))
            for piece in ("The answer ", "is 42."):
                events.append(event("content_block_delta", {"index": i, "delta": {"type": "text_delta", "text": piece}}))
        elif block["type"] == "tool_use":
            events.append(event("content_block_start", {"index": i, "content_block": block | {"input": {}}}))
            delta = {"type": "input_json_delta", "partial_json": json.dumps(block["input"])}
            events.append(event("content_block_delta", {"index": i, "delta": delta}))
        else:  # thinking
            events.append(event("content_block_start", {"index": i, "content_block": block}))
        events.append(event("content_block_stop", {"index": i}))
    events.append(event("message_delta", {"delta": {"stop_reason": message["stop_reason"], "stop_sequence": None},
                                          "usage": {"output_tokens": 5}}))
    events.append(event("message_stop", {}))
    return events


@pytest.mark.parametrize("stream", [False, True])
def test_anthropic_round_trip(fake_server, stream):
    url, requests = fake_server(anthropic_stream if stream else anthropic_reply)
    provider = create_provider("anthropic", "claude-opus-5-5", api_key="test-key")
    provider.client = provider.client.with_options(base_url=url)
    streamed = []
    agent = Agent(provider=provider, stream=stream)
    agent.events.on_text = streamed.append

    assert agent.ask("what is 6*7?") == "The answer is 42."

    first, second = (r["body"] for r in requests)
    assert requests[0]["path"] == "/v1/messages?beta=true"
    assert "server-side-fallback-2026-07-01" in requests[0]["headers"]["anthropic-beta"]
    assert first["fallbacks"] == "default"
    # Edited history (context management) drops stale thinking blocks instead of failing.
    assert "thinking-binding-controls-2026-08-01" in requests[0]["headers"]["anthropic-beta"]
    assert first["thinking"] == {"type": "adaptive", "block_binding": {"prefix_mismatch_behavior": "drop_block"}}
    calculator = next(t for t in first["tools"] if t["name"] == "calculator")
    assert calculator["input_schema"]["required"] == ["expression"]
    # The assistant turn is replayed exactly, thinking block included...
    assert second["messages"][1]["content"][0] == {"type": "thinking", "thinking": "", "signature": "sig-abc"}
    # ...and the tool result goes back inside a user message.
    assert second["messages"][2] == {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "42"}],
    }
    assert streamed == (["The answer ", "is 42."] if stream else [])
