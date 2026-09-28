"""Provider for any server that speaks the OpenAI Chat Completions API.

That covers LM Studio (http://localhost:1234/v1), Ollama, llama.cpp's server,
vLLM, OpenRouter and OpenAI itself: only base_url, api_key and model change.
"""

from __future__ import annotations

import json

from openai import OpenAI

from .base import Message, Reply, TextCallback, ToolCall, ToolSpec


class OpenAICompatibleProvider:
    name = "openai"

    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None):
        self.model = model
        # Local servers don't check the key, but the SDK insists on having one.
        self.client = OpenAI(base_url=base_url, api_key=api_key or "not-needed")

    def chat(
        self, system: str, messages: list[Message], tools: list[ToolSpec], on_text: TextCallback | None = None
    ) -> Reply:
        request = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}] + [_to_openai(m) for m in messages],
        }
        if tools:
            request["tools"] = [
                {
                    "type": "function",
                    "function": {"name": t.name, "description": t.description, "parameters": t.parameters},
                }
                for t in tools
            ]
        if on_text:
            return self._chat_streaming(request, on_text)

        response = self.client.chat.completions.create(**request)
        choice = response.choices[0]
        msg = choice.message

        tool_calls = [
            ToolCall(id=tc.id, name=tc.function.name, arguments=_parse_arguments(tc.function.arguments))
            for tc in (msg.tool_calls or [])
        ]
        return Reply(
            message=Message(role="assistant", content=msg.content or "", tool_calls=tool_calls),
            stop_reason=choice.finish_reason or "",
            usage=_usage(response.usage),
        )

    def _chat_streaming(self, request: dict, on_text: TextCallback) -> Reply:
        """Same request, but the reply arrives in small chunks as it is generated.

        Text chunks are passed to on_text right away. Tool calls also arrive in
        pieces (the name first, then the JSON arguments a few characters at a
        time), so we glue them back together by their index.
        """
        stream = self.client.chat.completions.create(
            **request, stream=True, stream_options={"include_usage": True}  # Ask for token counts at the end.
        )
        text: list[str] = []
        calls: dict[int, dict[str, str]] = {}
        finish_reason, usage = "", {}

        for chunk in stream:
            if chunk.usage:
                usage = _usage(chunk.usage)
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.delta
            if delta.content:
                text.append(delta.content)
                on_text(delta.content)
            for tc in delta.tool_calls or []:
                call = calls.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
                call["id"] = tc.id or call["id"]
                if tc.function:
                    call["name"] += tc.function.name or ""
                    call["arguments"] += tc.function.arguments or ""
            finish_reason = choice.finish_reason or finish_reason

        tool_calls = [
            ToolCall(id=c["id"], name=c["name"], arguments=_parse_arguments(c["arguments"]))
            for _, c in sorted(calls.items())
        ]
        return Reply(
            message=Message(role="assistant", content="".join(text), tool_calls=tool_calls),
            stop_reason=finish_reason,
            usage=usage,
        )


def _to_openai(m: Message) -> dict:
    """Translate one neutral Message into the OpenAI wire format."""
    if m.role == "tool":
        return {"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content}
    if m.role == "assistant" and m.tool_calls:
        return {
            "role": "assistant",
            "content": m.content or None,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                }
                for tc in m.tool_calls
            ],
        }
    return {"role": m.role, "content": m.content}


def _parse_arguments(raw: str | None) -> dict:
    """Tool arguments arrive as a JSON string; small local models sometimes get it wrong."""
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        # Hand the broken text to the tool layer, which reports it back to the model.
        return {"__invalid_json__": raw}
    return value if isinstance(value, dict) else {"__invalid_json__": raw}


def _usage(usage) -> dict[str, int]:
    if not usage:
        return {}
    return {"input_tokens": usage.prompt_tokens, "output_tokens": usage.completion_tokens}
