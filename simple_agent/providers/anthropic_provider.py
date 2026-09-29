"""Provider for Claude, using the official Anthropic SDK (Messages API)."""

from __future__ import annotations

import anthropic

from .base import Message, Reply, TextCallback, ToolCall, ToolSpec


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, model: str = "claude-opus-5-5", api_key: str | None = None, max_tokens: int = 16000):
        self.model = model
        self.max_tokens = max_tokens
        # With api_key=None the SDK reads ANTHROPIC_API_KEY from the environment.
        self.client = anthropic.Anthropic(api_key=api_key)

    def chat(
        self, system: str, messages: list[Message], tools: list[ToolSpec], on_text: TextCallback | None = None
    ) -> Reply:
        request = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=_to_anthropic(messages),
            tools=[{"name": t.name, "description": t.description, "input_schema": t.parameters} for t in tools],
            # If a safety classifier declines the request, let the API retry it
            # on a suitable fallback model instead of returning a refusal.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if on_text:
            # The SDK's stream helper hands us text as it arrives and still
            # assembles the complete message for us at the end.
            with self.client.beta.messages.stream(**request) as stream:
                for text in stream.text_stream:
                    on_text(text)
                response = stream.get_final_message()
        else:
            response = self.client.beta.messages.create(**request)

        text = "".join(block.text for block in response.content if block.type == "text")
        tool_calls = [
            ToolCall(id=block.id, name=block.name, arguments=block.input)
            for block in response.content
            if block.type == "tool_use"
        ]
        if response.stop_reason == "refusal" and not text:
            text = "(The model declined to answer this request.)"

        return Reply(
            # Keep the full content (including thinking blocks) in `raw` so the
            # next request can send this turn back exactly as Claude wrote it.
            message=Message(role="assistant", content=text, tool_calls=tool_calls, raw=response.content),
            stop_reason=response.stop_reason or "",
            usage={"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens},
        )


def _to_anthropic(messages: list[Message]) -> list[dict]:
    """Translate neutral Messages into Anthropic's format.

    The main difference from OpenAI: tool results are not their own role.
    They go back as "tool_result" blocks inside a user message, and all
    results for one assistant turn belong in the same user message.
    """
    out: list[dict] = []
    for m in messages:
        if m.role == "tool":
            block = {"type": "tool_result", "tool_use_id": m.tool_call_id, "content": m.content}
            if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                out[-1]["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
        elif m.role == "assistant":
            if m.raw is not None:
                out.append({"role": "assistant", "content": m.raw})
            else:
                # A turn we didn't get from Claude (e.g. history built by hand).
                blocks: list[dict] = []
                if m.content:
                    blocks.append({"type": "text", "text": m.content})
                for tc in m.tool_calls:
                    blocks.append({"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments})
                out.append({"role": "assistant", "content": blocks})
        else:
            out.append({"role": "user", "content": m.content})
    return out
