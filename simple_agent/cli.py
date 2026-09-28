"""The terminal interface: a read-eval-print loop around the Agent."""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__
from .agent import Agent, AgentEvents
from .config import PROVIDER_DEFAULTS, load_file, resolve
from .providers import create_provider
from .trace import Tracer

HELP = """Commands:
  /help     show this help
  /tools    list the tools the agent can use
  /history  show the conversation so far
  /usage    show tokens used this session
  /reset    start a new conversation
  /exit     quit (Ctrl+D also works)"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="simple-agent", description="A small CLI agent for learning.")
    parser.add_argument("--provider", choices=list(PROVIDER_DEFAULTS), help="Which backend to use.")
    parser.add_argument("--model", help="Model name, e.g. the id LM Studio shows, or claude-opus-5-5.")
    parser.add_argument("--base-url", help="API base URL for the openai provider (LM Studio, Ollama, ...).")
    parser.add_argument("--config", help="Path to a config.toml (default: ./config.toml, then ~/.config/simple-agent/).")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show tool results, not just tool calls.")
    parser.add_argument("--trace", action="store_true", default=None, help="Print every step of the agent loop and save it to a log file.")
    parser.add_argument("--trace-dir", help="Where trace logs go (default: ./traces).")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("prompt", nargs="*", help="Ask one question and exit instead of starting the REPL.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        settings = resolve(load_file(args.config), args.provider, args.model, args.base_url)
        provider = create_provider(settings.provider, settings.model, settings.base_url, settings.api_key)
    except Exception as exc:  # noqa: BLE001
        print(f"Setup error: {exc}", file=sys.stderr)
        return 1

    agent = Agent(
        provider=provider,
        system_prompt=settings.system_prompt,
        max_steps=settings.max_steps,
        events=_printing_events(args.verbose),
    )
    trace = settings.trace if args.trace is None else args.trace
    if trace:
        # The tracer prints tool calls itself, so it replaces the normal printing.
        tracer = Tracer(args.trace_dir or settings.trace_dir)
        tracer.start(provider, agent.system_prompt, agent.tools.specs())
        agent.events = tracer.events()

    if args.prompt:  # One-shot mode: simple-agent "what time is it?"
        return 0 if _ask(agent, " ".join(args.prompt)) else 1

    where = f" at {settings.base_url}" if settings.provider == "openai" else ""
    print(f"simple-agent {__version__} · {settings.provider} · {settings.model}{where}")
    print("Type a message, or /help for commands.\n")

    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue
        if line.startswith("/"):
            if _command(agent, line) == "exit":
                return 0
            continue
        _ask(agent, line)


def _ask(agent: Agent, text: str) -> bool:
    try:
        answer = agent.ask(text)
    except KeyboardInterrupt:
        print("\n(interrupted)")
        return False
    except Exception as exc:  # noqa: BLE001 - show API/network errors without crashing the REPL
        print(f"error> {type(exc).__name__}: {exc}", file=sys.stderr)
        if type(exc).__name__ == "APIConnectionError":  # Same class name in both SDKs.
            print("       Is the server running and is --base-url right? (LM Studio: Developer tab > Start Server)", file=sys.stderr)
        return False
    print(f"agent> {answer}\n")
    return True


def _command(agent: Agent, line: str) -> str | None:
    cmd = line.split()[0].lower()
    if cmd in ("/exit", "/quit"):
        return "exit"
    if cmd == "/help":
        print(HELP)
    elif cmd == "/tools":
        for spec in agent.tools.specs():
            print(f"  {spec.name}: {spec.description}")
    elif cmd == "/history":
        for m in agent.history:
            if m.role == "tool":
                print(f"  [tool result] {_short(m.content)}")
            else:
                calls = "".join(f" [calls {c.name}]" for c in m.tool_calls)
                print(f"  [{m.role}] {_short(m.content)}{calls}")
    elif cmd == "/usage":
        print(f"  input tokens: {agent.usage.get('input_tokens', 0)}, output tokens: {agent.usage.get('output_tokens', 0)}")
    elif cmd == "/reset":
        agent.reset()
        print("  (conversation cleared)")
    else:
        print(f"  Unknown command {cmd}. Type /help.")
    return None


def _printing_events(verbose: bool) -> AgentEvents:
    def on_call(call):
        print(f"  tool> {call.name}({json.dumps(call.arguments)})")

    def on_result(call, result):
        if verbose:
            print(f"  result> {_short(result, 300)}")

    return AgentEvents(on_tool_call=on_call, on_tool_result=on_result)


def _short(text: str, limit: int = 120) -> str:
    text = text.replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 3] + "..."


if __name__ == "__main__":
    sys.exit(main())
