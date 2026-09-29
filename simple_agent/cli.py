"""The terminal interface: a read-eval-print loop around the Agent."""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys

from . import __version__
from .agent import Agent, AgentEvents
from .config import PROVIDER_DEFAULTS, load_file, resolve
from . import sessions
from .providers import create_provider
from .context import CLEARED_PREFIX
from .trace import Tracer, describe_context

HELP = """Commands:
  /help          show this help
  /tools         list the tools the agent can use
  /history       show the conversation so far
  /usage         show tokens used this session
  /reset         start a new conversation
  /context       show how big the conversation is and the context settings
  /compact       summarize older turns now (keeps the last few word for word)
  /save [name]   save the conversation (default name: the current time)
  /load <name>   load a saved conversation
  /sessions      list saved conversations
  /exit          quit (Ctrl+D also works). The conversation is saved as "last"."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="simple-agent", description="A small CLI agent for learning.")
    parser.add_argument("--provider", choices=list(PROVIDER_DEFAULTS), help="Which backend to use.")
    parser.add_argument("--model", help="Model name, e.g. the id LM Studio shows, or claude-opus-5-5.")
    parser.add_argument("--base-url", help="API base URL for the openai provider (LM Studio, Ollama, ...).")
    parser.add_argument("--config", help="Path to a config.toml (default: ./config.toml, then ~/.config/simple-agent/).")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show tool results, not just tool calls.")
    parser.add_argument("--trace", action="store_true", default=None, help="Print every step of the agent loop and save it to a log file.")
    parser.add_argument("--resume", metavar="NAME", help='Continue a saved conversation, e.g. --resume last.')
    parser.add_argument("--no-stream", action="store_true", help="Wait for whole replies instead of streaming them.")
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

    printer = Printer(args.verbose)
    agent = Agent(
        provider=provider,
        system_prompt=settings.system_prompt,
        max_steps=settings.max_steps,
        stream=settings.stream and not args.no_stream,
        events=printer.events(),
        context=settings.context,
    )
    agent.tools.approve = _confirm  # write_file and run_shell ask here before running.
    trace = settings.trace if args.trace is None else args.trace
    if trace:
        # The tracer prints each step itself (whole replies, not streamed),
        # so it replaces the normal printing.
        tracer = Tracer(args.trace_dir or settings.trace_dir)
        tracer.start(provider, agent.system_prompt, agent.tools.specs())
        agent.events = tracer.events()
        agent.stream = False

    if args.resume:
        try:
            data = sessions.load(agent, sessions.session_path(settings.sessions_dir, args.resume))
        except (OSError, ValueError) as exc:
            print(f"Could not resume {args.resume!r}: {exc}", file=sys.stderr)
            return 1
        print(f"(resumed {args.resume}: {len(agent.history)} messages, saved {data['saved_at']})")

    if args.prompt:  # One-shot mode: simple-agent "what time is it?"
        return 0 if _ask(agent, printer, " ".join(args.prompt)) else 1

    where = f" at {settings.base_url}" if settings.provider == "openai" else ""
    print(f"simple-agent {__version__} · {settings.provider} · {settings.model}{where}")
    print("Type a message, or /help for commands.\n")

    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.startswith("/"):
            if _command(agent, line, settings.sessions_dir) == "exit":
                break
            continue
        _ask(agent, printer, line)

    if agent.history:  # So `--resume last` always picks up where you left off.
        path = sessions.save(agent, sessions.session_path(settings.sessions_dir, "last"))
        print(f"(conversation saved to {path})")
    return 0


def _ask(agent: Agent, printer: Printer, text: str) -> bool:
    try:
        answer = agent.ask(text)
    except KeyboardInterrupt:
        printer.end_line()
        print("(interrupted)")
        return False
    except Exception as exc:  # noqa: BLE001 - show API/network errors without crashing the REPL
        printer.end_line()
        print(f"error> {type(exc).__name__}: {exc}", file=sys.stderr)
        if type(exc).__name__ == "APIConnectionError":  # Same class name in both SDKs.
            print("       Is the server running and is --base-url right? (LM Studio: Developer tab > Start Server)", file=sys.stderr)
        return False
    printer.end_line()
    if printer.streamed:
        print()  # The answer is already on screen; just leave a blank line.
    else:
        print(f"agent> {answer}\n")
    return True


def _command(agent: Agent, line: str, sessions_dir: str = "sessions") -> str | None:
    cmd, *rest = line.split(maxsplit=1)
    cmd = cmd.lower()
    if cmd in ("/exit", "/quit"):
        return "exit"
    if cmd == "/help":
        print(HELP)
    elif cmd == "/tools":
        for spec in agent.tools.specs():
            note = " (asks first)" if agent.tools.asks_first(spec.name) else ""
            print(f"  {spec.name}{note}: {spec.description}")
    elif cmd == "/history":
        for m in agent.history:
            if m.role == "tool":
                print(f"  [tool result] {_short(m.content)}")
            else:
                calls = "".join(f" [calls {c.name}]" for c in m.tool_calls)
                print(f"  [{m.role}] {_short(m.content)}{calls}")
    elif cmd == "/usage":
        print(f"  input tokens: {agent.usage.get('input_tokens', 0)}, output tokens: {agent.usage.get('output_tokens', 0)}")
    elif cmd == "/context":
        ctx = agent.context
        tool_results = [m for m in agent.history if m.role == "tool"]
        cleared = sum(m.content.startswith(CLEARED_PREFIX) for m in tool_results)
        print(f"  {len(agent.history)} messages, {len(tool_results)} tool results ({cleared} cleared)")
        if ctx is None:
            print("  context management is off")
        else:
            print(f"  about {ctx.estimate(agent)} tokens (estimated) of a {ctx.max_tokens} token limit")
            print(f"  clear old tool results: {'on' if ctx.clear_tool_results else 'off'} (keeps the newest {ctx.keep_tool_results})")
            print(f"  compact older turns: {'on' if ctx.compact else 'off'} (keeps the last {ctx.keep_recent_turns} turns)")
    elif cmd == "/compact":
        if agent.context is None:
            print("  context management is off")
        elif not agent.context.compact_history(agent):
            print("  (nothing to compact yet)")
    elif cmd == "/reset":
        agent.reset()
        print("  (conversation cleared)")
    elif cmd == "/save":
        name = rest[0].strip() if rest else _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        try:
            print(f"  (saved to {sessions.save(agent, sessions.session_path(sessions_dir, name))})")
        except (OSError, ValueError) as exc:
            print(f"  Could not save: {exc}")
    elif cmd == "/load":
        if not rest:
            print("  Usage: /load <name>   (see /sessions)")
            return None
        try:
            data = sessions.load(agent, sessions.session_path(sessions_dir, rest[0].strip()))
        except (OSError, ValueError) as exc:
            print(f"  Could not load: {exc}")
            return None
        print(f"  (loaded {len(agent.history)} messages, saved {data['saved_at']} with {data['provider']} · {data['model']})")
    elif cmd == "/sessions":
        found = sessions.list_sessions(sessions_dir)
        if not found:
            print(f"  (no saved sessions in {sessions_dir}/)")
        for name, info in found:
            print(f"  {name}: {info['messages']} messages, {info['saved_at']}, {info['model']}")
            print(f"      {_short(info['first_message'], 80)}")
    else:
        print(f"  Unknown command {cmd}. Type /help.")
    return None


def _confirm(name: str, arguments: dict) -> bool:
    """Show a tool call that changes something and ask the user to allow it."""
    print(f"  approve> the agent wants to run {name} with:")
    for key, value in arguments.items():
        lines = str(value).splitlines() or [""]
        shown = lines[:15] + ([f"... ({len(lines) - 15} more lines)"] if len(lines) > 15 else [])
        print(f"    {key}: {shown[0]}")
        for line in shown[1:]:
            print(f"    {' ' * len(key)}  {line}")
    try:
        answer = input("  Allow? [y/N] ").strip().lower()
    except EOFError:  # No one to ask (e.g. input is piped in), so the answer is no.
        print()
        return False
    return answer in ("y", "yes")


class Printer:
    """Prints the agent's activity: streamed text as it arrives, and tool calls."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.mid_line = False  # True while a streamed reply is still being printed.
        self.streamed = False  # Whether any text was streamed during this turn.

    def events(self) -> AgentEvents:
        return AgentEvents(
            on_turn_start=self.turn_start,
            on_text=self.text,
            on_tool_call=self.tool_call,
            on_tool_result=self.tool_result,
            on_context=self.context,
        )

    def turn_start(self, user_input: str) -> None:
        self.streamed = False

    def text(self, chunk: str) -> None:
        if not self.mid_line:
            print("agent> ", end="")
            self.mid_line = True
        self.streamed = True
        print(chunk, end="", flush=True)

    def end_line(self) -> None:
        if self.mid_line:
            print()
            self.mid_line = False

    def tool_call(self, call) -> None:
        self.end_line()
        print(f"  tool> {call.name}({json.dumps(call.arguments)})")

    def tool_result(self, call, result: str) -> None:
        if self.verbose:
            print(f"  result> {_short(result, 300)}")


    def context(self, kind: str, details: dict) -> None:
        self.end_line()
        print(f"  context> {describe_context(kind, details)}")
        if kind == "compacted" and self.verbose:
            print(f"  summary> {_short(details['summary'], 600)}")


def _short(text: str, limit: int = 120) -> str:
    text = text.replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 3] + "..."


if __name__ == "__main__":
    sys.exit(main())
