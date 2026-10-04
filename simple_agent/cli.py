"""The terminal interface: a read-eval-print loop around the Agent."""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys

from . import __version__
from .agent import Agent, AgentEvents
from .config import PROVIDER_DEFAULTS, load_file, resolve
from . import config_update, instructions, mcp, memory, planning, sessions, subagents
from .plugins import PluginLoader
from .providers import create_provider
from .context import CLEARED_PREFIX
from .tools import default_tools
from . import usage
from .usage import UsageMeter
from .trace import Tracer, describe_context, describe_subagent, render_plan

HELP = """Commands:
  /help          show this help
  /tools         list the tools the agent can use
  /plan          show the agent's current plan
  /instructions  show the project instructions (AGENTS.md) the agent follows
  /memory        list the agent's long-term memory notes
  /memory forget <title>  delete a note
  /plugins       list the plugin files and the tools they added
  /reload        load the plugin files again (after editing one)
  /history       show the conversation so far
  /usage         show tokens (and cost) used this session
  /usage on|off  show or hide the tokens/cost line after each answer
  /reset         start a new conversation
  /context       show how big the conversation is and the context settings
  /compact       summarize older turns now (keeps the last few word for word)
  /save [name]   save the conversation (default name: the current time)
  /load <name>   load a saved conversation
  /sessions      list saved conversations
  /exit          quit (Ctrl+D also works). The conversation is saved as "last"."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="simple-agent", description="A small CLI agent for learning.", epilog="Run the eval tasks with: simple-agent eval. Add new settings to your config.toml with: simple-agent config.")
    parser.add_argument("--provider", choices=list(PROVIDER_DEFAULTS), help="Which backend to use.")
    parser.add_argument("--model", help="Model name, e.g. the id LM Studio shows, or claude-opus-5-5.")
    parser.add_argument("--base-url", help="API base URL for the openai provider (LM Studio, Ollama, ...).")
    parser.add_argument("--config", help="Path to a config.toml (default: ./config.toml, then ~/.config/simple-agent/).")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show tool results, not just tool calls.")
    parser.add_argument("--trace", action="store_true", default=None, help="Print every step of the agent loop and save it to a log file.")
    parser.add_argument("--resume", metavar="NAME", help='Continue a saved conversation, e.g. --resume last.')
    parser.add_argument("--no-stream", action="store_true", help="Wait for whole replies instead of streaming them.")
    parser.add_argument("--no-plugins", action="store_true", help="Don't load tools from the plugin folders.")
    parser.add_argument("--no-planning", action="store_true", help="Don't offer the model the update_plan tool.")
    parser.add_argument("--no-subagents", action="store_true", help="Don't offer the model the delegate tool.")
    parser.add_argument("--no-instructions", action="store_true", help="Don't read AGENTS.md project instructions.")
    parser.add_argument("--no-memory", action="store_true", help="Don't give the agent long-term memory notes.")
    parser.add_argument("--no-usage", action="store_true", help="Don't show tokens and cost after each answer.")
    parser.add_argument("--no-mcp", action="store_true", help="Don't start the MCP servers from the config file.")
    parser.add_argument("--trace-dir", help="Where trace logs go (default: ./traces).")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("prompt", nargs="*", help="Ask one question and exit instead of starting the REPL.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["eval"]:  # `simple-agent eval ...` runs the eval tasks instead of the chat.
        from . import evals

        return evals.main(argv[1:])
    if argv[:1] == ["config"]:  # `simple-agent config` adds new settings to your config.toml.
        return config_update.main(argv[1:])
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
    # MCP servers from config.toml add their tools next to the built-in ones.
    # (A copy of the registry, so the built-in list itself isn't changed.)
    servers = []
    if settings.mcp_servers and not args.no_mcp:
        agent.tools = default_tools.copy()
        servers, messages = mcp.connect(settings.mcp_servers, agent.tools)
        for message in messages:
            print(f"({message})")
    agent.tools = agent.tools.copy()  # Our own registry, so setting approve doesn't change the shared one.
    agent.tools.approve = _confirm  # write_file, run_shell and MCP tools ask here before running.
    if settings.project_instructions and not args.no_instructions:
        project = instructions.enable(agent)  # AGENTS.md files join the system prompt.
        for line in project.describe():
            print(f"(project instructions: {line})")
    if settings.memory and not args.no_memory:
        memory.enable(agent, memory.Memory(settings.memory_dir, on_change=_memory_line))  # See memory.py.
    if settings.subagents and not args.no_subagents:
        subagents.enable(agent)  # Adds the delegate tool (see subagents.py).
    if settings.planning and not args.no_planning:
        planning.enable(agent)  # Adds the update_plan tool (see planning.py).
    # Plugins go on top, from a loader that can load them again for /reload.
    loader = PluginLoader([] if args.no_plugins else settings.plugin_dirs, agent.tools, settings.plugin_settings)
    agent.tools = loader.load()
    for line in loader.describe():
        print(f"({line})")
    try:
        return _run(args, settings, agent, printer, provider, loader)
    finally:
        for server in servers:
            server.close()


def _run(args: argparse.Namespace, settings, agent: Agent, printer: Printer, provider, loader: PluginLoader) -> int:
    """Everything after setup: trace mode, --resume, then one question or the REPL."""
    meter = UsageMeter(settings.model, settings.prices, show=settings.show_usage and not args.no_usage)
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
        return 0 if _ask(agent, printer, " ".join(args.prompt), meter) else 1

    where = f" at {settings.base_url}" if settings.provider == "openai" else ""
    print(f"simple-agent {__version__} · {settings.provider} · {settings.model}{where}")
    note = config_update.hint(args.config)  # Your config.toml is older than this version?
    if note:
        print(note)
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
            if _command(agent, line, settings.sessions_dir, loader, meter) == "exit":
                break
            continue
        _ask(agent, printer, line, meter)

    if agent.history:  # So `--resume last` always picks up where you left off.
        path = sessions.save(agent, sessions.session_path(settings.sessions_dir, "last"))
        print(f"(conversation saved to {path})")
    return 0


def _ask(agent: Agent, printer: Printer, text: str, meter: UsageMeter | None = None) -> bool:
    if meter:
        meter.start_turn(agent.usage)
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
    if meter and meter.show:
        print(f"{meter.turn_line(agent.usage)}\n")
    return True


def _command(
    agent: Agent, line: str, sessions_dir: str = "sessions", loader: PluginLoader | None = None,
    meter: UsageMeter | None = None,
) -> str | None:
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
    elif cmd == "/plugins":
        lines = loader.describe() if loader else []
        if not lines:
            folders = ", ".join(loader.dirs) if loader and loader.dirs else "none (--no-plugins)"
            print(f"  (no plugins loaded; plugin folders: {folders})")
        for line in lines:
            print(f"  {line}")
    elif cmd == "/reload":
        if loader is None:
            print("  (plugins are not enabled)")
            return None
        agent.tools = loader.load()
        for line in loader.describe() or ["(no plugins found)"]:
            print(f"  {line}")
    elif cmd == "/plan":
        planner = planning.planner_of(agent)
        if planner is None:
            print("  (planning is off)")
        else:
            for line in render_plan(planner.steps).splitlines():
                print(f"  {line}")
    elif cmd == "/history":
        for m in agent.history:
            if m.role == "tool":
                print(f"  [tool result] {_short(m.content)}")
            else:
                calls = "".join(f" [calls {c.name}]" for c in m.tool_calls)
                print(f"  [{m.role}] {_short(m.content)}{calls}")
    elif cmd == "/instructions":
        project = instructions.instructions_of(agent)
        if not project:
            print("  (project instructions are off)")
        elif not project.files():
            print(f"  (no {instructions.FILE_NAME} here; create one to give the agent rules for this project)")
        else:
            for path in project.files():
                print(f"  --- {path} ---")
                print("  " + project.read(path).replace("\n", "\n  "))
    elif cmd == "/memory":
        notes = memory.memory_of(agent)
        arg = rest[0].strip() if rest else ""
        if notes is None:
            print("  (long-term memory is off)")
        elif arg.lower().startswith("forget"):
            title = arg[len("forget"):].strip()
            result = notes.delete(title) if title else "usage: /memory forget <title>"
            if not notes.on_change or not result.startswith("Deleted"):  # Otherwise _memory_line said it.
                print(f"  {result}")
        elif arg:
            print("  usage: /memory or /memory forget <title>")
        elif not notes.notes():
            print(f"  (no notes yet in {notes.folder}/; the agent saves them with save_memory)")
        else:
            for note in notes.notes():
                print(f"  {note.title}: {note.description or '(no description)'} ({note.updated or 'no date'})")
            print(f"  (the notes are Markdown files in {notes.folder}/; edit or delete them as you like)")
    elif cmd == "/usage":
        meter = meter or UsageMeter(model=getattr(agent.provider, "model", ""))
        arg = rest[0].strip().lower() if rest else ""
        if arg in ("on", "off"):
            meter.show = arg == "on"
            print(f"  (the tokens line after each answer is {arg})")
        elif arg:
            print("  usage: /usage, /usage on or /usage off")
        else:
            print(f"  this session: {meter.session_line(agent.usage)}")
            if usage.cost(agent.usage, meter.model, meter.prices) is None:
                print(f"  (no price for {meter.model!r}; add one under [prices] in config.toml to see the cost)")
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


def _memory_line(kind: str, title: str, path) -> None:
    """Every change to long-term memory shows in the chat, so nothing is remembered behind your back."""
    print(f"  memory> {kind} {title!r} ({path})")


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
            on_subagent=self.subagent,
            on_plan=self.plan,
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

    def subagent(self, kind: str, details: dict) -> None:
        if kind == "tool_result" and not self.verbose:
            return
        self.end_line()
        print(f"  sub-agent> {describe_subagent(kind, details, 300 if self.verbose else 120)}")

    def plan(self, steps: list[dict]) -> None:
        self.end_line()
        lines = render_plan(steps).splitlines()
        print(f"  plan> {lines[0]}")
        for line in lines[1:]:
            print(f"        {line}")


def _short(text: str, limit: int = 120) -> str:
    text = text.replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 3] + "..."


if __name__ == "__main__":
    sys.exit(main())
