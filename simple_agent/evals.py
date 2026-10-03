"""The eval runner: run a fixed set of tasks against a model and score the answers.

    simple-agent eval                                   # every task in evals/
    simple-agent eval --provider anthropic              # the same tasks on Claude
    simple-agent eval --compare eval_results/a.json eval_results/b.json

Each task is a prompt plus checks, written in a TOML file in evals/:

    [[task]]
    id = "multiply"
    prompt = "What is 1234 * 5678?"
    check.answer_number = 7006652

For every task the runner starts a fresh Agent (the same loop the chat uses,
with the same tools and plugins) in a new empty folder, puts the task's files
there, sends the prompt, and then runs the checks on the final answer and on
the folder. It records whether the task passed, how many model calls (steps)
it took, which tools were called, the tokens used and the whole conversation,
and saves one JSON file per run in eval_results/ so runs can be compared.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import json
import os
import re
import sys
import tempfile
import threading
import time
import tomllib
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import __version__, instructions, planning, subagents, usage
from .agent import Agent, AgentEvents
from .config import load_file, resolve
from .plugins import PluginLoader
from .providers import create_provider
from .providers.base import Provider
from .tools import ToolRegistry, default_tools
from .trace import _message_dict


@dataclass
class Task:
    id: str  # "<file>/<id>", e.g. "math/multiply".
    prompt: str
    checks: dict[str, Any]
    files: dict[str, str] = field(default_factory=dict)  # Created in the task's folder first.
    approve: list[str] = field(default_factory=lambda: ["write_file"])  # Tools allowed without asking.
    # Web pages served on a local test server while the task runs ({path: content}).
    # "{server}" in the prompt, files, pages and env is replaced by its address.
    pages: dict[str, str] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)  # Environment variables set during the task.


# --- loading tasks -------------------------------------------------------------


def load_tasks(paths: list[str | Path]) -> list[Task]:
    """Read tasks from TOML files, or from every .toml file in a folder."""
    files: list[Path] = []
    for p in map(Path, paths):
        files += sorted(p.glob("*.toml")) if p.is_dir() else [p]
    tasks = []
    for path in files:
        with path.open("rb") as f:
            data = tomllib.load(f)
        for entry in data.get("task", []):
            name = f"{path.stem}/{entry['id']}"
            unknown = set(entry.get("check", {})) - set(CHECKS)
            if unknown:
                raise ValueError(f"{name}: unknown check(s) {', '.join(sorted(unknown))}. Known: {', '.join(CHECKS)}")
            if not entry.get("check"):
                raise ValueError(f"{name}: a task needs at least one check")
            tasks.append(Task(
                id=name,
                prompt=entry["prompt"],
                checks=entry["check"],
                files=entry.get("files", {}),
                approve=entry.get("approve", ["write_file"]),
                pages=entry.get("pages", {}),
                env=entry.get("env", {}),
            ))
    return tasks


# --- checks ----------------------------------------------------------------------
# Each check gets the run (answer, folder, tools used, steps) and the value
# written in the task file, and returns (passed, detail).


@dataclass
class Run:
    answer: str
    folder: Path
    tools_called: list[str]
    steps: int
    plan: list[dict] = field(default_factory=list)  # The final plan, if the agent made one.
    pages_requested: list[str] = field(default_factory=list)  # Paths fetched from the task's test server.


def _as_list(value) -> list:
    return value if isinstance(value, list) else [value]


def _numbers(text: str) -> list[float]:
    """Every number in the text, with thousands separators removed: "7,006,652" -> 7006652."""
    found = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    return [float(n.replace(",", "")) for n in found if n.replace(",", "").lstrip("-")]


def check_answer_contains(run: Run, expected) -> tuple[bool, str]:
    missing = [e for e in _as_list(expected) if e.lower() not in run.answer.lower()]
    return not missing, f"answer is missing {missing}" if missing else "ok"


def check_answer_contains_any(run: Run, expected) -> tuple[bool, str]:
    ok = any(e.lower() in run.answer.lower() for e in _as_list(expected))
    return ok, "ok" if ok else f"answer has none of {expected}"


def check_answer_not_contains(run: Run, unwanted) -> tuple[bool, str]:
    found = [u for u in _as_list(unwanted) if u.lower() in run.answer.lower()]
    return not found, f"answer contains {found}" if found else "ok"


def check_answer_matches(run: Run, pattern) -> tuple[bool, str]:
    ok = re.search(pattern, run.answer, re.IGNORECASE | re.DOTALL) is not None
    return ok, "ok" if ok else f"answer doesn't match /{pattern}/"


def check_answer_number(run: Run, expected) -> tuple[bool, str]:
    ok = any(abs(n - expected) <= 1e-6 * max(1, abs(expected)) for n in _numbers(run.answer))
    return ok, "ok" if ok else f"{expected} not found in the answer"


def check_file_exists(run: Run, paths) -> tuple[bool, str]:
    missing = [p for p in _as_list(paths) if not (run.folder / p).exists()]
    return not missing, f"missing file(s) {missing}" if missing else "ok"


def check_file_contains(run: Run, expected: dict) -> tuple[bool, str]:
    """expected is {path = "text it must contain"}."""
    for path, text in expected.items():
        target = run.folder / path
        if not target.is_file():
            return False, f"{path} was not created"
        if text not in target.read_text(encoding="utf-8", errors="replace"):
            return False, f"{path} doesn't contain {text!r}"
    return True, "ok"


def check_file_matches(run: Run, expected: dict) -> tuple[bool, str]:
    """expected is {path = "regex"} or {path = ["regex", ...]}; every pattern must match."""
    for path, patterns in expected.items():
        target = run.folder / path
        if not target.is_file():
            return False, f"{path} was not created"
        text = target.read_text(encoding="utf-8", errors="replace")
        for pattern in _as_list(patterns):
            if not re.search(pattern, text, re.IGNORECASE | re.MULTILINE):
                return False, f"{path} doesn't match /{pattern}/"
    return True, "ok"


def check_tool_used(run: Run, names) -> tuple[bool, str]:
    missing = [n for n in _as_list(names) if n not in run.tools_called]
    return not missing, f"never called {missing}" if missing else "ok"


def check_plan_made(run: Run, expected: bool) -> tuple[bool, str]:
    made = bool(run.plan)
    return made == expected, "ok" if made == expected else ("no plan was made" if expected else "a plan was made")


def check_plan_completed(run: Run, expected: bool) -> tuple[bool, str]:
    done = bool(run.plan) and all(s["status"] == "done" for s in run.plan)
    if done == expected:
        return True, "ok"
    if not run.plan:
        return False, "no plan was made"
    left = [s["step"] for s in run.plan if s["status"] != "done"]
    return False, f"steps not done: {left}" if expected else "every step was done"


def check_file_absent(run: Run, paths) -> tuple[bool, str]:
    found = [p for p in _as_list(paths) if (run.folder / p).exists()]
    return not found, f"file(s) {found} should not exist" if found else "ok"


def check_tool_not_used(run: Run, names) -> tuple[bool, str]:
    used = [n for n in _as_list(names) if n in run.tools_called]
    return not used, f"called {used}" if used else "ok"


def check_page_not_requested(run: Run, paths) -> tuple[bool, str]:
    hits = [r for r in run.pages_requested for p in _as_list(paths) if p in r]
    return not hits, f"requested {hits}" if hits else "ok"


def check_max_steps(run: Run, limit: int) -> tuple[bool, str]:
    return run.steps <= limit, "ok" if run.steps <= limit else f"took {run.steps} steps (limit {limit})"


CHECKS = {
    "answer_contains": check_answer_contains,  # Text (or list of texts), case-insensitive.
    "answer_contains_any": check_answer_contains_any,
    "answer_not_contains": check_answer_not_contains,
    "answer_matches": check_answer_matches,  # A regular expression.
    "answer_number": check_answer_number,  # A number that must appear in the answer.
    "file_exists": check_file_exists,
    "file_contains": check_file_contains,
    "file_matches": check_file_matches,  # {path = regex or list of regexes}, ignoring case.
    "tool_used": check_tool_used,
    "tool_not_used": check_tool_not_used,  # E.g. a planted instruction tried to make it read a file.
    "file_absent": check_file_absent,
    "page_not_requested": check_page_not_requested,  # No request to the test server contained this text.
    "max_steps": check_max_steps,  # At most this many model calls.
    "plan_made": check_plan_made,  # true: the agent used update_plan (planning.py).
    "plan_completed": check_plan_completed,  # true: every step of its plan ended up done.
}


# --- running -------------------------------------------------------------------


def run_task(task: Task, make_agent, log=print, prices: dict | None = None) -> dict:
    """Run one task in a fresh temporary folder and return its result."""
    tools_called: list[str] = []  # By the agent and by any sub-agents it starts.
    helpers: list[dict] = []  # One entry per sub-agent: its task, answer and steps.
    steps = 0

    def count_step(step, reply):
        nonlocal steps
        steps = step

    def on_subagent(kind, details):
        if kind == "tool_call":
            tools_called.append(details["name"])
        elif kind == "end":
            helpers.append(dict(details))

    home = os.getcwd()
    requests: list[str] = []  # Paths fetched from the test server.
    with tempfile.TemporaryDirectory(prefix="simple-agent-eval-") as tmp, _serve(task.pages, requests) as server:
        fill = lambda text: text.replace("{server}", server)  # noqa: E731
        folder = Path(tmp)
        for path, content in task.files.items():
            (folder / path).parent.mkdir(parents=True, exist_ok=True)
            (folder / path).write_text(fill(content), encoding="utf-8")

        agent: Agent = make_agent()
        agent.events = AgentEvents(
            on_model_reply=count_step,
            on_tool_call=lambda call: tools_called.append(call.name),
            on_subagent=on_subagent,
        )
        agent.tools.approve = lambda name, arguments: name in task.approve  # No one to ask during an eval.
        started = time.perf_counter()
        error = None
        os.chdir(folder)  # The tools use relative paths, so the task works inside its own folder.
        saved_env = {key: os.environ.get(key) for key in task.env}
        os.environ.update({key: fill(value) for key, value in task.env.items()})
        try:
            answer = agent.ask(fill(task.prompt))
        except Exception as exc:  # noqa: BLE001 - a failed call fails the task, not the run
            answer, error = "", f"{type(exc).__name__}: {exc}"
        finally:
            os.chdir(home)
            for key, value in saved_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        seconds = time.perf_counter() - started

        planner = planning.planner_of(agent)
        plan = planner.steps if planner else []
        run = Run(answer, folder, tools_called, steps, plan, requests)
        checks = {}
        for name, expected in task.checks.items():
            passed, detail = CHECKS[name](run, expected)
            checks[name] = {"passed": passed, "detail": detail}

    passed = error is None and all(c["passed"] for c in checks.values())
    result = {
        "task": task.id,
        "passed": passed,
        "error": error,
        "checks": checks,
        "answer": answer,
        "steps": steps,
        "tools_called": tools_called,
        "subagents": helpers,
        "plan": plan,
        "pages_requested": requests,
        "input_tokens": agent.usage.get("input_tokens", 0),
        "output_tokens": agent.usage.get("output_tokens", 0),
        "model_calls": agent.usage.get("model_calls", 0),  # Including sub-agents' calls.
        "cost_usd": usage.cost(agent.usage, getattr(agent.provider, "model", ""), prices or usage.DEFAULT_PRICES),
        "seconds": round(seconds, 2),
        "messages": [_message_dict(m) for m in agent.history],
    }
    log(_result_line(result))
    return result


@contextlib.contextmanager
def _serve(pages: dict[str, str], requests: list[str]):
    """Serve `pages` on a local port for one task; yields its address ("" if no pages)."""
    if not pages:
        yield ""
        return

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = urllib.parse.urlsplit(self.path).path.lstrip("/")  # The query string is ignored.
            requests.append(self.path)
            if path not in served:
                self.send_error(404)
                return
            body = served[path].encode("utf-8")
            kind = {"html": "text/html", "json": "application/json"}.get(path.rsplit(".", 1)[-1], "text/plain")
            if path.endswith("search"):  # A stand-in for SearXNG's /search endpoint.
                kind = "application/json"
            self.send_response(200)
            self.send_header("Content-Type", f"{kind}; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    address = f"http://127.0.0.1:{server.server_port}"
    served = {path.lstrip("/"): content.replace("{server}", address) for path, content in pages.items()}
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield address
    finally:
        server.shutdown()
        server.server_close()


def run_tasks(tasks: list[Task], make_agent, repeat: int = 1, log=print, prices: dict | None = None) -> dict:
    """Run every task `repeat` times. Returns the results and a summary."""
    results = [run_task(task, make_agent, log, prices) for _ in range(repeat) for task in tasks]
    summary = summarize(results)
    log(_summary_line(summary))
    return {"results": results, "summary": summary}


def summarize(results: list[dict]) -> dict:
    n = len(results)
    passed = sum(r["passed"] for r in results)
    return {
        "tasks": n,
        "passed": passed,
        "score": round(passed / n, 3) if n else 0.0,
        "steps": sum(r["steps"] for r in results),
        "input_tokens": sum(r["input_tokens"] for r in results),
        "output_tokens": sum(r["output_tokens"] for r in results),
        # None when the model has no price (a local model).
        "cost_usd": None if any(r.get("cost_usd") is None for r in results) else sum(r["cost_usd"] for r in results),
        "seconds": round(sum(r["seconds"] for r in results), 2),
    }


def save(run: dict, out_dir: str | Path, provider: Provider) -> Path:
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    model = re.sub(r"[^A-Za-z0-9._-]+", "_", provider.model)
    path = Path(out_dir) / f"{stamp}-{provider.name}-{model}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(run, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return path


def compare(paths: list[str | Path], out=None) -> None:
    """Print a table: one row per task, one column per results file."""
    out = out or sys.stdout
    runs = [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]
    names = [f"{r['provider']}/{r['model']}" for r in runs]
    tasks = list(dict.fromkeys(res["task"] for r in runs for res in r["results"]))
    width = max([len("total")] + [len(t) for t in tasks])
    cols = [max(len(n), 22) for n in names]
    print(f"{'task':<{width}}  " + "  ".join(f"{n:<{c}}" for n, c in zip(names, cols)), file=out)
    for task in tasks:
        cells = []
        for r, c in zip(runs, cols):
            mine = [res for res in r["results"] if res["task"] == task]
            if not mine:
                cells.append(f"{'-':<{c}}")
                continue
            ok = sum(res["passed"] for res in mine)
            steps = sum(res["steps"] for res in mine) / len(mine)
            tokens = sum(res["input_tokens"] + res["output_tokens"] for res in mine) / len(mine)
            mark = "pass" if ok == len(mine) else "FAIL" if ok == 0 else f"{ok}/{len(mine)}"
            cells.append(f"{f'{mark} {steps:.0f} steps {tokens:.0f} tok':<{c}}")
        print(f"{task:<{width}}  " + "  ".join(cells), file=out)
    scores = [f"{r['summary']['passed']}/{r['summary']['tasks']} passed" for r in runs]
    print(f"{'total':<{width}}  " + "  ".join(f"{s:<{c}}" for s, c in zip(scores, cols)), file=out)


def _result_line(r: dict) -> str:
    status = "PASS" if r["passed"] else "FAIL"
    tokens = r["input_tokens"] + r["output_tokens"]
    helpers = f", {len(r['subagents'])} sub-agents" if r.get("subagents") else ""
    line = f"  {status}  {r['task']}  ({r['steps']} steps{helpers}, {tokens} tokens, {r['seconds']}s)"
    if r["error"]:  # The model call failed, so the checks don't tell us anything.
        return line + f"\n        error: {r['error']}"
    for name, c in r["checks"].items():
        if not c["passed"]:
            line += f"\n        {name}: {c['detail']}"
    return line


def _summary_line(s: dict) -> str:
    return (
        f"\n{s['passed']}/{s['tasks']} passed ({s['score']:.0%}), {s['steps']} steps, "
        f"{s['input_tokens']} input + {s['output_tokens']} output tokens"
        f"{', ' + usage.format_dollars(s['cost_usd']) if s.get('cost_usd') is not None else ''}, {s['seconds']}s"
    )


# --- the `simple-agent eval` command ------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="simple-agent eval", description="Run the eval tasks and score the answers.")
    parser.add_argument("tasks", nargs="*", default=["evals"], help="Task files or folders (default: evals/).")
    parser.add_argument("--provider", help="Which backend to use.")
    parser.add_argument("--model", help="Model name.")
    parser.add_argument("--base-url", help="API base URL for the openai provider.")
    parser.add_argument("--config", help="Path to a config.toml.")
    parser.add_argument("--only", help="Only run tasks whose id contains this text, e.g. --only math/.")
    parser.add_argument("--repeat", type=int, default=1, help="Run each task this many times (models vary).")
    parser.add_argument("--out", default="eval_results", help="Where to save the results (default: eval_results/).")
    parser.add_argument("--no-plugins", action="store_true", help="Don't load plugin tools.")
    parser.add_argument("--no-subagents", action="store_true", help="Don't offer the delegate tool.")
    parser.add_argument("--no-instructions", action="store_true", help="Don't read AGENTS.md files in task folders.")
    parser.add_argument("--no-planning", action="store_true", help="Don't offer the update_plan tool.")
    parser.add_argument("--compare", nargs="+", metavar="RESULTS", help="Compare saved results files instead of running.")
    args = parser.parse_args(argv)

    if args.compare:
        compare(args.compare)
        return 0

    try:
        settings = resolve(load_file(args.config), args.provider, args.model, args.base_url)
        provider = create_provider(settings.provider, settings.model, settings.base_url, settings.api_key)
        tasks = [t for t in load_tasks(args.tasks) if not args.only or args.only in t.id]
    except Exception as exc:  # noqa: BLE001
        print(f"Setup error: {exc}", file=sys.stderr)
        return 1
    if not tasks:
        print(f"No tasks found in {', '.join(args.tasks)}", file=sys.stderr)
        return 1

    # The same tools as the chat: built-ins plus plugins. (MCP servers are
    # left out so every run starts quickly and the same way.)
    tools: ToolRegistry = PluginLoader(
        [] if args.no_plugins else settings.plugin_dirs, default_tools, settings.plugin_settings
    ).load()

    use_subagents = settings.subagents and not args.no_subagents
    use_planning = settings.planning and not args.no_planning
    use_instructions = settings.project_instructions and not args.no_instructions

    def make_agent() -> Agent:
        agent = Agent(
            provider=provider,
            tools=tools.copy(),
            system_prompt=settings.system_prompt,
            max_steps=settings.max_steps,
            stream=False,
            context=settings.context,
        )
        if use_instructions:
            # Only AGENTS.md files in the task's own folder: not your personal
            # file, so results don't depend on who runs the evals.
            instructions.enable(agent, instructions.ProjectInstructions(user_file=None))
        if use_subagents:
            subagents.enable(agent)
        if use_planning:
            planning.enable(agent)
        return agent

    print(f"Running {len(tasks)} tasks x{args.repeat} on {settings.provider} · {settings.model}\n")
    started_at = _dt.datetime.now().astimezone().isoformat(timespec="seconds")
    run = run_tasks(tasks, make_agent, args.repeat, prices=settings.prices)
    run = {
        "provider": settings.provider,
        "model": settings.model,
        "base_url": settings.base_url if settings.provider == "openai" else None,
        "simple_agent_version": __version__,
        "started_at": started_at,
        "system_prompt": settings.system_prompt,
        "max_steps": settings.max_steps,
        "tools": make_agent().tools.names(),
        "subagents": use_subagents,
        "planning": use_planning,
        "project_instructions": use_instructions,
        **run,
    }
    print(f"Results saved to {save(run, args.out, provider)}")
    return 0
