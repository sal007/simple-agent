"""The sandbox: keep the file tools and run_shell inside one folder, the workspace.

By default the workspace is the folder you start simple-agent in. Inside it,
the agent works as before; outside it, the tools refuse, before you're even
asked y/N:

  - list_files, read_file and write_file resolve every path (following
    symlinks and "..") and refuse anything outside the workspace. Relative
    paths count from the workspace. Some files inside it are protected too
    (.env and config.toml by default, since they can hold API keys).

  - run_shell runs in the workspace, and a command that names a path outside
    it (/etc/passwd, ~/.ssh, ../other-project) or a protected file is refused.
    That check reads the command's text, so it's a speed bump, not a wall: a
    command can build a path the check can't see (python -c "...", $VAR,
    cd "$(...)"). The wall is the operating system's own sandbox, used when
    one is available (os_sandbox = "auto"):

      Linux: bubblewrap (bwrap). The whole filesystem is read-only, your home
             folder is hidden, and only the workspace (and a private /tmp)
             can be written.
      macOS: sandbox-exec. Writing is only allowed in the workspace and the
             temp folders, and your home folder can't be read outside it.

    Without one (Windows, or bwrap not installed), only the text check runs.

What the sandbox doesn't cover: MCP servers (they're separate programs with
their own settings, e.g. the folders you give server-filesystem), plugins
(your own Python code), and web_fetch (it reads web pages, not files).
"""

from __future__ import annotations

import fnmatch
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from . import tools as _tools
from .tools import Tool, ToolRegistry

DEFAULT_PROTECTED = [".env", "config.toml"]
FILE_TOOLS = ("list_files", "read_file", "write_file")
SAFE_PATHS = ("/dev/null", "/dev/stdout", "/dev/stderr")  # Fine to name in a command.


class SandboxError(PermissionError):
    pass


@dataclass
class Sandbox:
    folder: str | Path = "."  # Resolved at every call, so "." follows the current folder (evals rely on that).
    protected: list[str] = field(default_factory=lambda: list(DEFAULT_PROTECTED))
    os_sandbox: str = "auto"  # "auto" or "off".
    _os_tool: str | None = field(default=None, init=False, repr=False)  # "bwrap" or "sandbox-exec" once checked.
    _os_checked: bool = field(default=False, init=False, repr=False)

    @property
    def root(self) -> Path:
        return Path(self.folder).expanduser().resolve()

    # --- paths -------------------------------------------------------------------

    def check_path(self, path: str | Path) -> Path:
        """The absolute path, if it's inside the workspace and not protected; raises SandboxError otherwise."""
        target, problem = self._inspect(path)
        if problem:
            raise SandboxError(f"{path} {problem}, so the sandbox doesn't allow it. Ask the user if you need it.")
        return target

    def _inspect(self, path: str | Path) -> tuple[Path, str | None]:
        root = self.root
        target = (root / Path(path).expanduser()).resolve()  # An absolute path replaces root here.
        if not target.is_relative_to(root):
            return target, f"is outside the workspace ({root})"
        if self.is_protected(target):
            return target, "is a protected file (it may hold secrets)"
        return target, None

    def is_protected(self, target: Path) -> bool:
        relative = target.relative_to(self.root) if target.is_relative_to(self.root) else Path(target.name)
        return any(fnmatch.fnmatch(part, pattern) for pattern in self.protected for part in relative.parts) or any(
            fnmatch.fnmatch(relative.as_posix(), pattern) for pattern in self.protected
        )

    # --- shell commands ----------------------------------------------------------

    def check_command(self, command: str) -> None:
        """Refuse a command that names a path outside the workspace or a protected file."""
        for word in _words(command):
            for candidate in _path_candidates(word):
                if candidate in SAFE_PATHS:
                    continue
                if candidate.startswith(("~", "$HOME", "${HOME}")):
                    problem = "is your home folder, outside the workspace"
                elif candidate.startswith("/") or ".." in Path(candidate).parts or self._names_protected(candidate):
                    problem = self._inspect(candidate)[1]
                else:
                    continue
                if problem:
                    raise SandboxError(
                        f"the command names {candidate}, which {problem}. run_shell only works with files inside "
                        "the workspace; ask the user if you need more."
                    )

    def _names_protected(self, word: str) -> bool:
        return any(fnmatch.fnmatch(part, pattern) for pattern in self.protected for part in Path(word).parts)

    def os_tool(self) -> str | None:
        """Which OS sandbox run_shell uses: "bwrap", "sandbox-exec" or None. Checked once, by trying it."""
        if not self._os_checked:
            self._os_checked = True
            if self.os_sandbox != "off":
                for name in ("bwrap", "sandbox-exec"):
                    if shutil.which(name) and _works(self._wrapper(name)):
                        self._os_tool = name
                        break
        return self._os_tool

    def wrapper(self) -> list[str] | None:
        tool = self.os_tool()
        return self._wrapper(tool) if tool else None

    def _wrapper(self, tool: str) -> list[str]:
        root, home = str(self.root), str(Path.home().resolve())
        if tool == "bwrap":
            hide_home = [] if Path(home).is_relative_to(root) else ["--tmpfs", home]
            return [
                "bwrap", "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp",
                *hide_home, "--bind", root, root, "--chdir", root, "--unshare-pid", "--die-with-parent",
            ]
        return ["sandbox-exec", "-p", _macos_profile(root, home)]

    def run_shell(self, command: str, timeout: int = 60) -> str:
        return _tools.run_command(command, timeout, cwd=self.root, wrapper=self.wrapper())

    # --- describing it -------------------------------------------------------------

    def describe(self) -> str:
        tool = self.os_tool()
        if tool:
            shell = f"run_shell is also sandboxed by the OS ({tool})"
        elif self.os_sandbox == "off":
            shell = "run_shell: path checks only (os_sandbox = \"off\")"
        else:
            hint = "install bubblewrap for a real one" if sys.platform.startswith("linux") else "no OS sandbox here"
            shell = f"run_shell: path checks only, {hint}"
        return f"sandbox: files and run_shell stay in {self.root}; {shell}"


def apply(registry: ToolRegistry, sandbox: Sandbox) -> ToolRegistry:
    """A copy of the registry whose file tools and run_shell stay in the sandbox."""
    registry = registry.copy()
    for name in FILE_TOOLS:
        tool = registry.get(name)
        if tool:
            registry._tools[name] = Tool(tool.spec, _confined(tool.func, sandbox), tool.confirm, _path_check(sandbox))
    shell = registry.get("run_shell")
    if shell:
        registry._tools["run_shell"] = Tool(
            shell.spec, sandbox.run_shell, shell.confirm, lambda command, **_: sandbox.check_command(command)
        )
    return registry


def enable(agent, sandbox: Sandbox) -> Sandbox:
    agent.tools = apply(agent.tools, sandbox)
    return sandbox


def from_settings(section: dict, workspace: str | None = None) -> Sandbox | None:
    """The [sandbox] table of config.toml (None when enabled = false)."""
    if not section.get("enabled", True):
        return None
    unknown = set(section) - {"enabled", "workspace", "protected", "os_sandbox"}
    if unknown:
        raise ValueError(f"Unknown [sandbox] setting(s): {', '.join(sorted(unknown))}")
    os_sandbox = section.get("os_sandbox", "auto")
    if os_sandbox not in ("auto", "off"):
        raise ValueError('[sandbox] os_sandbox must be "auto" or "off"')
    return Sandbox(workspace or section.get("workspace", "."), list(section.get("protected", DEFAULT_PROTECTED)), os_sandbox)


# --- helpers -----------------------------------------------------------------------


def _confined(func, sandbox: Sandbox):
    def run(path: str = ".", **arguments):
        return func(path=str(sandbox.check_path(path)), **arguments)

    return run


def _path_check(sandbox: Sandbox):
    def check(path: str = ".", **_):
        sandbox.check_path(path)

    return check


def _words(command: str) -> list[str]:
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        return list(lexer)
    except ValueError:  # Unbalanced quotes: fall back to plain splitting.
        return command.split()


def _path_candidates(word: str) -> list[str]:
    """The parts of a word that could be paths: the word, and what follows = (--out=/x). URLs aren't paths."""
    if "://" in word:
        return []
    return [p for p in [word, *word.split("=")[1:]] if p]


def _works(wrapper: list[str]) -> bool:
    try:
        return subprocess.run([*wrapper, "/bin/sh", "-c", "true"], capture_output=True, timeout=10).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _macos_profile(root: str, home: str) -> str:
    quote = lambda path: '"' + path.replace("\\", "\\\\").replace('"', '\\"') + '"'  # noqa: E731
    temp = os.path.realpath(os.environ.get("TMPDIR", "/tmp"))
    # Later rules win: allow everything, then forbid writes and reading files in
    # your home folder, then allow both again inside the workspace (and writes to
    # temp). Only file contents are hidden, so programs can still find the
    # workspace's path.
    return f"""(version 1)
(allow default)
(deny file-write*)
(deny file-read-data (subpath {quote(home)}))
(allow file-read* file-write* (subpath {quote(root)}))
(allow file-write* (subpath "/private/tmp") (subpath "/private/var/folders") (subpath {quote(temp)})
    (literal "/dev/null") (literal "/dev/stdout") (literal "/dev/stderr") (regex #"^/dev/tty"))
"""
