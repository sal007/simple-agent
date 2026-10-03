"""Project instructions: rules and context for one project, read from AGENTS.md.

Put an AGENTS.md file in a project folder and the agent follows it whenever
you start it there, without you retyping it: how to run the tests, which
folders to leave alone, the style to write in, what the project is about.
AGENTS.md is a shared convention (https://agents.md) that other coding agents
read too, so one file serves them all.

Which files are read, in this order:

  1. ~/.config/simple-agent/AGENTS.md: your personal rules for every project.
  2. AGENTS.md in the top folder of the git repository you're in, then in
     each folder below it down to the one you started in. Outside a git
     repository, only the folder you started in is checked.

Later files come after earlier ones in the prompt, so the most specific file
(the one closest to where you are) has the last word.

The text is added to the end of the system prompt (as an Agent extension,
like the plan in planning.py), so the model sees it at every step, even after
context management has trimmed the conversation. Files are re-read when they
change, so edits take effect from the next message without a restart.

Turn it off with project_instructions = false in config.toml or
--no-instructions. /instructions shows what was loaded.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .agent import Agent

FILE_NAME = "AGENTS.md"
USER_FILE = Path.home() / ".config" / "simple-agent" / FILE_NAME
MAX_CHARS = 20_000  # Per file, so a huge file can't crowd out the conversation.

HEADER = (
    "Project instructions from {path}. Follow them unless the user asks otherwise; "
    "they override the general instructions above where the two disagree:"
)


def find_files(start: Path | None = None, user_file: Path | None | bool = True) -> list[Path]:
    """The AGENTS.md files that apply in `start` (default: the current folder), outermost first.

    user_file: True for ~/.config/simple-agent/AGENTS.md, a path, or None for no personal file.
    """
    user_file = USER_FILE if user_file is True else user_file
    start = (start or Path.cwd()).resolve()
    folders = []
    for folder in [start, *start.parents]:  # Walk up to the repository's top folder.
        folders.append(folder)
        if (folder / ".git").exists():
            break
    else:
        folders = [start]  # Not in a git repository: just the start folder.
    files = [user_file] if user_file else []
    files += [folder / FILE_NAME for folder in reversed(folders)]
    return [f for f in files if f.is_file()]


class ProjectInstructions:
    """An Agent extension that adds the AGENTS.md files to the system prompt."""

    def __init__(self, start: Path | None = None, user_file: Path | None | bool = True):
        self.start = start  # None: wherever the agent is when it asks (the eval runner moves around).
        self.user_file = user_file
        self._cache: dict[Path, tuple[float, str]] = {}  # path -> (modified time, text)

    def files(self) -> list[Path]:
        return find_files(self.start, self.user_file)

    def read(self, path: Path) -> str:
        """The file's text, read again only if it changed since last time."""
        mtime = path.stat().st_mtime
        cached = self._cache.get(path)
        if cached and cached[0] == mtime:
            return cached[1]
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if len(text) > MAX_CHARS:
            text = text[:MAX_CHARS] + f"\n[... cut off: {_display(path)} is longer than {MAX_CHARS} characters]"
        self._cache[path] = (mtime, text)
        return text

    def system_note(self) -> str:
        parts = []
        for path in self.files():
            text = self.read(path)
            if text:
                parts.append(f"{HEADER.format(path=_display(path))}\n\n{text}")
        return "\n\n".join(parts)

    def reset(self) -> None:
        pass  # The files belong to the project, not the conversation.

    def describe(self) -> list[str]:
        """One line per file, for the CLI."""
        return [f"{_display(p)} ({len(self.read(p)):,} characters)" for p in self.files()]


def enable(agent: Agent, instructions: ProjectInstructions | None = None) -> ProjectInstructions:
    """Add project instructions to an agent, ahead of other extensions such as the plan."""
    instructions = instructions or ProjectInstructions()
    agent.extensions.insert(0, instructions)
    return instructions


def instructions_of(agent: Agent) -> ProjectInstructions | None:
    return next((e for e in agent.extensions if isinstance(e, ProjectInstructions)), None)


def _display(path: Path) -> str:
    """A short name for a file: relative to the current folder, or with ~ for home."""
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        pass
    try:
        return "~/" + str(path.relative_to(Path.home()))
    except ValueError:
        return str(path)
