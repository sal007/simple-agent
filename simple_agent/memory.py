"""Long-term memory: notes the agent keeps between conversations.

Each conversation starts from nothing; memory is how the agent carries what
it learned (your preferences, facts about your projects, decisions) into the
next one. It's deliberately simple, so you can see and edit everything:

  - Each memory is a Markdown file in the memory/ folder (memory_dir in
    config.toml), with a title and a one-line description at the top:

        ---
        title: Preferred units
        description: Metric units in every answer
        updated: 2026-10-04
        ---
        Use metric units (km, kg, °C). Convert imperial values the user pastes.

  - The model gets three tools: save_memory, read_memory and delete_memory.
  - At every step, the system prompt lists each note's title and description
    (an Agent extension, like the plan in planning.py), not the full text.
    The model reads a full note with read_memory when it's relevant, so the
    prompt stays small even with many notes.

Saving doesn't ask first, but every save prints a line in the chat so you see
what was remembered; deleting asks y/N. Sub-agents can read notes but not
change them. Turn it off with memory = false in config.toml or --no-memory.

There is no search beyond that index (no embeddings or vector database): the
model picks notes by their descriptions, which works well up to a few hundred
notes.
"""

from __future__ import annotations

import datetime as _dt
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .providers.base import ToolSpec

if TYPE_CHECKING:
    from .agent import Agent

MAX_INDEX_NOTES = 200  # Notes listed in the system prompt; the rest are still readable by title.
MAX_NOTE_CHARS = 10_000
WRITE_TOOLS = ("save_memory", "delete_memory")  # Left out for read-only agents (sub-agents).

SAVE_DESCRIPTION = (
    "Save a note to your long-term memory, which you'll see in future conversations. Use it for things "
    "worth remembering beyond this conversation: the user's preferences, facts about their projects or "
    "setup, decisions, lessons from mistakes. Not for things only this conversation needs. Saving with "
    "an existing title replaces that note, so to update a note, read it, then save the whole new text."
)
READ_DESCRIPTION = "Read the full text of a note in your long-term memory, by its title."
DELETE_DESCRIPTION = "Delete a note from your long-term memory, e.g. when it's wrong or out of date. The user is asked first."


@dataclass
class Note:
    title: str
    description: str
    text: str
    updated: str = ""


def slug(title: str) -> str:
    """The file name for a title: 'Preferred units' -> 'preferred-units'."""
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:80] or "note"


def parse(content: str, fallback_title: str) -> Note:
    """Read a note file. A file without the --- header (written by hand) still works."""
    match = re.match(r"---\n(.*?)\n---\n?(.*)", content, re.DOTALL)
    if not match:
        return Note(fallback_title, "", content.strip())
    fields = dict(re.findall(r"^(\w+):\s*(.*)$", match.group(1), re.MULTILINE))
    return Note(fields.get("title", fallback_title), fields.get("description", ""), match.group(2).strip(), fields.get("updated", ""))


def render(note: Note) -> str:
    return f"---\ntitle: {note.title}\ndescription: {note.description}\nupdated: {note.updated}\n---\n{note.text}\n"


class Memory:
    """The notes in one folder, and the Agent extension that lists them in the system prompt."""

    def __init__(self, folder: str | Path = "memory", on_change: Callable[[str, str, Path], None] | None = None):
        self.folder = Path(folder).expanduser()
        # Called with ("saved", "updated" or "deleted", title, file) after each change, e.g. to print a line.
        self.on_change = on_change

    # --- the notes ------------------------------------------------------------

    def notes(self) -> list[Note]:
        if not self.folder.is_dir():
            return []
        files = sorted(self.folder.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)  # Newest first.
        return [parse(p.read_text(encoding="utf-8", errors="replace"), p.stem) for p in files]

    def path(self, title: str) -> Path:
        return self.folder / f"{slug(title)}.md"

    def find(self, title: str) -> Path | None:
        """The file for a title: by file name, or by the title inside a hand-written file."""
        path = self.path(title)
        if path.is_file():
            return path
        if self.folder.is_dir():
            for p in self.folder.glob("*.md"):
                if parse(p.read_text(encoding="utf-8", errors="replace"), p.stem).title.lower() == title.lower():
                    return p
        return None

    def save(self, title: str, description: str, text: str) -> str:
        title, description = " ".join(title.split()), " ".join(description.split())
        if not title or not text.strip():
            return "Error: a note needs a title and some text."
        if len(text) > MAX_NOTE_CHARS:
            return f"Error: notes are limited to {MAX_NOTE_CHARS} characters; save the essentials, or split it into several notes."
        path = self.find(title) or self.path(title)
        existed = path.exists()
        self.folder.mkdir(parents=True, exist_ok=True)
        path.write_text(render(Note(title, description, text.strip(), _dt.date.today().isoformat())), encoding="utf-8")
        if self.on_change:
            self.on_change("updated" if existed else "saved", title, path)
        return f"{'Updated' if existed else 'Saved'} the note {title!r}."

    def read(self, title: str) -> str:
        path = self.find(title)
        if not path:
            titles = ", ".join(repr(n.title) for n in self.notes()[:20]) or "none yet"
            return f"Error: there is no note titled {title!r}. Notes: {titles}."
        note = parse(path.read_text(encoding="utf-8", errors="replace"), path.stem)
        return f"# {note.title}\n({note.description}; last updated {note.updated or 'unknown'})\n\n{note.text}"

    def delete(self, title: str) -> str:
        path = self.find(title)
        if not path:
            return f"Error: there is no note titled {title!r}."
        path.unlink()
        if self.on_change:
            self.on_change("deleted", title, path)
        return f"Deleted the note {title!r}."

    # --- the Agent extension ----------------------------------------------------

    def system_note(self) -> str:
        notes = self.notes()
        if not notes:
            return ("Long-term memory: you have no saved notes yet. Use save_memory for anything worth "
                    "remembering in future conversations.")
        lines = [f"- {n.title}: {n.description or '(no description)'}" for n in notes[:MAX_INDEX_NOTES]]
        if len(notes) > MAX_INDEX_NOTES:
            lines.append(f"- ... and {len(notes) - MAX_INDEX_NOTES} older notes (read_memory works for any title)")
        return (
            "Long-term memory: notes you saved in earlier conversations (title: description). They may be out "
            "of date. Read a note with read_memory when it's relevant to the request; keep notes current with "
            "save_memory and delete_memory.\n" + "\n".join(lines)
        )

    def reset(self) -> None:
        pass  # Memory outlives the conversation; that's the point.


def enable(agent: Agent, memory: Memory, read_only: bool = False) -> Memory:
    """Give an agent the memory tools and the index in its system prompt."""
    agent.tools = agent.tools.copy()  # Don't add to a registry other agents share.
    title = {"title": {"type": "string", "description": "The note's title."}}
    agent.tools.add(ToolSpec("read_memory", READ_DESCRIPTION, _schema(title)), memory.read)
    if not read_only:
        save = {
            **title,
            "description": {"type": "string", "description": "One line saying what the note is about, shown in the index."},
            "text": {"type": "string", "description": "The note itself."},
        }
        agent.tools.add(ToolSpec(WRITE_TOOLS[0], SAVE_DESCRIPTION, _schema(save)), memory.save)
        agent.tools.add(ToolSpec(WRITE_TOOLS[1], DELETE_DESCRIPTION, _schema(title)), memory.delete, confirm=True)
    agent.extensions.append(memory)
    return memory


def memory_of(agent: Agent) -> Memory | None:
    return next((e for e in agent.extensions if isinstance(e, Memory)), None)


def _schema(properties: dict) -> dict:
    return {"type": "object", "properties": properties, "required": list(properties)}
