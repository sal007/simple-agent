"""`simple-agent config`: add settings that are new in config.example.toml to your config.toml.

Your config.toml is your own copy (it isn't in git), so when a new version of
the agent adds settings, your copy doesn't have them. This command compares
the two files and adds what's missing, each with the comment that explains
it, without touching anything you already have:

  - a missing top-level setting goes above your first [table];
  - a missing setting inside a table you have goes at the end of that table;
  - a missing table (including commented-out examples like # [plugins.web])
    goes at the end of the file.

A setting counts as present if your file has it at all, even commented out,
so settings you deliberately commented out stay that way. New settings are
added exactly as in the example: switched on if they're on there (with the
default value, so behavior doesn't change) and commented out if they're
commented out there.

    simple-agent config            # show what would be added
    simple-agent config --update   # add it (the old file is kept as config.toml.bak)
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .config import CONFIG_LOCATIONS

EXAMPLE = Path(__file__).resolve().parent.parent / "config.example.toml"

HEADER = re.compile(r"^(#\s*)?\[([A-Za-z0-9_.\-]+)\]\s*$")  # [table] or # [table]
KEY = re.compile(r"^(#\s*)?([A-Za-z0-9_\-]+)\s*=")  # key = ... or # key = ...


@dataclass
class Section:
    """The top level of the file (name "") or one [table], as lines of text."""

    name: str
    commented: bool = False  # A "# [table]" example rather than a real table.
    lines: list[str] = field(default_factory=list)

    def keys(self) -> list[str]:
        return [m.group(2) for line in self.lines if (m := KEY.match(line.strip()))]


def parse(text: str) -> list[Section]:
    """Split a config file into its top level and its tables.

    A table's comment lines just above its header belong to the table, so
    a table is copied together with the comment that introduces it.
    """
    sections = [Section("")]
    lines = text.splitlines()
    for i, line in enumerate(lines):
        match = HEADER.match(line.strip())
        if match:
            # Move the comment paragraph right above the header into the new table.
            previous = sections[-1].lines
            start = len(previous)
            while start > 0 and previous[start - 1].strip().startswith("#") and not KEY.match(previous[start - 1].strip()):
                start -= 1
            intro, previous[start:] = previous[start:], []
            sections.append(Section(match.group(2), bool(match.group(1)), intro + [line]))
        else:
            sections[-1].lines.append(line)
    return sections


def _key_lines(section: Section, key: str) -> list[str]:
    """The line setting `key` in an example section, with the comment lines right above it."""
    for i, line in enumerate(section.lines):
        match = KEY.match(line.strip())
        if match and match.group(2) == key:
            start = i
            while start > 0 and section.lines[start - 1].strip().startswith("#") and not KEY.match(section.lines[start - 1].strip()) and not HEADER.match(section.lines[start - 1].strip()):
                start -= 1
            return section.lines[start : i + 1]
    return []


def missing(example_text: str, user_text: str) -> list[tuple[str, str, list[str]]]:
    """What the example has that the user's file doesn't: (table, key or "", lines to add)."""
    example, user = parse(example_text), parse(user_text)
    have = {s.name: s for s in user}
    additions = []
    for section in example:
        if section.name not in have:
            if section.name:
                additions.append((section.name, "", _trim(section.lines)))
            continue
        if section.commented:
            continue  # The user has this example table, commented or not; leave it be.
        present = set(have[section.name].keys())
        for key in section.keys():
            if key not in present:
                additions.append((section.name, key, _key_lines(section, key)))
    return additions


def merge(user_text: str, additions: list[tuple[str, str, list[str]]]) -> str:
    """The user's file with the additions put in their places."""
    sections = parse(user_text)
    by_name = {s.name: s for s in sections}
    new_tables = []
    previous = None  # The last setting added, so one without a comment stays next to it.
    for table, key, lines in additions:
        if not key:
            new_tables.append(lines)
            continue
        target = by_name[table].lines
        end = len(target)
        while end > 0 and not target[end - 1].strip():
            end -= 1  # Add after the last setting, before the blank lines that end the table.
        follows = previous == table and not lines[0].lstrip().startswith("#")
        target[end:end] = ([""] if end and target[end - 1].strip() and not follows else []) + lines
        previous = table
    out = "\n".join(line for s in sections for line in s.lines).rstrip("\n")
    for lines in new_tables:
        out += "\n\n" + "\n".join(lines)
    # Keep the file tidy: one blank line between paragraphs at most, a newline at the end.
    return re.sub(r"\n{3,}", "\n\n", out).strip("\n") + "\n"


def _trim(lines: list[str]) -> list[str]:
    start, end = 0, len(lines)
    while start < end and not lines[start].strip():
        start += 1
    while end > start and not lines[end - 1].strip():
        end -= 1
    return lines[start:end]


def _flatten(data: dict, prefix: str = "") -> dict:
    flat = {}
    for key, value in data.items():
        if isinstance(value, dict):
            flat.update(_flatten(value, f"{prefix}{key}."))
        else:
            flat[f"{prefix}{key}"] = value
    return flat


def update(user_text: str, example_text: str) -> tuple[str, list[str]]:
    """Return the merged file and a description of each addition.

    Raises ValueError if the result would change or lose any setting the
    user already had (a safety net: then nothing is written).
    """
    additions = missing(example_text, user_text)
    if not additions:
        return user_text, []
    merged = merge(user_text, additions)
    before, after = _flatten(tomllib.loads(user_text)), _flatten(tomllib.loads(merged))
    changed = [k for k, v in before.items() if after.get(k, object()) != v]
    if changed:
        raise ValueError(f"merging would change {changed}; please add the new settings by hand")
    described = [f"[{t}]" if not k else (k if not t else f"{k} in [{t}]") for t, k, _ in additions]
    return merged, described


def find_config(path: str | None) -> Path:
    if path:
        return Path(path)
    for candidate in CONFIG_LOCATIONS:
        if candidate.is_file():
            return candidate
    return CONFIG_LOCATIONS[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="simple-agent config", description=__doc__.split("\n\n")[0])
    parser.add_argument("--update", action="store_true", help="Add the missing settings (keeps a .bak copy).")
    parser.add_argument("--config", help="Which config file (default: ./config.toml, then ~/.config/simple-agent/config.toml).")
    parser.add_argument("--example", default=str(EXAMPLE), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    example_path = Path(args.example)
    if not example_path.is_file():
        print(f"Can't find {example_path} (it ships with the source checkout).", file=sys.stderr)
        return 1
    example_text = example_path.read_text(encoding="utf-8")
    path = find_config(args.config)

    if not path.is_file():
        if not args.update:
            print(f"There is no {path} yet. Run `simple-agent config --update` to create it from the example.")
            return 0
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(example_path, path)
        print(f"Created {path} from {example_path.name}. Edit it to suit you.")
        return 0

    try:
        merged, added = update(path.read_text(encoding="utf-8"), example_text)
    except (ValueError, tomllib.TOMLDecodeError) as exc:
        print(f"Couldn't update {path}: {exc}", file=sys.stderr)
        return 1
    if not added:
        print(f"{path} already has every setting in {example_path.name}.")
        return 0
    if not args.update:
        print(f"{path} is missing these settings from {example_path.name}:")
        for item in added:
            print(f"  {item}")
        print("Run `simple-agent config --update` to add them, with their comments and default values.")
        return 0
    backup = path.with_name(path.name + ".bak")
    shutil.copyfile(path, backup)
    path.write_text(merged, encoding="utf-8")
    print(f"Added to {path} (your old file is {backup}):")
    for item in added:
        print(f"  {item}")
    return 0


def hint(path: str | None) -> str | None:
    """A one-line note for the chat's start-up if the config file is missing settings."""
    config = find_config(path)
    if not config.is_file() or not EXAMPLE.is_file():
        return None
    try:
        _, added = update(config.read_text(encoding="utf-8"), EXAMPLE.read_text(encoding="utf-8"))
    except (ValueError, tomllib.TOMLDecodeError, OSError):
        return None
    if not added:
        return None
    return f"({config} is missing {len(added)} new setting(s); `simple-agent config` lists them, `--update` adds them)"
