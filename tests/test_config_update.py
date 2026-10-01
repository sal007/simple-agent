"""Tests for `simple-agent config`, which adds new settings to an existing config.toml."""

import subprocess
import tomllib
from pathlib import Path

import pytest

from simple_agent import config_update
from simple_agent.cli import main

REPO = Path(__file__).parent.parent
EXAMPLE = (REPO / "config.example.toml").read_text()

OLD = """# My settings
provider = "anthropic"
max_steps = 25
# trace = true

[providers.openai]
base_url = "http://localhost:1234/v1"
model = "my-own-model"

[context]
max_tokens = 4000

[mcp_servers.files]
command = "npx"
args = ["-y", "@modelcontextprotocol/server-filesystem", "."]
"""


def test_adds_only_what_is_missing_and_keeps_every_value():
    merged, added = config_update.update(OLD, EXAMPLE)
    old, new = tomllib.loads(OLD), tomllib.loads(merged)

    assert new["provider"] == "anthropic" and new["max_steps"] == 25
    assert new["providers"]["openai"]["model"] == "my-own-model"
    assert new["context"]["max_tokens"] == 4000 and new["context"]["compact"] is True
    assert new["mcp_servers"] == old["mcp_servers"]
    assert new["subagents"] is True and new["planning"] is True  # Top level, not inside a table.
    assert new["providers"]["anthropic"]["model"] == "claude-opus-5-5"

    assert "trace" not in added  # Commented out on purpose; stays that way.
    assert "# [plugins.web]" in merged and "plugins" not in new  # Added, commented out like the example.
    assert "[mcp_servers.files]" not in added  # The user already has that table.
    assert {"stream", "subagents", "compact in [context]", "[providers.anthropic]", "[plugins.web]"} <= set(added)
    assert merged.startswith("# My settings\nprovider = \"anthropic\"")


def test_running_it_twice_adds_nothing_more():
    merged, _ = config_update.update(OLD, EXAMPLE)
    assert config_update.update(merged, EXAMPLE) == (merged, [])
    assert config_update.update(EXAMPLE, EXAMPLE) == (EXAMPLE, [])


@pytest.mark.parametrize("commit", ["228a81c", "da8e7a7", "9afefb4"])
def test_every_older_example_catches_up(commit):
    try:
        old = subprocess.run(["git", "show", f"{commit}:config.example.toml"], cwd=REPO, capture_output=True,
                             text=True, check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("git history not available")
    merged, added = config_update.update(old, EXAMPLE)
    assert added and config_update.missing(EXAMPLE, merged) == []
    assert tomllib.loads(merged) == tomllib.loads(EXAMPLE)


def test_the_command(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["config"]) == 0
    assert "no config.toml yet" in capsys.readouterr().out

    Path("config.toml").write_text(OLD)
    assert main(["config"]) == 0
    assert "missing these settings" in capsys.readouterr().out
    assert Path("config.toml").read_text() == OLD  # Only shown, not changed.

    assert main(["config", "--update"]) == 0
    assert "Added to config.toml" in capsys.readouterr().out
    assert Path("config.toml.bak").read_text() == OLD
    assert "subagents = true" in Path("config.toml").read_text()
    assert config_update.hint(None) is None

    assert main(["config", "--update"]) == 0
    assert "already has every setting" in capsys.readouterr().out


def test_creates_the_file_if_there_is_none(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(["config", "--update"]) == 0
    assert Path("config.toml").read_text() == EXAMPLE


def test_hint_for_an_old_file(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(OLD)
    assert "missing" in config_update.hint(str(path))


def test_refuses_rather_than_changing_a_setting():
    # A key that the example has at the top level, but the user's file sets inside a
    # multi-line string: a naive merge could break it, so the check refuses.
    tricky = 'system_prompt = """\n[context]\nmax_tokens = 1\n"""\n'
    try:
        merged, _ = config_update.update(tricky, EXAMPLE)
    except ValueError:
        return
    assert tomllib.loads(merged)["system_prompt"] == tomllib.loads(tricky)["system_prompt"]
