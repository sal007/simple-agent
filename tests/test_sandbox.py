"""Tests for the sandbox (sandbox.py): file tools and run_shell stay in the workspace."""

import os
import shutil

import pytest

from simple_agent import sandbox
from simple_agent.agent import Agent
from simple_agent.cli import main
from simple_agent.config import resolve
from simple_agent.providers.base import Message, Reply, ToolCall
from simple_agent.sandbox import Sandbox, SandboxError
from simple_agent.subagents import enable as enable_subagents
from simple_agent.tools import default_tools


@pytest.fixture
def box(tmp_path):
    """A workspace with a file, a protected file and a file next to it (outside)."""
    work = tmp_path / "work"
    (work / "src").mkdir(parents=True)
    (work / "notes.txt").write_text("inside")
    (work / ".env").write_text("API_KEY=secret")
    (work / "src" / "config.toml").write_text("api_key = 'secret'")
    (tmp_path / "outside.txt").write_text("outside")
    return Sandbox(work, os_sandbox="off")


def tools_for(box, asked=None):
    registry = sandbox.apply(default_tools, box)
    registry.approve = lambda name, arguments: asked.append(name) or True if asked is not None else True
    return registry


def test_paths_inside_are_allowed_and_outside_refused(box, tmp_path):
    work = box.root
    assert box.check_path("notes.txt") == work / "notes.txt"
    assert box.check_path(work / "src" / "new.py") == work / "src" / "new.py"  # Doesn't have to exist yet.
    assert box.check_path("src/../notes.txt") == work / "notes.txt"
    for path in ("../outside.txt", str(tmp_path / "outside.txt"), "/etc/passwd", "~"):
        with pytest.raises(SandboxError, match="outside the workspace"):
            box.check_path(path)


def test_symlinks_out_of_the_workspace_are_refused(box, tmp_path):
    (box.root / "sneaky").symlink_to(tmp_path / "outside.txt")
    with pytest.raises(SandboxError, match="outside the workspace"):
        box.check_path("sneaky")


def test_protected_files(box):
    for path in (".env", "src/config.toml"):
        with pytest.raises(SandboxError, match="protected"):
            box.check_path(path)
    assert Sandbox(box.root, protected=[]).check_path(".env")


def test_the_file_tools_use_the_workspace_not_the_current_folder(box, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # Somewhere else.
    tools = tools_for(box)
    assert tools.run("read_file", {"path": "notes.txt"}) == "inside"
    assert "notes.txt" in tools.run("list_files", {}) and "outside.txt" not in tools.run("list_files", {})
    assert tools.run("write_file", {"path": "out/a.txt", "content": "hi"}).startswith("Wrote")
    assert (box.root / "out" / "a.txt").read_text() == "hi"
    assert "outside the workspace" in tools.run("read_file", {"path": "../outside.txt"})
    assert "protected" in tools.run("read_file", {"path": ".env"})


def test_refused_calls_are_never_put_to_the_user(box):
    asked = []
    tools = tools_for(box, asked)
    result = tools.run("write_file", {"path": "/tmp/evil.sh", "content": "x"})
    assert "outside the workspace" in result and asked == []
    assert "names /etc/passwd" in tools.run("run_shell", {"command": "cat /etc/passwd"}) and asked == []
    tools.run("write_file", {"path": "fine.txt", "content": "x"})
    assert asked == ["write_file"]  # Allowed calls still ask y/N.
    assert tools.asks_first("write_file") and tools.asks_first("run_shell")


@pytest.mark.parametrize("command", [
    "cat /etc/passwd", "ls ..", "cat ../outside.txt", "cat ~/.ssh/id_rsa", "echo $HOME/x", "cat .env",
    "cp notes.txt --target-directory=/tmp", "grep -r key src/config.toml", "echo hi>/tmp/x",
])
def test_commands_naming_paths_outside_are_refused(box, command):
    with pytest.raises(SandboxError):
        box.check_command(command)


@pytest.mark.parametrize("command", [
    "ls -la", "cat notes.txt", "python3 -c 'print(1+1)' > /dev/null", "ls src/../notes.txt",
    "curl -s https://example.com/a/b", "git status", "echo 'unbalanced",
])
def test_ordinary_commands_are_allowed(box, command):
    box.check_command(command)


def test_shell_runs_in_the_workspace(box, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = tools_for(box).run("run_shell", {"command": "pwd && cat notes.txt"})
    assert f"stdout:\n{box.root}\ninside" in result


@pytest.mark.skipif(not (shutil.which("bwrap") and sandbox._works(["bwrap", "--ro-bind", "/", "/"])),
                    reason="bubblewrap isn't available here")
def test_the_os_sandbox_stops_what_the_text_check_cannot_see(box, tmp_path):
    box.os_sandbox = "auto"
    assert box.os_tool() == "bwrap" and "bwrap" in box.describe()
    # The path is built at run time, so the text check can't see it; the OS sandbox stops it.
    # (/tmp is a private, throwaway folder inside the sandbox, so that write lands there.)
    command = 'D=$(dirname "$PWD"); echo changed > "$D/outside.txt"; B=$(dirname "$(command -v sh)"); touch "$B/x"; echo done > made.txt'
    box.check_command(command)
    result = tools_for(box).run("run_shell", {"command": command})
    assert "Read-only file system" in result or "Permission denied" in result  # The touch in /usr/bin (or /bin).
    assert (tmp_path / "outside.txt").read_text() == "outside"
    assert (box.root / "made.txt").read_text() == "done\n"  # The workspace itself is writable.


def test_os_sandbox_off_and_the_macos_profile(box):
    assert box.os_tool() is None and box.wrapper() is None and "path checks only" in box.describe()
    profile = sandbox._macos_profile('/Users/me/my "work"', "/Users/me")
    assert '(subpath "/Users/me/my \\"work\\"")' in profile and "(deny file-write*)" in profile


def test_sub_agents_share_the_sandbox(box):
    class Provider:
        name, model = "fake", "fake"

        def __init__(self):
            self.replies = [
                Reply(Message("assistant", "", [ToolCall("d1", "delegate", {"task": "read it"})]), {}),
                Reply(Message("assistant", "", [ToolCall("r1", "read_file", {"path": "../outside.txt"})]), {}),
                Reply(Message("assistant", "refused"), {}),
                Reply(Message("assistant", "done"), {}),
            ]
            self.seen = []

        def chat(self, system, messages, tools, on_text=None):
            self.seen.append(messages[-1].content)
            return self.replies.pop(0)

    provider = Provider()
    agent = Agent(provider=provider, stream=False)
    sandbox.enable(agent, box)
    enable_subagents(agent)
    assert agent.ask("go") == "done"
    assert "outside the workspace" in provider.seen[2]  # What the helper's read_file returned.


def test_settings(tmp_path):
    assert resolve({}).sandbox.folder == "." and resolve({}).sandbox.os_sandbox == "auto"
    assert resolve({"sandbox": {"enabled": False}}).sandbox is None
    custom = resolve({"sandbox": {"workspace": "~/agent", "protected": ["*.key"], "os_sandbox": "off"}}).sandbox
    assert (custom.folder, custom.protected, custom.os_sandbox) == ("~/agent", ["*.key"], "off")
    with pytest.raises(ValueError, match="Unknown"):
        resolve({"sandbox": {"folder": "x"}})
    with pytest.raises(ValueError, match="os_sandbox"):
        resolve({"sandbox": {"os_sandbox": "maybe"}})


def test_cli(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "work").mkdir()
    lines = iter(["/exit"] * 4)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(lines))

    assert main(["--no-plugins", "--no-usage"]) == 0
    assert f"(sandbox: files and run_shell stay in {tmp_path.resolve()};" in capsys.readouterr().out

    assert main(["--no-plugins", "--workspace", "work"]) == 0
    assert f"stay in {(tmp_path / 'work').resolve()};" in capsys.readouterr().out

    assert main(["--no-plugins", "--no-sandbox"]) == 0
    assert "sandbox off" in capsys.readouterr().out

    assert main(["--no-plugins", "--workspace", "missing"]) == 1
    assert "is not a folder" in capsys.readouterr().err
    os.chdir(tmp_path)


def test_self_test_reports_when_the_sandbox_leaks(box, monkeypatch):
    lines = []
    assert sandbox.self_test(box, lines.append) is False and "nothing to test" in lines[-1]

    # Pretend `env` is an OS sandbox: it runs the command unchanged, so writes get out.
    monkeypatch.setattr(Sandbox, "os_tool", lambda self: "fake")
    monkeypatch.setattr(Sandbox, "wrapper", lambda self: ["env"])
    monkeypatch.setattr(sandbox.Path, "home", classmethod(lambda cls: box.root.parent / "home"))
    (box.root.parent / "home").mkdir()
    lines.clear()
    assert sandbox.self_test(box, lines.append) is False
    assert any(line.startswith("  ok   write a file in the workspace") for line in lines)
    assert any(line.startswith("  FAIL block writing") for line in lines)
    assert any(line.startswith("  FAIL block reading") for line in lines)
    assert "did NOT hold" in lines[-1]
    assert not list(box.root.parent.glob("**/.simple-agent-sandbox-test*"))  # Cleaned up.


@pytest.mark.skipif(not (shutil.which("bwrap") and sandbox._works(["bwrap", "--ro-bind", "/", "/"])),
                    reason="bubblewrap isn't available here")
def test_self_test_passes_with_bubblewrap(box, capsys):
    box.os_sandbox = "auto"
    assert main(["sandbox", "--workspace", str(box.root)]) == 0
    assert "The sandbox holds." in capsys.readouterr().out
