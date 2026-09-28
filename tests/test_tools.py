import pytest

from simple_agent.tools import calculator, default_tools


def test_calculator():
    assert calculator("(2 + 3) * 4") == "20"
    assert calculator("-2 ** 2") == "-4"


@pytest.mark.parametrize("expr", ["__import__('os')", "open('x')", "2 ** 100000"])
def test_calculator_rejects_unsafe_input(expr):
    assert default_tools.run("calculator", {"expression": expr}).startswith("Error")


def test_errors_are_returned_not_raised(tmp_path):
    assert default_tools.run("nope", {}).startswith("Error: unknown tool")
    assert default_tools.run("read_file", {"path": str(tmp_path / "missing.txt")}).startswith("Error")
    assert default_tools.run("calculator", {"__invalid_json__": "{bad"}).startswith("Error: arguments")


def test_read_and_list_files(tmp_path):
    (tmp_path / "a.txt").write_text("hello world")
    (tmp_path / "sub").mkdir()
    assert default_tools.run("list_files", {"path": str(tmp_path)}) == "a.txt\nsub/"
    assert default_tools.run("read_file", {"path": str(tmp_path / "a.txt"), "max_chars": 5}).startswith("hello\n...")


def test_specs_describe_every_tool():
    names = {s.name for s in default_tools.specs()}
    assert names == {"get_current_time", "calculator", "list_files", "read_file", "write_file", "run_shell"}
    assert default_tools.asks_first("run_shell") and not default_tools.asks_first("read_file")


def test_confirm_tools_are_declined_without_an_approver(tmp_path, monkeypatch):
    monkeypatch.setattr(default_tools, "approve", None)
    target = tmp_path / "out.txt"
    result = default_tools.run("write_file", {"path": str(target), "content": "hi"})
    assert result.startswith("The user declined")
    assert not target.exists()


def test_approver_sees_the_call_and_decides(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(default_tools, "approve", lambda name, args: seen.append((name, args)) or True)
    target = tmp_path / "sub" / "out.txt"

    assert default_tools.run("write_file", {"path": str(target), "content": "hello"}).startswith("Wrote 5")
    assert target.read_text() == "hello"
    assert seen == [("write_file", {"path": str(target), "content": "hello"})]

    result = default_tools.run("run_shell", {"command": "echo hi && echo oops >&2 && exit 3"})
    assert "exit code: 3" in result and "hi" in result and "oops" in result

    # Read-only tools never ask.
    default_tools.run("calculator", {"expression": "1+1"})
    assert len(seen) == 2


def test_declined_calls_do_not_run(tmp_path, monkeypatch):
    monkeypatch.setattr(default_tools, "approve", lambda name, args: False)
    marker = tmp_path / "marker"
    result = default_tools.run("run_shell", {"command": f"touch {marker}"})
    assert result.startswith("The user declined") and not marker.exists()


def test_run_shell_timeout(monkeypatch):
    monkeypatch.setattr(default_tools, "approve", lambda name, args: True)
    assert "did not finish within 1 seconds" in default_tools.run("run_shell", {"command": "sleep 5", "timeout": 1})
