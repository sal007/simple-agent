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
    assert names == {"get_current_time", "calculator", "list_files", "read_file"}
