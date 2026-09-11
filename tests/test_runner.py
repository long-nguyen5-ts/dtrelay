import sys
from pathlib import Path

from dtrelay.config import Settings
from dtrelay.runner import build_command, run

FAKE = str(Path(__file__).parent / "fake_claude.py")


def settings(**kw):
    return Settings(claude_bin=f"{sys.executable} {FAKE}", timeout=30, **kw)


def test_command_has_stream_json_and_disallowed_tools():
    cmd = build_command(settings(), session_id=None, system_prompt="sys")
    assert "--output-format" in cmd and "stream-json" in cmd
    assert "--disallowedTools" in cmd
    assert "Bash" in cmd and "WebSearch" in cmd
    assert "--resume" not in cmd


def test_command_resumes_when_a_session_is_given():
    cmd = build_command(settings(), session_id="sid_A", system_prompt="sys")
    assert cmd[cmd.index("--resume") + 1] == "sid_A"


def test_command_includes_model_only_when_configured():
    assert "--model" not in build_command(settings(), None, "sys")
    cmd = build_command(settings(model="opus"), None, "sys")
    assert cmd[cmd.index("--model") + 1] == "opus"


def test_run_returns_result_text_and_session(monkeypatch):
    monkeypatch.setenv("FAKE_RESULT", "hello there")
    monkeypatch.setenv("FAKE_SESSION", "sid_X")
    r = run(settings(), "hi", None, "sys")
    assert r.text == "hello there"
    assert r.session_id == "sid_X"
    assert r.is_error is False


def test_prompt_is_fed_via_stdin_not_argv(monkeypatch, tmp_path):
    stdin_file = tmp_path / "stdin.txt"
    argv_file = tmp_path / "argv.txt"
    monkeypatch.setenv("FAKE_ECHO_STDIN", str(stdin_file))
    monkeypatch.setenv("FAKE_ECHO_ARGV", str(argv_file))
    run(settings(), "--not-a-flag", None, "sys")
    assert stdin_file.read_text() == "--not-a-flag"
    assert "--not-a-flag" not in argv_file.read_text().split("\n")


def test_deltas_are_forwarded_live(monkeypatch):
    monkeypatch.setenv("FAKE_DELTAS", "Hel|lo")
    seen = []
    run(settings(), "hi", None, "sys", on_delta=seen.append)
    assert "".join(seen) == "Hello"


def test_nonzero_exit_while_resuming_retries_fresh(monkeypatch, tmp_path):
    argv_file = tmp_path / "argv.txt"
    monkeypatch.setenv("FAKE_ECHO_ARGV", str(argv_file))
    monkeypatch.setenv("FAKE_EXIT", "1")
    r = run(settings(), "hi", "stale_sid", "sys")
    assert "--resume" not in argv_file.read_text().split("\n")
    assert r.is_error is True


def test_nonzero_exit_without_session_reports_stderr(monkeypatch):
    monkeypatch.setenv("FAKE_EXIT", "1")
    r = run(settings(), "hi", None, "sys")
    assert r.is_error is True
    assert "fake claude failure" in r.text


def test_timeout_is_reported():
    s = Settings(claude_bin=f"{sys.executable} -c 'import time;time.sleep(5)'", timeout=1)
    r = run(s, "hi", None, "sys")
    assert r.is_error is True
    assert "timed out" in r.text.lower()
