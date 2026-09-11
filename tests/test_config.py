from pathlib import Path
from dtrelay.config import load_settings


def test_defaults_when_env_empty():
    s = load_settings({})
    assert s.port == 8787
    assert s.claude_bin == "claude"
    assert s.model == ""
    assert s.timeout == 900
    assert s.max_concurrent == 2
    assert s.session_ttl_hours == 72


def test_env_overrides_are_typed():
    s = load_settings({
        "DTRELAY_PORT": "9000",
        "DTRELAY_MAX_CONCURRENT": "5",
        "DTRELAY_TIMEOUT": "60",
        "DTRELAY_STATE_DIR": "/tmp/dtstate",
    })
    assert s.port == 9000
    assert s.max_concurrent == 5
    assert s.timeout == 60
    assert s.state_dir == Path("/tmp/dtstate")


def test_max_concurrent_is_floored_at_one():
    assert load_settings({"DTRELAY_MAX_CONCURRENT": "0"}).max_concurrent == 1
