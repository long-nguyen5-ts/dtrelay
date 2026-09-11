"""Env-backed settings. Every knob has a default that works on a fresh machine."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent

DISALLOWED_TOOLS = [
    "Bash", "Read", "Write", "Edit", "Glob", "Grep",
    "WebSearch", "WebFetch", "Task", "TodoWrite",
]
MODEL_ID = "claude-code"


@dataclass(frozen=True)
class Settings:
    port: int = 8787
    claude_bin: str = "claude"
    model: str = ""
    timeout: int = 900
    max_concurrent: int = 2
    cwd: str = ""
    state_dir: Path = _ROOT / "state"
    session_ttl_hours: int = 72


def _int(env, key, default):
    try:
        return int(env.get(key) or default)
    except (TypeError, ValueError):
        return default


def load_settings(env: dict | None = None) -> Settings:
    env = os.environ if env is None else env
    state = env.get("DTRELAY_STATE_DIR")
    return Settings(
        port=_int(env, "DTRELAY_PORT", 8787),
        claude_bin=env.get("DTRELAY_CLAUDE_BIN") or "claude",
        model=env.get("DTRELAY_MODEL") or "",
        timeout=_int(env, "DTRELAY_TIMEOUT", 900),
        max_concurrent=max(1, _int(env, "DTRELAY_MAX_CONCURRENT", 2)),
        cwd=env.get("DTRELAY_CWD") or "",
        state_dir=Path(state) if state else _ROOT / "state",
        session_ttl_hours=_int(env, "DTRELAY_SESSION_TTL_HOURS", 72),
    )
