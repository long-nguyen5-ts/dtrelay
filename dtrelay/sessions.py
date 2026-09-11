"""Conversation prefix -> Claude Code session id.

The OpenAI API carries no conversation id, so identity is derived from the
message array itself. A chained hash gives longest-prefix lookup in one pass:
chain[i] covers messages[:i+1], so a later turn whose earlier messages are
unchanged hits the entry stored on the previous turn.

A branch, a regeneration, or a changed system prompt simply misses and starts a
fresh session. That is correct behaviour, not a failure.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path

log = logging.getLogger(__name__)


def _text(content) -> str:
    """OpenAI content is a string or a list of typed parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            p.get("text", "") for p in content if isinstance(p, dict)
        )
    return "" if content is None else str(content)


def canon(message: dict) -> str:
    """Stable identity of one message: role, text, and tool linkage only."""
    return json.dumps(
        {
            "role": message.get("role", ""),
            "text": _text(message.get("content")),
            "name": message.get("name", ""),
            "tool_call_id": message.get("tool_call_id", ""),
        },
        sort_keys=True,
        ensure_ascii=False,
    )


def fingerprint_chain(messages: list[dict]) -> list[str]:
    """chain[i] identifies messages[:i+1]."""
    out, acc = [], ""
    for m in messages:
        acc = hashlib.sha256((acc + canon(m)).encode("utf-8")).hexdigest()
        out.append(acc)
    return out


class SessionStore:
    def __init__(self, path: Path, ttl_hours: int):
        self.path = Path(path)
        self.ttl_seconds = max(0, int(ttl_hours)) * 3600

    def _load(self) -> dict:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)

    def lookup(self, messages: list[dict]) -> tuple[str | None, int]:
        """(session_id, prefix_len) for the longest known prefix; (None, 0) if none."""
        data = self._load()
        chain = fingerprint_chain(messages)
        for i in range(len(chain) - 1, -1, -1):
            entry = data.get(chain[i])
            if entry and not self._expired(entry):
                return entry["session_id"], i + 1
        return None, 0

    def remember(self, messages: list[dict], session_id: str) -> None:
        if not messages or not session_id:
            return
        data = self._load()
        data[fingerprint_chain(messages)[-1]] = {
            "session_id": session_id,
            "ts": time.time(),
        }
        self._save(data)

    def _expired(self, entry: dict) -> bool:
        return (time.time() - float(entry.get("ts", 0))) > self.ttl_seconds

    def reap(self) -> int:
        data = self._load()
        keep = {k: v for k, v in data.items() if not self._expired(v)}
        dropped = len(data) - len(keep)
        if dropped:
            self._save(keep)
        return dropped
