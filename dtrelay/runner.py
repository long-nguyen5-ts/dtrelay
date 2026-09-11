"""The only module that spawns `claude`.

Ported from ~/Documents/bots/kaide/kaide_bot.py:run_claude, which has run in
production against Slack. Two inherited details matter most: the prompt goes in
via stdin (argv would mangle a leading '-'), and a non-zero exit while resuming
triggers one retry with a fresh session, which is how a stale session id
recovers instead of wedging the conversation.
"""
from __future__ import annotations

import json
import logging
import shlex
import subprocess
import threading
from dataclasses import dataclass, field

from dtrelay.config import DISALLOWED_TOOLS, Settings

log = logging.getLogger(__name__)


@dataclass
class RunResult:
    text: str = ""
    session_id: str | None = None
    is_error: bool = False
    stats: dict = field(default_factory=dict)


def build_command(settings: Settings, session_id: str | None, system_prompt: str) -> list[str]:
    cmd = shlex.split(settings.claude_bin) + [
        "-p",
        "--output-format", "stream-json",
        "--verbose",
        "--include-partial-messages",
        "--disallowedTools", *DISALLOWED_TOOLS,
        "--append-system-prompt", system_prompt,
    ]
    if settings.model:
        cmd += ["--model", settings.model]
    if session_id:
        cmd += ["--resume", session_id]
    return cmd


def _extract_delta(event: dict) -> str:
    inner = event.get("event") or {}
    if inner.get("type") == "content_block_delta":
        return (inner.get("delta") or {}).get("text", "") or ""
    return ""


def run(
    settings: Settings,
    prompt: str,
    session_id: str | None,
    system_prompt: str,
    on_delta=None,
) -> RunResult:
    cmd = build_command(settings, session_id, system_prompt)
    log.info("claude run (session=%s)", session_id or "new")
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=settings.cwd or None,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
    except Exception as e:
        return RunResult(text=f"failed to start claude: {e}", is_error=True)

    def _feed():
        try:
            proc.stdin.write(prompt)
            proc.stdin.close()
        except Exception:
            pass

    threading.Thread(target=_feed, daemon=True).start()

    stderr_buf: list[str] = []
    threading.Thread(
        target=lambda: stderr_buf.append(proc.stderr.read() or ""), daemon=True
    ).start()

    killed = {"v": False}

    def _kill():
        killed["v"] = True
        try:
            proc.kill()
        except Exception:
            pass

    timer = threading.Timer(settings.timeout, _kill)
    timer.start()

    text, new_sid, is_err, stats = "", session_id, False, {}
    try:
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("type") == "stream_event":
                chunk = _extract_delta(ev)
                if chunk and on_delta:
                    on_delta(chunk)
            elif ev.get("type") == "result":
                text = ev.get("result") or text
                new_sid = ev.get("session_id", new_sid)
                is_err = bool(ev.get("is_error"))
                u = ev.get("usage") or {}
                stats = {
                    "duration_ms": ev.get("duration_ms"),
                    "num_turns": ev.get("num_turns"),
                    "total_cost_usd": ev.get("total_cost_usd"),
                    "input_tokens": u.get("input_tokens"),
                    "output_tokens": u.get("output_tokens"),
                }
    finally:
        timer.cancel()
    proc.wait()

    if killed["v"]:
        return RunResult(
            text=f"claude timed out after {settings.timeout}s",
            session_id=session_id, is_error=True, stats=stats,
        )
    if proc.returncode != 0 and not text:
        err = "".join(stderr_buf).strip()
        log.error("claude exited %s: %s", proc.returncode, err[:2000])
        if session_id:
            log.info("stale session; retrying without --resume")
            return run(settings, prompt, None, system_prompt, on_delta)
        return RunResult(text=f"claude error: {err[:1500]}", is_error=True, stats=stats)
    return RunResult(text=text, session_id=new_sid, is_error=is_err, stats=stats)
