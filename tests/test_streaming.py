import json

from fastapi.testclient import TestClient

from dtrelay.config import Settings
from dtrelay.runner import RunResult
from dtrelay.server import create_app
from tests.test_server import TOOLS, FakeRunner


def sse_events(text):
    out = []
    for line in text.splitlines():
        if line.startswith("data: "):
            payload = line[6:].strip()
            if payload != "[DONE]":
                out.append(json.loads(payload))
    return out


def test_prose_streams_content_deltas(tmp_path):
    runner = FakeRunner(RunResult(text="Hello there", session_id="sid_A"))
    c = TestClient(create_app(Settings(state_dir=tmp_path), runner=runner))
    r = c.post("/v1/chat/completions", json={
        "model": "claude-code",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    })
    events = sse_events(r.text)
    joined = "".join(
        e["choices"][0]["delta"].get("content", "") or "" for e in events
    )
    assert joined == "Hello there"
    assert events[-1]["choices"][0]["finish_reason"] == "stop"
    assert r.text.rstrip().endswith("data: [DONE]")


def test_tool_call_is_not_leaked_as_text(tmp_path):
    reply = '```json\n{"tool_calls":[{"name":"search_kb","arguments":{"query":"x"}}]}\n```'
    runner = FakeRunner(RunResult(text=reply, session_id="sid_A"))
    c = TestClient(create_app(Settings(state_dir=tmp_path), runner=runner))
    r = c.post("/v1/chat/completions", json={
        "model": "claude-code",
        "messages": [{"role": "user", "content": "find x"}],
        "tools": TOOLS,
        "stream": True,
    })
    events = sse_events(r.text)
    text = "".join(e["choices"][0]["delta"].get("content", "") or "" for e in events)
    assert "tool_calls" not in text
    assert text == ""
    tool_deltas = [e for e in events if e["choices"][0]["delta"].get("tool_calls")]
    assert tool_deltas
    assert events[-1]["choices"][0]["finish_reason"] == "tool_calls"


def test_chunks_are_well_formed(tmp_path):
    runner = FakeRunner(RunResult(text="ok", session_id="sid_A"))
    c = TestClient(create_app(Settings(state_dir=tmp_path), runner=runner))
    r = c.post("/v1/chat/completions", json={
        "model": "claude-code",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    })
    for e in sse_events(r.text):
        assert e["object"] == "chat.completion.chunk"
        assert e["model"] == "claude-code"
        assert e["choices"][0]["index"] == 0
