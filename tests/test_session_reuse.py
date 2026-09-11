"""Regression: the round-2 shape DeepTutor actually sends.

Captured from a live DeepTutor turn. After a tool call, DeepTutor echoes the
assistant turn back with EMPTY content plus a tool_calls array -- not the raw
JSON the model emitted. Persisting the raw JSON made every round miss the
fingerprint and start a cold session.
"""
from fastapi.testclient import TestClient

from dtrelay.config import Settings
from dtrelay.runner import RunResult
from dtrelay.server import create_app
from tests.test_server import TOOLS, FakeRunner

SYSTEM = {"role": "system", "content": "## general\nYou are DeepTutor."}
USER = {"role": "user", "content": "Search for X."}
TOOL_REPLY = '{"tool_calls":[{"name":"search_kb","arguments":{"query":"X"}}]}'


def test_round_two_after_a_tool_call_resumes_the_session(tmp_path):
    runner = FakeRunner(
        RunResult(text=TOOL_REPLY, session_id="sid_A"),
        RunResult(text="Here is X.", session_id="sid_A"),
    )
    c = TestClient(create_app(Settings(state_dir=tmp_path), runner=runner))

    first = c.post("/v1/chat/completions", json={
        "model": "claude-code", "messages": [SYSTEM, USER], "tools": TOOLS,
    }).json()
    call_id = first["choices"][0]["message"]["tool_calls"][0]["id"]

    # Exactly what DeepTutor sends next: assistant with empty content.
    c.post("/v1/chat/completions", json={
        "model": "claude-code",
        "tools": TOOLS,
        "messages": [
            SYSTEM, USER,
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": call_id, "type": "function",
                 "function": {"name": "search_kb", "arguments": '{"query": "X"}'}}
            ]},
            {"role": "tool", "name": "search_kb", "tool_call_id": call_id,
             "content": "### Search Results"},
        ],
    })

    second = runner.calls[1]
    assert second["session_id"] == "sid_A", "round 2 must resume, not start cold"
    assert "You are DeepTutor" not in second["prompt"], "system prompt must not be resent"
    assert "Search for X." not in second["prompt"], "user turn must not be resent"
    assert "Search Results" in second["prompt"], "only the tool result is new"


def test_assistant_content_none_also_resumes(tmp_path):
    """Some clients send content: null rather than an empty string."""
    runner = FakeRunner(
        RunResult(text=TOOL_REPLY, session_id="sid_A"),
        RunResult(text="Here is X.", session_id="sid_A"),
    )
    c = TestClient(create_app(Settings(state_dir=tmp_path), runner=runner))
    c.post("/v1/chat/completions", json={
        "model": "claude-code", "messages": [SYSTEM, USER], "tools": TOOLS,
    })
    c.post("/v1/chat/completions", json={
        "model": "claude-code", "tools": TOOLS,
        "messages": [
            SYSTEM, USER,
            {"role": "assistant", "content": None, "tool_calls": [{"id": "call_x"}]},
            {"role": "tool", "name": "search_kb", "content": "### Search Results"},
        ],
    })
    assert runner.calls[1]["session_id"] == "sid_A"
