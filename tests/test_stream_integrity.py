"""The streamed text must equal the final text exactly - no loss, no duplication.

A tool-call payload must never appear in the stream, even when the model
prefaces it with prose. But an ordinary answer containing a code fence must
still stream in full and exactly once.
"""
import json

from fastapi.testclient import TestClient

from dtrelay.config import Settings
from dtrelay.runner import RunResult
from dtrelay.server import create_app
from tests.test_server import TOOLS

CALL = '{"tool_calls":[{"name":"search_kb","arguments":{"query":"x"}}]}'


class ChunkRunner:
    """Emits the reply in small pieces, like real token streaming."""

    def __init__(self, text, size=7):
        self.text, self.size = text, size

    def __call__(self, settings, prompt, session_id, system_prompt, on_delta=None):
        if on_delta:
            for i in range(0, len(self.text), self.size):
                on_delta(self.text[i:i + self.size])
        return RunResult(text=self.text, session_id="sid_A")


def stream(tmp_path, reply, tools=None):
    app = create_app(Settings(state_dir=tmp_path), runner=ChunkRunner(reply))
    r = TestClient(app).post("/v1/chat/completions", json={
        "model": "claude-code",
        "messages": [{"role": "user", "content": "q"}],
        "tools": tools or [],
        "stream": True,
    })
    events = [json.loads(l[6:]) for l in r.text.splitlines()
              if l.startswith("data: ") and l[6:].strip() != "[DONE]"]
    text = "".join(e["choices"][0]["delta"].get("content") or "" for e in events)
    calls = [e for e in events if e["choices"][0]["delta"].get("tool_calls")]
    finish = [e["choices"][0]["finish_reason"] for e in events
              if e["choices"][0]["finish_reason"]]
    return text, calls, finish


def test_plain_prose_streams_exactly_once(tmp_path):
    answer = "Photosynthesis converts light energy into chemical energy in two stages."
    text, calls, finish = stream(tmp_path, answer)
    assert text == answer
    assert not calls
    assert finish == ["stop"]


def test_prose_with_a_code_fence_streams_exactly_once(tmp_path):
    answer = "The net equation:\n\n```\n6CO2 + 6H2O -> C6H12O6 + 6O2\n```\n\nThat is it."
    text, calls, finish = stream(tmp_path, answer)
    assert text == answer, "must not be truncated or duplicated"
    assert not calls
    assert finish == ["stop"]


def test_a_bare_tool_call_never_streams_text(tmp_path):
    text, calls, finish = stream(tmp_path, f"```json\n{CALL}\n```", TOOLS)
    assert text == ""
    assert calls
    assert finish == ["tool_calls"]


def test_a_tool_call_with_a_preamble_never_leaks_json(tmp_path):
    reply = f"Let me look that up.\n\n```json\n{CALL}\n```"
    text, calls, finish = stream(tmp_path, reply, TOOLS)
    assert "tool_calls" not in text, "payload must never reach the user"
    assert "{" not in text, "not even a fragment of the payload"
    assert calls, "the call must still be dispatched"
    assert finish == ["tool_calls"]
