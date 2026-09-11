import json

from fastapi.testclient import TestClient

from dtrelay.config import Settings
from dtrelay.runner import RunResult
from dtrelay.server import create_app

TOOLS = [{
    "type": "function",
    "function": {
        "name": "search_kb",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
}]


class FakeRunner:
    """Records every call and returns queued results."""

    def __init__(self, *results):
        self.results = list(results)
        self.calls = []

    def __call__(self, settings, prompt, session_id, system_prompt, on_delta=None):
        self.calls.append({
            "prompt": prompt, "session_id": session_id, "system_prompt": system_prompt,
        })
        r = self.results.pop(0)
        if on_delta and r.text:
            on_delta(r.text)
        return r


def client(runner, tmp_path, **kw):
    s = Settings(state_dir=tmp_path, **kw)
    return TestClient(create_app(s, runner=runner)), runner


def test_healthz(tmp_path):
    c, _ = client(FakeRunner(), tmp_path)
    assert c.get("/healthz").json()["status"] == "ok"


def test_models_advertises_claude_code(tmp_path):
    c, _ = client(FakeRunner(), tmp_path)
    ids = [m["id"] for m in c.get("/v1/models").json()["data"]]
    assert "claude-code" in ids


def test_prose_reply_becomes_a_message(tmp_path):
    c, _ = client(FakeRunner(RunResult(text="4", session_id="sid_A")), tmp_path)
    body = c.post("/v1/chat/completions", json={
        "model": "claude-code",
        "messages": [{"role": "user", "content": "2+2?"}],
    }).json()
    assert body["choices"][0]["message"]["content"] == "4"
    assert body["choices"][0]["finish_reason"] == "stop"


def test_tool_call_reply_becomes_tool_calls(tmp_path):
    reply = '{"tool_calls":[{"name":"search_kb","arguments":{"query":"x"}}]}'
    c, _ = client(FakeRunner(RunResult(text=reply, session_id="sid_A")), tmp_path)
    body = c.post("/v1/chat/completions", json={
        "model": "claude-code",
        "messages": [{"role": "user", "content": "find x"}],
        "tools": TOOLS,
    }).json()
    choice = body["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    call = choice["message"]["tool_calls"][0]
    assert call["function"]["name"] == "search_kb"
    assert json.loads(call["function"]["arguments"]) == {"query": "x"}


def test_second_turn_resumes_and_sends_only_the_delta(tmp_path):
    runner = FakeRunner(
        RunResult(text="hi", session_id="sid_A"),
        RunResult(text="again", session_id="sid_A"),
    )
    c, _ = client(runner, tmp_path)
    first = [{"role": "user", "content": "u1"}]
    c.post("/v1/chat/completions", json={"model": "claude-code", "messages": first})
    c.post("/v1/chat/completions", json={"model": "claude-code", "messages": first + [
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "u2"},
    ]})
    second = runner.calls[1]
    assert second["session_id"] == "sid_A"
    assert "u1" not in second["prompt"]
    assert "u2" in second["prompt"]


def test_malformed_tool_json_retries_once_then_degrades(tmp_path):
    runner = FakeRunner(
        RunResult(text='```json\n{"tool_calls":[{oops}]}\n```', session_id="sid_A"),
        RunResult(text='```json\n{"still":"broken"', session_id="sid_A"),
    )
    c, _ = client(runner, tmp_path)
    body = c.post("/v1/chat/completions", json={
        "model": "claude-code",
        "messages": [{"role": "user", "content": "find x"}],
        "tools": TOOLS,
    }).json()
    assert len(runner.calls) == 2
    assert "rejected" in runner.calls[1]["prompt"].lower()
    assert body["choices"][0]["finish_reason"] == "stop"


def test_runner_error_returns_502(tmp_path):
    c, _ = client(FakeRunner(RunResult(text="claude error: boom", is_error=True)), tmp_path)
    r = c.post("/v1/chat/completions", json={
        "model": "claude-code", "messages": [{"role": "user", "content": "x"}],
    })
    assert r.status_code == 502


def test_tool_schemas_reach_the_system_prompt(tmp_path):
    runner = FakeRunner(RunResult(text="ok", session_id="sid_A"))
    c, _ = client(runner, tmp_path)
    c.post("/v1/chat/completions", json={
        "model": "claude-code",
        "messages": [{"role": "user", "content": "x"}],
        "tools": TOOLS,
    })
    assert "search_kb" in runner.calls[0]["system_prompt"]
