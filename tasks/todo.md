# dtrelay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A local OpenAI-compatible HTTP daemon that makes Claude Code serve as DeepTutor's LLM, so DeepTutor runs entirely on the user's Claude subscription with no API key, while keeping DeepTutor's own tool loop and agent delegation intact.

**Architecture:** FastAPI daemon on 127.0.0.1:8787 exposing `/v1/chat/completions`. Each request is fingerprinted against a persisted map of conversation prefixes to Claude Code session ids; on a hit the relay resumes that session and sends only the new messages. DeepTutor's tool schemas are injected into an appended system prompt with a strict JSON reply contract, and Claude Code runs with its own tools disabled so it reasons rather than acts. Replies that parse as tool calls become OpenAI `tool_calls`; everything else streams as text.

**Tech Stack:** Python 3.13 (`/Users/long.nguyen5/miniconda3/bin/python3`), FastAPI 0.141, uvicorn 0.52, jsonschema 4.26, pytest 9.0. Synchronous/threaded — no asyncio, no pytest-asyncio.

**Spec:** `docs/superpowers/specs/2026-09-11-claude-code-relay-design.md`

## Global Constraints

- **Zero edits to site-packages.** The installed `deeptutor` package is never modified. DeepTutor is configured through its Catalog UI only.
- **Prompt is fed via stdin, never argv.** A prompt starting with `-` or `---` is otherwise parsed as a CLI option.
- **Claude Code runs with tools disabled:** `--disallowedTools Bash Read Write Edit Glob Grep WebSearch WebFetch Task TodoWrite`.
- **No network in unit tests.** All `claude` invocations in tests go through a fake CLI script. Only Task 9 touches the real `claude`.
- **Synchronous, thread-based.** `threading.Lock` / `threading.Semaphore`; `StreamingResponse` over a sync generator.
- **Project root:** `/Users/long.nguyen5/Documents/bots/dtrelay`.
- **Run tests with:** `cd ~/Documents/bots/dtrelay && python3 -m pytest`
- **Default port 8787; default `DTRELAY_MAX_CONCURRENT=2`; default `DTRELAY_TIMEOUT=900`.**
- OpenAI wire detail: `tool_calls[].function.arguments` is a **JSON-encoded string**, not an object.

## File Structure

| File | Responsibility |
|---|---|
| `dtrelay/config.py` | Env-backed settings dataclass |
| `dtrelay/sessions.py` | Conversation prefix → Claude session id |
| `dtrelay/translate.py` | OpenAI messages+tools ⇄ prompt / tool_calls |
| `dtrelay/runner.py` | The only module that spawns `claude` |
| `dtrelay/limits.py` | Per-session lock + global semaphore |
| `dtrelay/server.py` | HTTP/SSE only, no business logic |
| `tests/fake_claude.py` | Fake CLI emitting stream-json, for hermetic tests |
| `run.sh` / `start.sh` | Foreground / idempotent background start |

---

### Task 1: Project scaffolding and config

**Files:**
- Create: `dtrelay/__init__.py`, `dtrelay/config.py`, `.env.example`, `.gitignore`, `pytest.ini`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing
- Produces: `Settings` dataclass with fields `port:int`, `claude_bin:str`, `model:str`, `timeout:int`, `max_concurrent:int`, `cwd:str`, `state_dir:Path`, `session_ttl_hours:int`; `load_settings(env: dict | None = None) -> Settings`

- [ ] **Step 1: Create the package skeleton**

```bash
cd ~/Documents/bots/dtrelay
mkdir -p dtrelay tests state
touch dtrelay/__init__.py tests/__init__.py
printf '[pytest]\ntestpaths = tests\n' > pytest.ini
printf 'state/\n__pycache__/\n*.pyc\n.env\n' > .gitignore
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_config.py
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
```

- [ ] **Step 3: Run it to make sure it fails**

Run: `cd ~/Documents/bots/dtrelay && python3 -m pytest tests/test_config.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'dtrelay.config'`

- [ ] **Step 4: Implement config**

```python
# dtrelay/config.py
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
```

- [ ] **Step 5: Run the tests and make sure they pass**

Run: `python3 -m pytest tests/test_config.py -v`
Expected: 3 passed

- [ ] **Step 6: Write `.env.example`**

```bash
cat > .env.example <<'EOF'
DTRELAY_PORT=8787
DTRELAY_CLAUDE_BIN=claude
DTRELAY_MODEL=
DTRELAY_TIMEOUT=900
DTRELAY_MAX_CONCURRENT=2
DTRELAY_CWD=
DTRELAY_STATE_DIR=
DTRELAY_SESSION_TTL_HOURS=72
EOF
```

- [ ] **Step 7: Commit**

```bash
git init -q 2>/dev/null; git add -A && git commit -q -m "feat: project scaffolding and env-backed config"
```

---

### Task 2: Session fingerprint map

**Files:**
- Create: `dtrelay/sessions.py`
- Test: `tests/test_sessions.py`

**Interfaces:**
- Consumes: `Settings` from Task 1
- Produces:
  - `canon(message: dict) -> str`
  - `fingerprint_chain(messages: list[dict]) -> list[str]` — `chain[i]` is the hash of `messages[:i+1]`
  - `SessionStore(path: Path, ttl_hours: int)` with `lookup(messages) -> tuple[str | None, int]` returning `(session_id, prefix_len)`, `remember(messages, session_id) -> None`, `reap() -> int`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sessions.py
from dtrelay.sessions import SessionStore, canon, fingerprint_chain


def M(role, content, **kw):
    return {"role": role, "content": content, **kw}


def test_canon_ignores_irrelevant_keys():
    assert canon(M("user", "hi")) == canon(M("user", "hi", extra="ignored"))


def test_canon_distinguishes_role_and_text():
    assert canon(M("user", "hi")) != canon(M("assistant", "hi"))
    assert canon(M("user", "hi")) != canon(M("user", "ho"))


def test_chain_is_prefix_stable():
    a = [M("system", "s"), M("user", "u1")]
    b = a + [M("assistant", "a1"), M("user", "u2")]
    assert fingerprint_chain(b)[:2] == fingerprint_chain(a)


def test_lookup_returns_longest_known_prefix(tmp_path):
    store = SessionStore(tmp_path / "s.json", ttl_hours=72)
    convo = [M("system", "s"), M("user", "u1"), M("assistant", "a1")]
    store.remember(convo, "sid_A")

    sid, n = store.lookup(convo + [M("user", "u2")])
    assert sid == "sid_A"
    assert n == 3


def test_lookup_misses_on_a_branch(tmp_path):
    store = SessionStore(tmp_path / "s.json", ttl_hours=72)
    store.remember([M("system", "s"), M("user", "u1")], "sid_A")

    sid, n = store.lookup([M("system", "s"), M("user", "DIFFERENT")])
    assert sid is None
    assert n == 0


def test_lookup_misses_when_system_prompt_changes(tmp_path):
    store = SessionStore(tmp_path / "s.json", ttl_hours=72)
    store.remember([M("system", "s"), M("user", "u1")], "sid_A")

    sid, _ = store.lookup([M("system", "DIFFERENT"), M("user", "u1")])
    assert sid is None


def test_remember_survives_reload(tmp_path):
    p = tmp_path / "s.json"
    convo = [M("user", "u1")]
    SessionStore(p, ttl_hours=72).remember(convo, "sid_A")

    sid, n = SessionStore(p, ttl_hours=72).lookup(convo)
    assert (sid, n) == ("sid_A", 1)


def test_reap_drops_expired_entries(tmp_path):
    p = tmp_path / "s.json"
    store = SessionStore(p, ttl_hours=0)
    store.remember([M("user", "u1")], "sid_A")
    assert store.reap() == 1
    assert store.lookup([M("user", "u1")]) == (None, 0)
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `python3 -m pytest tests/test_sessions.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'dtrelay.sessions'`

- [ ] **Step 3: Implement sessions**

```python
# dtrelay/sessions.py
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
```

- [ ] **Step 4: Run the tests and make sure they pass**

Run: `python3 -m pytest tests/test_sessions.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add dtrelay/sessions.py tests/test_sessions.py && git commit -q -m "feat: conversation fingerprint -> claude session map"
```

---

### Task 3: Outbound translation (messages + tools → prompt)

**Files:**
- Create: `dtrelay/translate.py`
- Test: `tests/test_translate_out.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `contract_system_prompt(tools: list[dict]) -> str`
  - `render_messages(messages: list[dict]) -> str`
  - `to_prompt(delta_messages: list[dict]) -> str`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_translate_out.py
from dtrelay.translate import contract_system_prompt, render_messages, to_prompt

TOOLS = [{
    "type": "function",
    "function": {
        "name": "search_kb",
        "description": "Search the knowledge base",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
}]


def test_contract_names_the_tool_and_the_json_shape():
    p = contract_system_prompt(TOOLS)
    assert "search_kb" in p
    assert "Search the knowledge base" in p
    assert '"tool_calls"' in p
    assert "executed by the CALLER" in p


def test_contract_without_tools_forbids_tool_calls():
    p = contract_system_prompt([])
    assert "no tools are available" in p.lower()


def test_tool_result_messages_render_with_their_name():
    out = render_messages([
        {"role": "tool", "name": "search_kb", "content": "3 hits"},
    ])
    assert "TOOL RESULT (search_kb): 3 hits" in out


def test_user_and_assistant_turns_are_labelled():
    out = render_messages([
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ])
    assert "USER: hello" in out
    assert "ASSISTANT: hi" in out


def test_system_messages_render_as_instructions():
    out = render_messages([{"role": "system", "content": "be terse"}])
    assert "INSTRUCTIONS: be terse" in out


def test_to_prompt_strips_a_lone_user_turn_to_bare_text():
    # A single user message needs no labelling — it reads as a plain question.
    assert to_prompt([{"role": "user", "content": "what is 2+2?"}]) == "what is 2+2?"


def test_to_prompt_labels_multi_message_deltas():
    out = to_prompt([
        {"role": "tool", "name": "search_kb", "content": "3 hits"},
        {"role": "user", "content": "summarise"},
    ])
    assert "TOOL RESULT (search_kb): 3 hits" in out
    assert "USER: summarise" in out
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `python3 -m pytest tests/test_translate_out.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'dtrelay.translate'`

- [ ] **Step 3: Implement outbound translation**

```python
# dtrelay/translate.py
"""OpenAI wire format <-> Claude Code prompt/reply.

Outbound: DeepTutor's tool schemas become an instruction block, and messages
become labelled plain text. Inbound: a JSON reply becomes OpenAI tool_calls.

This is the only fragile part of the relay, so every failure degrades to prose
rather than erroring the turn.
"""
from __future__ import annotations

import json

from dtrelay.sessions import _text

_CONTRACT_HEAD = (
    "You are the reasoning engine for an application. The TOOLS below are "
    "executed by the CALLER, never by you. You have no tools of your own.\n"
    "To call one, reply with ONLY a fenced json block and nothing else:\n"
    '```json\n{"tool_calls":[{"name":"<tool>","arguments":{...}}]}\n```\n'
    "Call a tool only when you need its result. Otherwise reply normally in prose.\n"
)

_NO_TOOLS = (
    "You are the reasoning engine for an application. You have no tools of "
    "your own and no tools are available this turn. Reply in prose.\n"
)


def _signature(fn: dict) -> str:
    params = (fn.get("parameters") or {}).get("properties") or {}
    required = set((fn.get("parameters") or {}).get("required") or [])
    args = ", ".join(
        f"{k}: {v.get('type', 'any')}{'' if k in required else '?'}"
        for k, v in params.items()
    )
    desc = fn.get("description") or ""
    return f"- {fn.get('name', '')}({args}) -> {desc}".rstrip()


def contract_system_prompt(tools: list[dict]) -> str:
    if not tools:
        return _NO_TOOLS
    lines = [_signature(t.get("function") or {}) for t in tools]
    return _CONTRACT_HEAD + "\nTOOLS:\n" + "\n".join(lines) + "\n"


def render_messages(messages: list[dict]) -> str:
    out = []
    for m in messages:
        role = m.get("role", "")
        body = _text(m.get("content"))
        if role == "tool":
            out.append(f"TOOL RESULT ({m.get('name', '')}): {body}")
        elif role == "system":
            out.append(f"INSTRUCTIONS: {body}")
        elif role == "assistant":
            out.append(f"ASSISTANT: {body}")
        else:
            out.append(f"USER: {body}")
    return "\n\n".join(out)


def to_prompt(delta_messages: list[dict]) -> str:
    """Render the messages the session has not seen yet."""
    if len(delta_messages) == 1 and delta_messages[0].get("role") == "user":
        return _text(delta_messages[0].get("content"))
    return render_messages(delta_messages)
```

- [ ] **Step 4: Run the tests and make sure they pass**

Run: `python3 -m pytest tests/test_translate_out.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add dtrelay/translate.py tests/test_translate_out.py && git commit -q -m "feat: render messages and tool schemas into a claude prompt"
```

---

### Task 4: Inbound translation (reply → tool_calls)

**Files:**
- Modify: `dtrelay/translate.py` (append)
- Test: `tests/test_translate_in.py`

**Interfaces:**
- Consumes: `contract_system_prompt` from Task 3
- Produces:
  - `ParsedReply` dataclass: `.tool_calls: list[dict] | None`, `.content: str | None`, `.error: str | None`
  - `parse_reply(text: str, tools: list[dict]) -> ParsedReply`
  - `looks_like_json(prefix: str) -> bool`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_translate_in.py
from dtrelay.translate import looks_like_json, parse_reply

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

FENCED = '```json\n{"tool_calls":[{"name":"search_kb","arguments":{"query":"x"}}]}\n```'
BARE = '{"tool_calls":[{"name":"search_kb","arguments":{"query":"x"}}]}'


def test_parses_a_fenced_tool_call():
    r = parse_reply(FENCED, TOOLS)
    assert r.error is None
    assert r.tool_calls[0]["function"]["name"] == "search_kb"


def test_arguments_are_a_json_encoded_string():
    # OpenAI wire format requires a string here, not an object.
    r = parse_reply(FENCED, TOOLS)
    assert r.tool_calls[0]["function"]["arguments"] == '{"query": "x"}'


def test_parses_a_bare_tool_call():
    assert parse_reply(BARE, TOOLS).tool_calls is not None


def test_each_call_gets_a_unique_id():
    r = parse_reply(
        '{"tool_calls":[{"name":"search_kb","arguments":{"query":"a"}},'
        '{"name":"search_kb","arguments":{"query":"b"}}]}',
        TOOLS,
    )
    ids = [c["id"] for c in r.tool_calls]
    assert len(set(ids)) == 2
    assert all(i.startswith("call_") for i in ids)


def test_prose_is_returned_as_content():
    r = parse_reply("The answer is 4.", TOOLS)
    assert r.tool_calls is None
    assert r.content == "The answer is 4."


def test_unknown_tool_name_is_an_error():
    r = parse_reply('{"tool_calls":[{"name":"nope","arguments":{}}]}', TOOLS)
    assert r.tool_calls is None
    assert "nope" in r.error


def test_arguments_failing_schema_are_an_error():
    r = parse_reply(
        '{"tool_calls":[{"name":"search_kb","arguments":{"query":123}}]}', TOOLS
    )
    assert r.tool_calls is None
    assert r.error


def test_missing_required_argument_is_an_error():
    r = parse_reply('{"tool_calls":[{"name":"search_kb","arguments":{}}]}', TOOLS)
    assert r.tool_calls is None
    assert r.error


def test_malformed_json_in_a_fence_is_an_error():
    r = parse_reply('```json\n{"tool_calls":[{oops}]}\n```', TOOLS)
    assert r.tool_calls is None
    assert r.error


def test_prose_that_merely_mentions_json_is_not_a_tool_call():
    r = parse_reply("You could call search_kb with a query.", TOOLS)
    assert r.tool_calls is None
    assert r.error is None


def test_looks_like_json_detects_fence_and_brace():
    assert looks_like_json("```json")
    assert looks_like_json("  {")
    assert not looks_like_json("The answer")
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `python3 -m pytest tests/test_translate_in.py -v`
Expected: FAIL, `ImportError: cannot import name 'parse_reply'`

- [ ] **Step 3: Append inbound translation to `dtrelay/translate.py`**

```python
# --- inbound ---------------------------------------------------------------
import re
import uuid
from dataclasses import dataclass

import jsonschema

_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


@dataclass
class ParsedReply:
    tool_calls: list[dict] | None = None
    content: str | None = None
    error: str | None = None


def looks_like_json(prefix: str) -> bool:
    """Could this reply be a tool call? Decides whether to buffer or stream."""
    s = prefix.lstrip()
    return s.startswith("```") or s.startswith("{")


def _candidate(text: str) -> str | None:
    m = _FENCE.search(text)
    if m:
        return m.group(1)
    s = text.strip()
    return s if s.startswith("{") else None


def parse_reply(text: str, tools: list[dict]) -> ParsedReply:
    raw = _candidate(text)
    if raw is None:
        return ParsedReply(content=text)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as e:
        return ParsedReply(error=f"reply was not valid JSON: {e}")
    calls = payload.get("tool_calls") if isinstance(payload, dict) else None
    if not isinstance(calls, list) or not calls:
        return ParsedReply(content=text)

    by_name = {
        (t.get("function") or {}).get("name"): (t.get("function") or {})
        for t in tools
    }
    out = []
    for call in calls:
        name = (call or {}).get("name")
        if name not in by_name:
            return ParsedReply(
                error=f"unknown tool {name!r}; valid tools: {sorted(by_name)}"
            )
        args = call.get("arguments")
        if not isinstance(args, dict):
            return ParsedReply(error=f"arguments for {name!r} must be an object")
        schema = by_name[name].get("parameters") or {}
        try:
            jsonschema.validate(args, schema)
        except jsonschema.ValidationError as e:
            return ParsedReply(error=f"arguments for {name!r} invalid: {e.message}")
        out.append({
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        })
    return ParsedReply(tool_calls=out)
```

- [ ] **Step 4: Run the tests and make sure they pass**

Run: `python3 -m pytest tests/test_translate_in.py -v`
Expected: 11 passed

- [ ] **Step 5: Commit**

```bash
git add dtrelay/translate.py tests/test_translate_in.py && git commit -q -m "feat: parse and validate tool calls out of claude replies"
```

---

### Task 5: The `claude` runner

**Files:**
- Create: `dtrelay/runner.py`, `tests/fake_claude.py`
- Test: `tests/test_runner.py`

**Interfaces:**
- Consumes: `Settings`, `DISALLOWED_TOOLS` from Task 1
- Produces:
  - `RunResult` dataclass: `.text: str`, `.session_id: str | None`, `.is_error: bool`, `.stats: dict`
  - `build_command(settings, session_id, system_prompt) -> list[str]`
  - `run(settings, prompt, session_id, system_prompt, on_delta=None) -> RunResult`

- [ ] **Step 1: Write the fake CLI**

```python
# tests/fake_claude.py
"""Stand-in for the `claude` binary: emits stream-json without touching a model.

Behaviour is driven by env vars so one script covers every test case:
  FAKE_RESULT     text the result event carries
  FAKE_SESSION    session id to report
  FAKE_EXIT       process exit code
  FAKE_IS_ERROR   "1" to set is_error on the result event
  FAKE_DELTAS     "|"-separated partial text chunks emitted before the result
  FAKE_ECHO_ARGV  path to write argv to, so tests can assert on flags
  FAKE_ECHO_STDIN path to write the received prompt to
"""
import json
import os
import sys


def main():
    argv_path = os.environ.get("FAKE_ECHO_ARGV")
    if argv_path:
        with open(argv_path, "w") as f:
            f.write("\n".join(sys.argv[1:]))
    prompt = sys.stdin.read()
    stdin_path = os.environ.get("FAKE_ECHO_STDIN")
    if stdin_path:
        with open(stdin_path, "w") as f:
            f.write(prompt)

    exit_code = int(os.environ.get("FAKE_EXIT", "0"))
    if exit_code:
        sys.stderr.write("fake claude failure\n")
        sys.exit(exit_code)

    for chunk in filter(None, os.environ.get("FAKE_DELTAS", "").split("|")):
        print(json.dumps({
            "type": "stream_event",
            "event": {
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": chunk},
            },
        }), flush=True)

    print(json.dumps({
        "type": "result",
        "result": os.environ.get("FAKE_RESULT", "ok"),
        "session_id": os.environ.get("FAKE_SESSION", "sid_fake"),
        "is_error": os.environ.get("FAKE_IS_ERROR") == "1",
        "duration_ms": 12,
        "num_turns": 1,
        "total_cost_usd": 0.01,
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }), flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_runner.py
import sys
from pathlib import Path

import pytest

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
    # A prompt starting with '-' must not be parsed as a CLI flag.
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
    # The retry is the LAST invocation, and it must not carry --resume.
    assert "--resume" not in argv_file.read_text().split("\n")
    assert r.is_error is True


def test_nonzero_exit_without_session_reports_stderr(monkeypatch):
    monkeypatch.setenv("FAKE_EXIT", "1")
    r = run(settings(), "hi", None, "sys")
    assert r.is_error is True
    assert "fake claude failure" in r.text


def test_timeout_is_reported(monkeypatch):
    monkeypatch.setenv("FAKE_SLEEP", "5")
    s = Settings(claude_bin=f"{sys.executable} -c \"import time;time.sleep(5)\"", timeout=1)
    r = run(s, "hi", None, "sys")
    assert r.is_error is True
    assert "timed out" in r.text.lower()
```

- [ ] **Step 3: Run it to make sure it fails**

Run: `python3 -m pytest tests/test_runner.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'dtrelay.runner'`

- [ ] **Step 4: Implement the runner**

```python
# dtrelay/runner.py
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
```

- [ ] **Step 5: Run the tests and make sure they pass**

Run: `python3 -m pytest tests/test_runner.py -v`
Expected: 9 passed

- [ ] **Step 6: Commit**

```bash
git add dtrelay/runner.py tests/test_runner.py tests/fake_claude.py && git commit -q -m "feat: claude subprocess runner with stale-session retry"
```

---

### Task 6: Concurrency limits

**Files:**
- Create: `dtrelay/limits.py`
- Test: `tests/test_limits.py`

**Interfaces:**
- Consumes: nothing
- Produces: `Limiter(max_concurrent: int)` with context manager `slot(session_key: str)`, and `.in_flight: int`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_limits.py
import threading
import time

from dtrelay.limits import Limiter


def test_same_session_runs_are_serialised():
    limiter = Limiter(max_concurrent=4)
    order = []

    def work(tag, hold):
        with limiter.slot("sess_A"):
            order.append(f"{tag}-in")
            time.sleep(hold)
            order.append(f"{tag}-out")

    t1 = threading.Thread(target=work, args=("a", 0.2))
    t2 = threading.Thread(target=work, args=("b", 0.0))
    t1.start(); time.sleep(0.05); t2.start()
    t1.join(); t2.join()
    # b cannot enter before a has left.
    assert order == ["a-in", "a-out", "b-in", "b-out"]


def test_global_cap_limits_distinct_sessions():
    limiter = Limiter(max_concurrent=2)
    peak = {"v": 0}
    lock = threading.Lock()

    def work(i):
        with limiter.slot(f"sess_{i}"):
            with lock:
                peak["v"] = max(peak["v"], limiter.in_flight)
            time.sleep(0.1)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(6)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert peak["v"] <= 2


def test_slot_is_released_on_exception():
    limiter = Limiter(max_concurrent=1)
    try:
        with limiter.slot("sess_A"):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert limiter.in_flight == 0
    with limiter.slot("sess_A"):
        assert limiter.in_flight == 1
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `python3 -m pytest tests/test_limits.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'dtrelay.limits'`

- [ ] **Step 3: Implement limits**

```python
# dtrelay/limits.py
"""Per-conversation serialisation plus a global cap.

One subscription backs every run, so the global semaphore is what keeps a
fan-out turn from melting the plan window. The session lock is acquired FIRST
so a queued conversation never holds a global slot while waiting on itself.
"""
from __future__ import annotations

import threading
from contextlib import contextmanager


class Limiter:
    def __init__(self, max_concurrent: int):
        self._sem = threading.Semaphore(max(1, int(max_concurrent)))
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()
        self._in_flight = 0

    @property
    def in_flight(self) -> int:
        return self._in_flight

    def _lock_for(self, key: str) -> threading.Lock:
        with self._guard:
            return self._locks.setdefault(key, threading.Lock())

    @contextmanager
    def slot(self, session_key: str):
        lock = self._lock_for(session_key)
        lock.acquire()
        try:
            self._sem.acquire()
            with self._guard:
                self._in_flight += 1
            try:
                yield
            finally:
                with self._guard:
                    self._in_flight -= 1
                self._sem.release()
        finally:
            lock.release()
```

- [ ] **Step 4: Run the tests and make sure they pass**

Run: `python3 -m pytest tests/test_limits.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add dtrelay/limits.py tests/test_limits.py && git commit -q -m "feat: per-session lock and global concurrency cap"
```

---

### Task 7: HTTP server, non-streaming

**Files:**
- Create: `dtrelay/server.py`
- Test: `tests/test_server.py`

**Interfaces:**
- Consumes: everything from Tasks 1–6
- Produces: `create_app(settings, runner=None) -> FastAPI`; `handle_completion(...)` internals are private. `runner` is injectable so tests never spawn a process.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_server.py
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
        if on_delta and r.text and not r.text.lstrip().startswith(("{", "```")):
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
    assert "corrective" in runner.calls[1]["prompt"].lower() or "invalid" in runner.calls[1]["prompt"].lower()
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
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `python3 -m pytest tests/test_server.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'dtrelay.server'`

- [ ] **Step 3: Implement the server**

```python
# dtrelay/server.py
"""HTTP surface only — every decision lives in the modules this calls."""
from __future__ import annotations

import logging
import time
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from dtrelay import runner as runner_mod
from dtrelay import translate
from dtrelay.config import MODEL_ID, Settings
from dtrelay.limits import Limiter
from dtrelay.sessions import SessionStore, fingerprint_chain

log = logging.getLogger(__name__)

CORRECTION = (
    "Your previous reply was rejected: {error}\n"
    "This is a corrective retry. Reply with ONLY the fenced json tool_calls "
    "block, or with prose if no tool is needed."
)


def _completion(content, tool_calls):
    message = {"role": "assistant", "content": content}
    finish = "stop"
    if tool_calls:
        message = {"role": "assistant", "content": None, "tool_calls": tool_calls}
        finish = "tool_calls"
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_ID,
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
    }


def create_app(settings: Settings, runner=None) -> FastAPI:
    app = FastAPI(title="dtrelay")
    run = runner or runner_mod.run
    store = SessionStore(settings.state_dir / "sessions.json", settings.session_ttl_hours)
    limiter = Limiter(settings.max_concurrent)

    @app.on_event("startup")
    def _reap_expired_sessions():
        dropped = store.reap()
        if dropped:
            log.info("reaped %d expired session(s)", dropped)

    @app.get("/healthz")
    def healthz():
        return {"status": "ok", "in_flight": limiter.in_flight}

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [
            {"id": MODEL_ID, "object": "model", "owned_by": "dtrelay"}
        ]}

    @app.post("/v1/chat/completions")
    def completions(body: dict):
        messages = body.get("messages") or []
        tools = body.get("tools") or []
        if not messages:
            raise HTTPException(400, "messages is required")

        session_id, prefix_len = store.lookup(messages)
        delta = messages[prefix_len:] if session_id else messages
        if not delta:
            delta = messages[-1:]
        prompt = translate.to_prompt(delta)
        system_prompt = translate.contract_system_prompt(tools)
        # Serialise on the bound session when there is one; a brand-new
        # conversation has no contention to protect against yet.
        session_key = session_id or fingerprint_chain(messages)[-1]

        collected: list[str] = []

        with limiter.slot(session_key):
            result = run(settings, prompt, session_id, system_prompt,
                         on_delta=collected.append)
            if result.is_error:
                raise HTTPException(502, result.text)
            parsed = translate.parse_reply(result.text, tools)
            if parsed.error:
                log.info("corrective retry: %s", parsed.error)
                retry = run(settings, CORRECTION.format(error=parsed.error),
                            result.session_id, system_prompt)
                if not retry.is_error:
                    reparsed = translate.parse_reply(retry.text, tools)
                    result = retry
                    parsed = (reparsed if not reparsed.error
                              else translate.ParsedReply(content=retry.text))

        if result.session_id:
            store.remember(messages + [{"role": "assistant",
                                        "content": parsed.content or result.text}],
                           result.session_id)
        return _completion(parsed.content if parsed.tool_calls is None else None,
                           parsed.tool_calls)

    return app
```

- [ ] **Step 4: Run the tests and make sure they pass**

Run: `python3 -m pytest tests/test_server.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add dtrelay/server.py tests/test_server.py && git commit -q -m "feat: openai-compatible completions endpoint"
```

---

### Task 8: Streaming responses

**Files:**
- Modify: `dtrelay/server.py`
- Test: `tests/test_streaming.py`

**Interfaces:**
- Consumes: `create_app` from Task 7, `looks_like_json` from Task 4
- Produces: `stream=true` support on `/v1/chat/completions`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_streaming.py
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
        e["choices"][0]["delta"].get("content", "") for e in events
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
    tool_deltas = [
        e for e in events if e["choices"][0]["delta"].get("tool_calls")
    ]
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
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `python3 -m pytest tests/test_streaming.py -v`
Expected: FAIL — the endpoint ignores `stream`, so the body is a single JSON object with no `data:` lines

- [ ] **Step 3: Add streaming to `dtrelay/server.py`**

Add these helpers above `create_app`:

```python
def _chunk(delta: dict, finish=None):
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": MODEL_ID,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


def _sse(payload) -> str:
    return f"data: {json.dumps(payload)}\n\n"
```

Add `import json` at the top. Inside `completions`, before the non-streaming
path, insert:

```python
        if body.get("stream"):
            def generate():
                buffered: list[str] = []
                streaming = {"decided": False, "live": False}
                queue: list[str] = []

                def on_delta(chunk: str):
                    buffered.append(chunk)
                    if not streaming["decided"]:
                        head = "".join(buffered).lstrip()
                        if not head:
                            return
                        # Only start streaming once we know it is not a tool call.
                        streaming["decided"] = True
                        streaming["live"] = not translate.looks_like_json(head)
                        if streaming["live"]:
                            queue.append("".join(buffered))
                    elif streaming["live"]:
                        queue.append(chunk)

                with limiter.slot(session_key):
                    result = run(settings, prompt, session_id, system_prompt,
                                 on_delta=on_delta)
                    if result.is_error:
                        yield _sse(_chunk({"content": result.text}, "stop"))
                        yield "data: [DONE]\n\n"
                        return
                    parsed = translate.parse_reply(result.text, tools)
                    if parsed.error:
                        retry = run(settings, CORRECTION.format(error=parsed.error),
                                    result.session_id, system_prompt)
                        if not retry.is_error:
                            reparsed = translate.parse_reply(retry.text, tools)
                            result = retry
                            parsed = (reparsed if not reparsed.error
                                      else translate.ParsedReply(content=retry.text))

                if parsed.tool_calls:
                    for i, call in enumerate(parsed.tool_calls):
                        yield _sse(_chunk({"tool_calls": [{"index": i, **call}]}))
                    yield _sse(_chunk({}, "tool_calls"))
                else:
                    text = parsed.content or result.text
                    if queue and "".join(queue) == text:
                        for piece in queue:
                            yield _sse(_chunk({"content": piece}))
                    else:
                        yield _sse(_chunk({"content": text}))
                    yield _sse(_chunk({}, "stop"))

                if result.session_id:
                    store.remember(
                        messages + [{"role": "assistant", "content": parsed.content or result.text}],
                        result.session_id,
                    )
                yield "data: [DONE]\n\n"

            return StreamingResponse(generate(), media_type="text/event-stream")
```

- [ ] **Step 4: Run the tests and make sure they pass**

Run: `python3 -m pytest tests/test_streaming.py -v`
Expected: 3 passed

- [ ] **Step 5: Run the whole suite**

Run: `python3 -m pytest -v`
Expected: all tests pass (51 total across tasks 1–8)

- [ ] **Step 6: Commit**

```bash
git add dtrelay/server.py tests/test_streaming.py && git commit -q -m "feat: SSE streaming with tool-call buffering"
```

---

### Task 9: Launch scripts, real-CLI smoke test, DeepTutor wiring

**Files:**
- Create: `dtrelay/__main__.py`, `run.sh`, `start.sh`, `stop.sh`, `README.md`
- Test: `tests/test_smoke_real.py` (marked, opt-in)

**Interfaces:**
- Consumes: `create_app`, `load_settings`
- Produces: a running daemon on 127.0.0.1:8787

- [ ] **Step 1: Write the entrypoint**

```python
# dtrelay/__main__.py
"""Run the relay: python3 -m dtrelay"""
import logging

import uvicorn

from dtrelay.config import load_settings
from dtrelay.server import create_app


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    settings = load_settings()
    uvicorn.run(create_app(settings), host="127.0.0.1", port=settings.port)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write the lifecycle scripts**

```bash
cd ~/Documents/bots/dtrelay
cat > run.sh <<'EOF'
#!/usr/bin/env bash
# Foreground. Sources .env if present.
set -a; [ -f "$(dirname "$0")/.env" ] && . "$(dirname "$0")/.env"; set +a
exec /Users/long.nguyen5/miniconda3/bin/python3 -m dtrelay
EOF
cat > start.sh <<'EOF'
#!/usr/bin/env bash
# Idempotent background start.
cd "$(dirname "$0")" || exit 1
pgrep -f "python3 -m dtrelay" >/dev/null && exit 0
nohup ./run.sh >> bot.log 2>&1 &
EOF
cat > stop.sh <<'EOF'
#!/usr/bin/env bash
pkill -f "python3 -m dtrelay"
EOF
chmod +x run.sh start.sh stop.sh
echo 'bot.log' >> .gitignore
```

- [ ] **Step 3: Write the opt-in real-CLI smoke test**

```python
# tests/test_smoke_real.py
"""Opt-in: exercises the REAL claude CLI. Run with:
    DTRELAY_SMOKE=1 python3 -m pytest tests/test_smoke_real.py -v -s
"""
import os

import pytest

from dtrelay.config import load_settings
from dtrelay.runner import run
from dtrelay.translate import contract_system_prompt, parse_reply

pytestmark = pytest.mark.skipif(
    not os.environ.get("DTRELAY_SMOKE"), reason="set DTRELAY_SMOKE=1 to run"
)

TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}]


def test_real_cli_emits_a_valid_tool_call():
    s = load_settings()
    result = run(s, "What is the weather in Hanoi? Use the tool.", None,
                 contract_system_prompt(TOOLS))
    assert not result.is_error, result.text
    parsed = parse_reply(result.text, TOOLS)
    assert parsed.error is None, f"{parsed.error}\nraw: {result.text}"
    assert parsed.tool_calls[0]["function"]["name"] == "get_weather"
    assert result.session_id


def test_real_cli_resumes_with_a_delta():
    s = load_settings()
    first = run(s, "Remember the number 41. Reply OK.", None,
                contract_system_prompt([]))
    assert not first.is_error, first.text
    second = run(s, "Add one to the number you remembered. Reply with digits only.",
                 first.session_id, contract_system_prompt([]))
    assert not second.is_error, second.text
    assert "42" in second.text
```

- [ ] **Step 4: Run the smoke test against the real CLI**

Run: `cd ~/Documents/bots/dtrelay && DTRELAY_SMOKE=1 python3 -m pytest tests/test_smoke_real.py -v -s`
Expected: 2 passed. This proves the contract and resume work against `claude` 2.1.268, not just the fake.

- [ ] **Step 5: Start the daemon and verify it serves**

```bash
cd ~/Documents/bots/dtrelay && ./start.sh && sleep 3
curl -s http://127.0.0.1:8787/healthz
curl -s http://127.0.0.1:8787/v1/models
```

Expected: `{"status":"ok","in_flight":0}` and a model list containing `claude-code`.

- [ ] **Step 6: Prove an end-to-end completion through HTTP**

```bash
curl -s http://127.0.0.1:8787/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"claude-code","messages":[{"role":"user","content":"Reply with exactly: RELAY OK"}]}' \
  | python3 -m json.tool
```

Expected: a `chat.completion` whose message content contains `RELAY OK`.

- [ ] **Step 7: Point DeepTutor at the relay**

In DeepTutor (http://localhost:3782) → Settings ▸ Catalog → add a **Custom** connection:
- `base_url`: `http://127.0.0.1:8787/v1`
- `api_key`: `dummy`

Refresh models, then set `claude-code` as the active model for the **llm** service. Verify:

```bash
curl -s http://127.0.0.1:8001/api/settings/catalog | python3 -c "
import json,sys
c=json.load(sys.stdin)['catalog']
llm=c['services']['llm']
print('active_model:', llm['active_model_id'])
print('active_profile:', llm['active_profile_id'])
"
```

Expected: both non-null, `active_model_id` = `claude-code`.

- [ ] **Step 8: Connect the delegation subagent**

```bash
curl -s -X POST http://127.0.0.1:8001/api/subagents/connections \
  -H 'Content-Type: application/json' \
  -d '{"name":"Claude Code","agent_kind":"claude_code","cwd":""}'
```

Then set a conservative consult budget (the relay cannot see these runs):

```bash
curl -s -X PUT http://127.0.0.1:8001/api/subagents/settings \
  -H 'Content-Type: application/json' \
  -d '{"consult_budget":3}'
```

- [ ] **Step 9: Verify a real DeepTutor turn**

Send a message in the DeepTutor UI. Then confirm no LLM-config errors were logged:

```bash
tail -20 ~/Documents/my-deeptutor/data/user/logs/deeptutor.jsonl \
  | grep -c "No active LLM model is configured" || echo "0 errors — good"
tail -5 ~/Documents/bots/dtrelay/bot.log
```

Expected: zero "No active LLM model" errors, and relay log lines showing `claude run (session=...)`.

- [ ] **Step 10: Write the README**

```bash
cat > README.md <<'EOF'
# dtrelay — Claude Code as DeepTutor's model

An OpenAI-compatible daemon that makes DeepTutor run on a Claude subscription
with no API key. Each DeepTutor conversation is bound to a Claude Code session;
DeepTutor's tool schemas are translated into a JSON reply contract, and Claude
Code runs with its own tools disabled so it reasons instead of acting.

    DeepTutor ─► http://127.0.0.1:8787/v1 ─► claude -p --resume <sid>

## Run

    cp .env.example .env
    ./start.sh          # background, logs to bot.log
    ./run.sh            # foreground

## Wire up DeepTutor

Settings ▸ Catalog ▸ Custom: base_url `http://127.0.0.1:8787/v1`, key `dummy`.
Set `claude-code` active for the **llm** service.

## Tests

    python3 -m pytest                                   # hermetic, uses a fake CLI
    DTRELAY_SMOKE=1 python3 -m pytest tests/test_smoke_real.py   # real claude

## Limits

- Embeddings are not served — knowledge bases need Ollama or an API key.
- Subagent consults spawn outside the relay and share the plan window; keep
  `consult_budget` low.
- Each call carries Claude Code's agent system prompt, so cost per call is
  roughly flat regardless of prompt size.

Design: `docs/superpowers/specs/2026-09-11-claude-code-relay-design.md`
EOF
```

- [ ] **Step 11: Final commit**

```bash
git add -A && git commit -q -m "feat: entrypoint, lifecycle scripts, smoke test, README"
```

---

## Review

**Status: complete and verified live on 2026-09-11.**

All 9 tasks executed. 54 hermetic tests pass (fake CLI, no network) plus 2
opt-in real-CLI smoke tests.

### Verified end to end

| Check | Evidence |
|---|---|
| Relay serves OpenAI wire format | `/healthz`, `/v1/models`, `/v1/chat/completions` all 200 |
| Tool translation over HTTP | tool schema in -> `finish_reason: "tool_calls"`, args as JSON string |
| DeepTutor reaches the relay | `/api/system/test/llm` -> `success: true`, model `claude-code` |
| Full tutoring turn | live WS turn: `tool_call(web_search)` -> `tool_result` -> `result` |
| DeepTutor dispatches its OWN tools | `web_search` fired with translated args, sources returned |
| Session resume | round 2 logs `claude run (session=dac1c810-...)` |
| Runs on the subscription | no `ANTHROPIC_API_KEY` in env; spawned `claude` uses `~/.claude` OAuth |

### Deviations from the plan

1. **Sync/threaded, not async.** `pytest-asyncio` is absent from the host
   environment; threads match kaide's proven model and keep every unit
   testable. Spec updated to match.
2. **`lifespan` instead of `@app.on_event("startup")`**, which FastAPI 0.141
   deprecates.
3. **`_resolve` / `_persist` extracted** in `server.py` so the streaming and
   non-streaming paths share one copy of the corrective-retry and persistence
   logic instead of duplicating it as the plan sketched.
4. **`POST /api/settings/apply/service`** used to configure DeepTutor rather
   than a whole-catalog `PUT`, so only the `llm` service was touched.
5. **`DTRELAY_DUMP` debug hook added** to `server.py` -- env-gated capture of
   incoming message arrays. Kept because it is what diagnosed the bug below.

### Bug found and fixed during verification

The plan's own integration test passed while the feature was broken. The
`FakeRunner` echoed back exactly what the relay had stored, so the fingerprint
always matched; real DeepTutor does not.

After a tool call, DeepTutor echoes the assistant turn back with **empty
content** plus a `tool_calls` array. The relay had persisted that turn as the
raw JSON the model emitted, so every fingerprint missed and **every round
started a cold session** -- 8 distinct sessions for what should have been one,
each paying a full agent boot and re-sending the whole conversation.

Caught by inspecting `bot.log` (`session=new` on every line) rather than by the
test suite. Fixed in `_persist`; regression test in `tests/test_session_reuse.py`
built from the captured live payload, not from an assumption.

### Follow-up work (2026-09-11, after the plan closed)

**Knowledge bases now work.** `dtrelay/embeddings.py` serves `/v1/embeddings`
from a local ONNX model (BAAI/bge-small-en-v1.5, 384 dims, CPU, ~130MB), so the
spec's hardest carve-out is closed and the system needs no API key at all.
Verified by indexing a document containing a planted fictional marker and
confirming the tutor retrieved it.

**Second bug found in production.** `_candidate()` searched for a code fence
anywhere in a reply, so any tutoring answer containing a code block was
misparsed as a failed tool call and burned a ~60s corrective retry. Six fired
before it was caught in `bot.log`. Fixed: the whole reply must BE the payload.

### Known limitations (remaining)

- Subagent consults spawn outside the relay and share the plan window.
- Fixed agent-prompt cost per call; multi-round turns are slow (~10s/round).
