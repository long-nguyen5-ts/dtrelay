# dtrelay — Claude Code as DeepTutor's model

**Date:** 2026-09-11
**Status:** Approved design, ready for implementation planning

## Goal

Run DeepTutor entirely on the user's Claude subscription, with no Anthropic API
key, while preserving DeepTutor's own tool loop — including its ability to
delegate to a full-tool agent when a task warrants one.

Success is a DeepTutor chat turn that:

1. reasons via the user's `~/.claude` OAuth credentials (no API key anywhere),
2. fires DeepTutor's own tools (KB retrieval, mastery, reading) as usual, and
3. can delegate to a tool-enabled Claude Code subagent mid-turn.

## Non-goals

- Serving embeddings (see Known limitations).
- Modifying the installed `deeptutor` package. Zero edits to site-packages.
- Replacing DeepTutor's subagent layer, which already works.

## Background

DeepTutor's provider contract (`services/llm/provider_core/base.py`) is a
stateless model API:

```python
async def chat(messages, tools=None, max_tokens=..., temperature=0.7, ...) -> LLMResponse  # .tool_calls
async def chat_stream(...)
```

Claude Code is an agent, not a model endpoint: it owns its loop, its tools and
its context. The relay's job is to make one look like the other.

Two provider specs in `services/provider_registry.py` are `is_direct=True` —
`custom` (openai_compat backend) and `custom_anthropic` (anthropic backend) —
meaning the operator supplies base URL and key with no auto-detection. That is
the seam: a local daemon speaking OpenAI wire format plugs in through the
Catalog UI with no code changes to DeepTutor.

### Two roles for Claude Code, one subscription

| | Role | Tools | Session scope |
|---|---|---|---|
| **Relay** | DeepTutor's model — pure reasoning engine | disabled via `--disallowedTools` | one per DeepTutor chat, via `--resume` |
| **Subagent** | The agent DeepTutor delegates *to* | full access, `bypassPermissions` | one per `(chat_session_id, connection)` — already implemented |

Delegation needs no new code. DeepTutor's `consult` tool is simply one more
entry in the tool schema list the relay injects; when the reasoner emits a
`consult` tool call, DeepTutor dispatches its existing subagent machinery.

### Prior art

`~/Documents/bots/kaide/kaide_bot.py` (1321 lines) is a proven Slack ⇄ Claude
Code relay. `runner.py` and `sessions.py` below are ports of its `run_claude`
and session-map code, including the stale-session retry. The one problem kaide
never had: Slack supplies `channel:thread_ts` as a conversation id, while the
OpenAI API supplies none. See Session binding.

## Proven assumptions

Both load-bearing behaviours were verified against `claude` 2.1.268 before this
design was written.

**1. Tools off + contract yields a clean tool call.** With
`--disallowedTools Bash Read Write Edit Glob Grep WebSearch WebFetch Task TodoWrite`
and the contract in the appended system prompt, the prompt *"What is the current
temperature in Hanoi? Use the tool."* returned exactly:

```json
{"tool_calls":[{"name":"get_weather","arguments":{"city":"Hanoi","unit":"c"}}]}
```

No self-execution, no prose wrapper, `is_error: false`.

**2. Resume accepts a delta and keeps context.** Resuming that session id with
only `TOOL RESULT (get_weather): {"city":"Hanoi","temp":31,...}` — no history
resent — returned *"Hanoi is currently 31 °C, humid with scattered clouds"* in
`num_turns: 1`, with the same session id.

## Architecture

```
DeepTutor  (site-packages — UNTOUCHED)
  Settings ▸ Catalog ▸ Custom provider
  base_url = http://127.0.0.1:8787/v1   api_key = dummy
      │  OpenAI /v1/chat/completions  (messages, tools, stream)
      ▼
dtrelay/  ~/Documents/bots/dtrelay/
  server.py    FastAPI: /v1/chat/completions, /v1/models, /healthz
  translate.py OpenAI msgs+tools ⇄ claude prompt / tool_calls
  sessions.py  prefix fingerprint → session_id (atomic tmp+replace)
  runner.py    claude -p --resume, stream-json parse
  limits.py    per-session lock + global semaphore
      │  subprocess, prompt via stdin
      ▼
claude -p --output-format stream-json --verbose --include-partial-messages
       --resume <sid> --disallowedTools ...
```

## Components

Each unit has one purpose and is independently testable.

### `runner.py`

The only module that knows the CLI exists. Ported from kaide's `run_claude`.

```python
def run(prompt: str, session_id: str | None, *, on_delta, timeout: int) -> RunResult
# RunResult: text, session_id, is_error, stats
```

Synchronous and thread-based, mirroring kaide. The host environment has no
`pytest-asyncio`, and threads keep every unit testable without it.

- Prompt fed via **stdin**, never argv — a prompt starting with `-` or `---`
  would otherwise be parsed as a CLI option (kaide learned this the hard way).
- Parses `stream-json` events; `result` event carries `session_id`, `is_error`,
  `duration_ms`, `total_cost_usd`, usage.
- Timeout kills the process group (`start_new_session=True`).
- **Stale-session retry:** non-zero exit while `--resume`-ing → retry once with
  no session id. This is the single most valuable line inherited from kaide.

### `sessions.py`

Maps a conversation prefix to a Claude Code session id.

```python
def fingerprint_chain(messages) -> list[str]   # rolling hash, one per prefix
def lookup(messages) -> tuple[str | None, int] # (session_id, prefix_len)
def remember(messages, reply, session_id) -> None
```

Chained hash: `fp[0] = sha256(canon(m0))`, `fp[i] = sha256(fp[i-1] + canon(m_i))`.
`canon` normalises `(role, text, tool name/id)` to stable JSON. Lookup walks the
chain longest-prefix-first; the first hit gives the session and how much of the
array it already knows. Persisted as `sessions.json` with atomic
tmp-write + `replace`, as kaide does. TTL-reaped.

A changed system prompt, a branch, or a regeneration simply misses and starts a
fresh session. That is correct, not a failure mode.

### `translate.py`

The genuinely new code, and the only fragile part.

```python
def to_prompt(delta_messages, tools, is_new_session) -> str
def contract_system_prompt(tools) -> str
def parse_reply(text, tools) -> ParsedReply   # .tool_calls | .content
```

Outbound: `role:"tool"` messages render as `TOOL RESULT (name): <content>`;
user messages pass through; on a new session the whole conversation is rendered
once, on a resume only the delta.

Inbound: parse a fenced or bare JSON object with a `tool_calls` array, then
**jsonschema-validate** each call's arguments against the tool that declared
them.

### `limits.py`

```python
session_lock(session_key)   # one in-flight run per conversation
global_slots = Semaphore(DTRELAY_MAX_CONCURRENT)   # default 2
```

Acquire session lock first, then a global slot, so a queued conversation cannot
hold a global slot while waiting on itself.

### `server.py`

HTTP and SSE only, no business logic. `/v1/chat/completions` (streaming and
non-streaming), `/v1/models` (advertises one model id, `claude-code`),
`/healthz`.

## Data flow, one call

1. `POST /v1/chat/completions {messages, tools, temperature, stream}`
2. `sessions.lookup(messages)` — longest known prefix of the full array
   - **hit** → `--resume sid`, delta = `messages[prefix_len:]`
   - **miss** → new session, render the full conversation
3. `translate.to_prompt(delta, tools, is_new)`
4. acquire session lock → acquire global slot
5. `runner.run(...)`, parsing `stream-json`
6. decide shape (below), emit SSE
7. `sessions.remember(messages, reply, sid)`

### Streaming vs. tool-call ambiguity

A tool call must never be streamed to the client as visible text. Because the
shape is only knowable from the first characters, the relay **buffers until it
can decide**: if the first non-whitespace characters are ` ``` ` or `{`, buffer
the whole reply and emit nothing until parsed; otherwise stream content deltas
live. Worst case for a prose reply that happens to start with `{` is a loss of
streaming, never a corrupted stream.

## Tool translation contract

Appended via `--append-system-prompt` on every run:

> You are the reasoning engine for an application. The TOOLS below are executed
> by the CALLER, never by you. You have no tools of your own. To call one, reply
> with ONLY a fenced json block and nothing else:
> ```json
> {"tool_calls":[{"name":"<tool>","arguments":{...}}]}
> ```
> Otherwise reply normally in prose.
>
> TOOLS:
> - `name(arg: type, ...) -> description`

Rendered from the OpenAI `tools` array on each request, since DeepTutor varies
the tool set per capability.

### Degradation ladder

| Condition | Response |
|---|---|
| Valid JSON, valid args | emit `tool_calls`, `finish_reason: "tool_calls"` |
| Malformed JSON | one corrective re-prompt in the same session |
| Args fail schema validation | one corrective re-prompt naming the violation |
| Unknown tool name | one corrective re-prompt listing valid names |
| Second failure | return the prose as a normal message |

A turn degrades to a plain answer; it never dies.

## Configuration

`.env`, mirroring kaide's conventions:

```
DTRELAY_PORT=8787
DTRELAY_CLAUDE_BIN=claude
DTRELAY_MODEL=                 # empty = CLI default
DTRELAY_TIMEOUT=900
DTRELAY_MAX_CONCURRENT=2
DTRELAY_CWD=                   # working dir for spawned sessions
DTRELAY_STATE_DIR=./state
DTRELAY_SESSION_TTL_HOURS=72
```

### DeepTutor side (UI only, no code)

1. Settings ▸ Catalog ▸ add a **Custom** connection:
   `base_url = http://127.0.0.1:8787/v1`, `api_key = dummy`
2. Refresh models, set `claude-code` active for the **llm** service.
3. Settings ▸ Partners & Agents: connect **Claude Code** as a subagent, and set
   `consult_budget` to 3–4.

## Error handling

| Failure | Handling |
|---|---|
| `claude` won't start | 502 with the spawn error |
| Non-zero exit while resuming | retry once with a fresh session |
| Non-zero exit, fresh session | 502 with stderr tail |
| Timeout | kill process group, 504 |
| Global slots exhausted | request queues until a slot frees; the caller's own HTTP timeout bounds the wait |
| Tool JSON unparseable twice | degrade to prose (see ladder) |

## Testing

**Unit.** Fingerprint chain: prefix hit, branch miss, regeneration miss, system
prompt change miss. Delta computation on a hit. Tool JSON parsing: fenced,
bare, prose-wrapped, malformed. Schema validation incl. unknown tool. Streaming
decision heuristic on `{`-leading prose.

**Integration.** A real DeepTutor turn through the relay, asserting: a KB tool
actually fires; turn 2 sends a delta rather than the full array (assert on the
prompt handed to `runner`); a `consult` tool call reaches the subagent layer and
spawns a tool-enabled session.

**Manual.** One full tutoring session with a knowledge base attached.

## Known limitations

- **Embeddings are not served.** Knowledge bases still need a real embedding
  provider; local Ollama is the no-key option. Chat, solve, research and
  delegation all run on the subscription.
- **Throttling is not globally coordinated.** `limits.py` sees relay traffic
  only; subagent consults are spawned by DeepTutor itself, outside the relay,
  and compete for the same plan window. Mitigated — not solved — by
  `DTRELAY_MAX_CONCURRENT=2` plus a low `consult_budget`. Patching
  site-packages is the only complete fix and is out of scope.
- **Fixed cost per call.** Every call carries Claude Code's full agent system
  prompt; a trivial probe billed $0.19. Notional on a subscription, but a
  12-round turn is roughly 12× that against the plan window.
- **Latency.** Agent cold boot per call, roughly 3–8s. Multi-round turns feel
  slow.
- **`tts` / `stt` / `imagegen`** are untouched and still need their own providers.

## Future work

Fold the relay into DeepTutor as a native `claude_code` ProviderSpec, which
would also let throttling cover subagent consults. Deliberately deferred: it
couples the work to a pip package that upgrades overwrite.
