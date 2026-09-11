# Running DeepTutor on a Claude subscription

A step-by-step guide to pointing DeepTutor at dtrelay so it reasons through
your Claude Code subscription instead of an API key.

**Nothing in the DeepTutor package is modified.** Every change below is either
a setting in DeepTutor's own UI or a call to its own API. A `pip install -U
deeptutor` will not undo any of it.

Verified against DeepTutor **1.6.7** and claude CLI **2.1.268**.

---

## What you get, and what you don't

| | Runs on the subscription |
|---|---|
| Chat, solve, research, question | yes |
| Knowledge bases (indexing + retrieval) | yes — embeddings run locally on CPU |
| Delegation to a tool-enabled agent | yes |
| Image/vision input | **no** — silently dropped (see Limitations) |
| `tts` / `stt` / `imagegen` | no — unaffected, still need their own providers |

---

## 1. Prerequisites

```bash
claude --version                    # the CLI must be installed and logged in
echo "${ANTHROPIC_API_KEY:-unset}"  # MUST print: unset
```

If `ANTHROPIC_API_KEY` is set, the spawned `claude` bills that key instead of
your subscription. Unset it in the shell that starts the relay.

Install the relay's dependencies:

```bash
cd /path/to/dtrelay
python3 -m pip install -r requirements.txt
```

`fastembed` pulls ~50 MB of onnxruntime; the embedding model (~130 MB) is
downloaded lazily on the first embeddings request, not at startup.

---

## 2. Start the relay

```bash
cp .env.example .env      # defaults are fine
./start.sh                # background, logs to bot.log
curl -s http://127.0.0.1:8787/healthz
# {"status":"ok","in_flight":0}
```

`start.sh` is idempotent — safe to run twice, does nothing if already up.
It does **not** survive a reboot; add a watchdog if you want that:

```cron
@reboot     /path/to/dtrelay/start.sh
*/5 * * * * /path/to/dtrelay/start.sh
```

---

## 3. Point DeepTutor's `llm` service at the relay

### Option A — the UI

**Settings ▸ Catalog**, add a connection:

| Field | Value |
|---|---|
| Provider | **Custom** |
| Base URL | `http://127.0.0.1:8787/v1` |
| API key | `dummy` (required by the form; never used) |

Refresh models, then set **`claude-code`** active for the **llm** service.

### Option B — the API

`POST /api/settings/apply/service` commits one service without disturbing the
others. This is safer than a whole-catalog `PUT`:

```bash
curl -s -X POST http://127.0.0.1:8001/api/settings/apply/service \
  -H 'Content-Type: application/json' -d '{
  "service": "llm",
  "config": {
    "active_profile_id": "llm-profile-dtrelay",
    "active_model_id": "llm-model-claudecode",
    "profiles": [{
      "id": "llm-profile-dtrelay",
      "name": "dtrelay (Claude Code)",
      "binding": "custom",
      "base_url": "http://127.0.0.1:8787/v1",
      "api_key": "dummy",
      "api_version": "",
      "extra_headers": {},
      "models": [{"id": "llm-model-claudecode", "model": "claude-code", "name": "Claude Code"}]
    }]
  }
}'
```

Why `custom` works: DeepTutor's `custom` provider spec is `is_direct=True`,
meaning the operator supplies base URL and key with no auto-detection. That is
the seam the whole design hangs on.

---

## 4. Point the `embedding` service at the relay

Claude Code cannot produce embedding vectors at any price — there is no
interface for it. The relay serves them from a local ONNX model instead
(`BAAI/bge-small-en-v1.5`, 384 dimensions, CPU only), so knowledge bases work
with no API key.

> **The one real gotcha.** The embedding service needs the **full endpoint
> URL**, not the `/v1` base. DeepTutor's embedding adapter does not append
> `/embeddings` the way the chat adapter does — give it the base and every
> request 404s.

```bash
curl -s -X POST http://127.0.0.1:8001/api/settings/apply/service \
  -H 'Content-Type: application/json' -d '{
  "service": "embedding",
  "config": {
    "active_profile_id": "emb-profile-dtrelay",
    "active_model_id": "emb-model-bge",
    "profiles": [{
      "id": "emb-profile-dtrelay",
      "name": "dtrelay (local ONNX)",
      "binding": "custom",
      "base_url": "http://127.0.0.1:8787/v1/embeddings",
      "api_key": "dummy",
      "api_version": "",
      "extra_headers": {},
      "models": [{"id": "emb-model-bge", "model": "bge-small-en-v1.5", "name": "bge-small-en-v1.5", "dim": 384}]
    }]
  }
}'
```

---

## 5. Connect the delegation subagent (optional)

This is the second, separate use of Claude Code: the relay runs it as a
*reasoner with tools off*, while the subagent layer runs it as a *full agent
with tools on*, which DeepTutor delegates to mid-turn.

```bash
curl -s -X POST http://127.0.0.1:8001/api/subagents/connections \
  -H 'Content-Type: application/json' \
  -d '{"name":"Claude Code","agent_kind":"claude_code","cwd":""}'

# Keep the budget low: these runs spawn OUTSIDE the relay and its limiter
# cannot see them, so both paths compete for one plan window.
curl -s -X PUT http://127.0.0.1:8001/api/subagents/settings \
  -H 'Content-Type: application/json' -d '{"consult_budget":3}'
```

---

## 6. Verify

```bash
curl -s -X POST http://127.0.0.1:8001/api/system/test/llm -d '{}' \
  -H 'Content-Type: application/json'
# {"success": true, "model": "claude-code", ...}

curl -s -X POST http://127.0.0.1:8001/api/system/test/embeddings -d '{}' \
  -H 'Content-Type: application/json'
# {"success": true, "model": "bge-small-en-v1.5", ...}
```

Then send a message in the UI. A plain turn should start streaming text in
about five seconds.

---

## 7. Troubleshooting

**`No active LLM model is configured`**
The catalog was never applied, or `active_model_id` is null. Check:
```bash
curl -s http://127.0.0.1:8001/api/settings/catalog | python3 -c \
 "import json,sys; s=json.load(sys.stdin)['catalog']['services']; print(s['llm']['active_model_id'], s['embedding']['active_model_id'])"
```

**`HTTP 404 from http://127.0.0.1:8787/v1`** on embeddings
The embedding `base_url` is missing `/embeddings`. See step 4.

**`Session already has an active or recovering turn`**
Not a bug, and **do not raise `lease_ttl_seconds` to fix it** — that makes it
worse, not better.

The error is raised when `coordinator.acquire_turn()` returns `None`, meaning
the previous turn still holds the per-session lease (keyed
`{scope}:{session_id}`). A turn keeps that lease through finalization, which
includes the background title-generation turn, so the session stays locked for
a few seconds after you receive `done`.

Measured on this setup: a follow-up was accepted **6 s** after the previous
turn's `done`. The lease is released normally; it is not expiring, so a longer
TTL would only extend the window in the cases where it *is* held.

The fix is client-side: retry with a short backoff rather than sending the next
turn instantly. DeepTutor's own UI does this. Only a hand-rolled WebSocket
client hits the error.

Note the lease is renewed by a separate task every `min(10, ttl/3)` seconds
while a turn runs, so a slow turn does not lose its lease on a healthy event
loop — a 70 s turn is renewed seven times.

**The turn hangs and the whole API stops responding**
Seen after a series of long turns: 0 % CPU, no connections to the relay, even
trivial reads time out. Restart the DeepTutor backend.

**A turn that calls a tool shows nothing for a long time**
Expected. The relay never streams a tool-call payload, and there is no prose to
show in its place, so a knowledge-base question pauses and then answers.

**Check what the relay is doing**
```bash
tail -f bot.log            # "claude run (session=new)" vs "(session=<id>)"
                           # <id> means the session resumed — history not resent
```

---

## 8. Limitations you should know about

The relay reads only `messages`, `tools` and `stream` off each request.
Everything else DeepTutor sends is discarded **without an error**:

| Field | Consequence |
|---|---|
| `temperature` | ignored — the per-capability values in `agents.yaml` are inert |
| `max_tokens` | ignored — no ceiling is enforced |
| `tool_choice` | ignored — a forced tool call becomes a request, not a guarantee |
| `response_format` | ignored — JSON mode is not enforced |
| image parts | **dropped silently** — attach an image and it vanishes |
| `usage` (response) | never returned — DeepTutor cannot track token cost |

`temperature` and `max_tokens` have no equivalent on the Claude Code CLI; only
a real API key restores them.

**Cost shape.** A cold call costs roughly $0.16 in notional terms (the agent
system prompt is written to cache); a resumed call about $0.013. Long
conversations are cheap; many short ones are not.

---

## 9. Reverting

There is nothing to uninstall. Point the `llm` and `embedding` services back at
a real provider in **Settings ▸ Catalog**, and stop the relay:

```bash
./stop.sh
```
