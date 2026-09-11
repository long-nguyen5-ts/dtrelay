# dtrelay — Claude Code as DeepTutor's model

An OpenAI-compatible daemon that makes DeepTutor run on a Claude subscription
with no API key. Each DeepTutor conversation is bound to a Claude Code session;
DeepTutor's tool schemas are translated into a JSON reply contract, and Claude
Code runs with its own tools disabled so it reasons instead of acts.

    DeepTutor ─► http://127.0.0.1:8787/v1/chat/completions ─► claude -p --resume <sid>
              ─► http://127.0.0.1:8787/v1/embeddings      ─► local ONNX (bge-small-en-v1.5)

## Run

    cp .env.example .env
    ./start.sh          # background, logs to bot.log
    ./run.sh            # foreground
    ./stop.sh

## Wire up DeepTutor

**llm** service — Custom binding, base_url `http://127.0.0.1:8787/v1`, key
`dummy`, model `claude-code`.

**embedding** service — Custom binding, base_url
`http://127.0.0.1:8787/v1/embeddings` (the FULL path, not the base — the
embedding adapter does not append it), key `dummy`, model
`bge-small-en-v1.5`, dim 384.

## Tests

    python3 -m pytest                                           # hermetic, fake CLI
    DTRELAY_SMOKE=1 python3 -m pytest tests/test_smoke_real.py  # real claude

## Limits

- Subagent consults spawn outside the relay and share the plan window; keep
  `consult_budget` low.
- A cold call costs ~$0.16 (the agent system prompt is written to cache);
  resumed calls cost ~$0.013. Long conversations are cheap, many short ones
  are not.
- Embeddings run locally on CPU and never touch the subscription.

Design: `docs/superpowers/specs/2026-09-11-claude-code-relay-design.md`
