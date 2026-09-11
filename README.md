# dtrelay — Claude Code as DeepTutor's model

An OpenAI-compatible daemon that makes DeepTutor run on a Claude subscription
with no API key. Each DeepTutor conversation is bound to a Claude Code session;
DeepTutor's tool schemas are translated into a JSON reply contract, and Claude
Code runs with its own tools disabled so it reasons instead of acts.

    DeepTutor ─► http://127.0.0.1:8787/v1 ─► claude -p --resume <sid>

## Run

    cp .env.example .env
    ./start.sh          # background, logs to bot.log
    ./run.sh            # foreground
    ./stop.sh

## Wire up DeepTutor

Settings ▸ Catalog ▸ Custom: base_url `http://127.0.0.1:8787/v1`, key `dummy`.
Set `claude-code` active for the **llm** service.

## Tests

    python3 -m pytest                                           # hermetic, fake CLI
    DTRELAY_SMOKE=1 python3 -m pytest tests/test_smoke_real.py  # real claude

## Limits

- Embeddings are not served — knowledge bases need Ollama or an API key.
- Subagent consults spawn outside the relay and share the plan window; keep
  `consult_budget` low.
- Each call carries Claude Code's agent system prompt, so cost per call is
  roughly flat regardless of prompt size.

Design: `docs/superpowers/specs/2026-09-11-claude-code-relay-design.md`
