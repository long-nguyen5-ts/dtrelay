"""HTTP surface only - every decision lives in the modules this calls."""
from __future__ import annotations

import json
import logging
import os
import time
import uuid

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from dtrelay import runner as runner_mod
from dtrelay import translate
from dtrelay.config import MODEL_ID, Settings
from dtrelay.embeddings import EMBED_DIM, EMBED_MODEL_ID, embed_texts, normalize_input
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


def create_app(settings: Settings, runner=None) -> FastAPI:
    run = runner or runner_mod.run
    store = SessionStore(settings.state_dir / "sessions.json", settings.session_ttl_hours)
    limiter = Limiter(settings.max_concurrent)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        dropped = store.reap()
        if dropped:
            log.info("reaped %d expired session(s)", dropped)
        yield

    app = FastAPI(title="dtrelay", lifespan=lifespan)

    @app.get("/healthz")
    def healthz():
        return {"status": "ok", "in_flight": limiter.in_flight}

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [
            {"id": MODEL_ID, "object": "model", "owned_by": "dtrelay"},
            {"id": EMBED_MODEL_ID, "object": "model", "owned_by": "dtrelay"},
        ]}

    @app.post("/v1/embeddings")
    def embeddings(body: dict):
        """Local ONNX embeddings -- Claude Code cannot produce vectors.

        Without this, DeepTutor's knowledge bases would need an API key purely
        for indexing, which would defeat the point of the relay.
        """
        try:
            texts = normalize_input(body.get("input"))
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        try:
            vectors = embed_texts(texts)
        except Exception as exc:  # model load or inference failure
            log.error("embedding failed: %s", exc, exc_info=True)
            raise HTTPException(500, f"embedding failed: {exc}") from exc
        return {
            "object": "list",
            "model": EMBED_MODEL_ID,
            "data": [
                {"object": "embedding", "index": i, "embedding": v}
                for i, v in enumerate(vectors)
            ],
            "usage": {
                "prompt_tokens": sum(len(t.split()) for t in texts),
                "total_tokens": sum(len(t.split()) for t in texts),
            },
        }

    @app.post("/v1/chat/completions")
    def completions(body: dict):
        messages = body.get("messages") or []
        tools = body.get("tools") or []
        if not messages:
            raise HTTPException(400, "messages is required")

        if os.environ.get("DTRELAY_DUMP"):
            with open(os.environ["DTRELAY_DUMP"], "a") as fh:
                fh.write(json.dumps([
                    {"role": m.get("role"), "name": m.get("name"),
                     "tool_call_id": m.get("tool_call_id"),
                     "has_tool_calls": bool(m.get("tool_calls")),
                     "content": m.get("content")}
                    for m in messages
                ], default=str) + "\n")

        session_id, prefix_len = store.lookup(messages)
        delta = messages[prefix_len:] if session_id else messages
        if not delta:
            delta = messages[-1:]
        prompt = translate.to_prompt(delta)
        system_prompt = translate.contract_system_prompt(tools)
        # Serialise on the bound session when there is one; a brand-new
        # conversation has no contention to protect against yet.
        session_key = session_id or fingerprint_chain(messages)[-1]

        def _resolve(result):
            """Parse, and on a contract violation take one corrective retry."""
            parsed = translate.parse_reply(result.text, tools)
            if not parsed.error:
                return result, parsed
            log.info("corrective retry: %s", parsed.error)
            retry = run(settings, CORRECTION.format(error=parsed.error),
                        result.session_id, system_prompt)
            if retry.is_error:
                return result, translate.ParsedReply(content=result.text)
            reparsed = translate.parse_reply(retry.text, tools)
            if reparsed.error:
                return retry, translate.ParsedReply(content=retry.text)
            return retry, reparsed

        def _persist(result, parsed):
            """Record the prefix as the CLIENT will send it back next round.

            After a tool call DeepTutor echoes the assistant turn with empty
            content and a tool_calls array -- not the raw JSON the model
            emitted. Storing the raw JSON made every round miss and start a
            cold session. See tests/test_session_reuse.py.
            """
            if not result.session_id:
                return
            echoed = "" if parsed.tool_calls else (parsed.content or result.text)
            store.remember(
                messages + [{"role": "assistant", "content": echoed}],
                result.session_id,
            )

        if body.get("stream"):
            def generate():
                buffered: list[str] = []
                state = {"decided": False, "live": False, "stopped": False}
                queue: list[str] = []

                def on_delta(chunk: str):
                    """Forward text, but never let a tool-call payload reach the user.

                    The model may open with prose ("Let me look that up.") and
                    only then emit the fence, so the decision cannot be made
                    once from the first chunk -- it has to stay revisable.
                    Streaming stops the moment a payload starts; the prose
                    already sent reads as a status line, which is harmless.
                    """
                    buffered.append(chunk)
                    whole = "".join(buffered)
                    if state["stopped"]:
                        return
                    if translate.payload_has_begun(whole):
                        state["stopped"] = True
                        return
                    if not state["decided"]:
                        if not whole.strip():
                            return
                        state["decided"] = True
                        state["live"] = True
                        queue.append(whole)
                    else:
                        queue.append(chunk)

                with limiter.slot(session_key):
                    result = run(settings, prompt, session_id, system_prompt,
                                 on_delta=on_delta)
                    if result.is_error:
                        yield _sse(_chunk({"content": result.text}, "stop"))
                        yield "data: [DONE]\n\n"
                        return
                    result, parsed = _resolve(result)

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

                _persist(result, parsed)
                yield "data: [DONE]\n\n"

            return StreamingResponse(generate(), media_type="text/event-stream")

        with limiter.slot(session_key):
            result = run(settings, prompt, session_id, system_prompt)
            if result.is_error:
                raise HTTPException(502, result.text)
            result, parsed = _resolve(result)

        _persist(result, parsed)
        return _completion(parsed.content if parsed.tool_calls is None else None,
                           parsed.tool_calls)

    return app
