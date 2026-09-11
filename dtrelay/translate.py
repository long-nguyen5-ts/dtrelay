"""OpenAI wire format <-> Claude Code prompt/reply.

Outbound: DeepTutor's tool schemas become an instruction block, and messages
become labelled plain text. Inbound: a JSON reply becomes OpenAI tool_calls.

This is the only fragile part of the relay, so every failure degrades to prose
rather than erroring the turn.
"""
from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass

import jsonschema

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


# --- inbound ---------------------------------------------------------------

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
    """The JSON payload of a tool-call reply, or None if this is prose.

    The whole reply must BE the payload -- a lone fenced block, or a bare JSON
    object. A fence merely appearing inside prose is a code example, not a tool
    call: searching anywhere made every tutoring answer that contained a code
    block burn a corrective retry. See tests/test_prose_with_fences.py.
    """
    stripped = text.strip()
    m = _FENCE.fullmatch(stripped)
    if m:
        return m.group(1)
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    return None


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
