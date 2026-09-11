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


def payload_has_begun(text: str) -> bool:
    """Has a tool-call payload started in the text streamed so far?

    Used to cut streaming off mid-reply, so a payload never reaches the user
    even when the model prefaces it with prose.
    """
    stripped = text.lstrip()
    return (
        stripped.startswith("```")
        or stripped.startswith("{")
        or "```" in text
        or "tool_calls" in text
    )


def _json_candidates(text: str) -> list[str]:
    """Every substring of a reply that might be the tool-call payload.

    Position is not the signal -- content is. The model wraps calls in prose
    ("Let me look that up." before the fence) and it writes code fences inside
    ordinary answers, so neither "starts with {" nor "is entirely a fence"
    separates the two. We gather every plausible payload and let the presence
    of a tool_calls key decide. See tests/test_reply_shapes.py for both
    failure modes this balances.
    """
    out: list[str] = []
    out.extend(_FENCE.findall(text))
    stripped = text.strip()
    if stripped:
        out.append(stripped)
    first, last = text.find("{"), text.rfind("}")
    if first != -1 and last > first:
        out.append(text[first:last + 1])
    return out


def _extract_calls(text: str) -> list | None:
    """The tool_calls array from a reply, or None if this reply is prose."""
    for raw in _json_candidates(text):
        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(payload, dict):
            calls = payload.get("tool_calls")
            if isinstance(calls, list) and calls:
                return calls
    return None


def _looks_like_a_failed_attempt(text: str) -> bool:
    """Did the model TRY to call a tool and botch the JSON?

    Only then is a corrective retry worth a round trip. Ordinary prose that
    happens to contain a fence must never trigger one.
    """
    return "tool_calls" in text


def parse_reply(text: str, tools: list[dict]) -> ParsedReply:
    calls = _extract_calls(text)
    if calls is None:
        if _looks_like_a_failed_attempt(text):
            return ParsedReply(error="a tool_calls payload was present but unparseable")
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
