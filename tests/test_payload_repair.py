"""Salvage malformed tool-call payloads instead of degrading the turn.

Observed live while building a mastery path: a large ask_user payload came
back unparseable, the corrective retry produced another, and the user got the
fallback message. Repair is attempted ONLY when a tool_calls key is actually
present, so ordinary prose is never coerced into a tool call.
"""
from dtrelay.translate import parse_reply

TOOLS = [{
    "type": "function",
    "function": {
        "name": "ask_user",
        "parameters": {
            "type": "object",
            "properties": {
                "intro": {"type": "string"},
                "questions": {"type": "array", "items": {"type": "object"}},
            },
            "required": ["questions"],
        },
    },
}]

GOOD = '{"tool_calls":[{"name":"ask_user","arguments":{"questions":[{"prompt":"Level?"}]}}]}'


def test_a_well_formed_payload_still_parses():
    assert parse_reply(f"```json\n{GOOD}\n```", TOOLS).tool_calls


def test_a_truncated_payload_is_repaired():
    truncated = '```json\n{"tool_calls":[{"name":"ask_user","arguments":{"questions":[{"prompt":"How much do you know'
    r = parse_reply(truncated, TOOLS)
    assert r.tool_calls, f"should salvage, got error={r.error}"
    assert r.tool_calls[0]["function"]["name"] == "ask_user"


def test_a_payload_with_trailing_prose_parses():
    r = parse_reply(f"```json\n{GOOD}\n```\n\nLet me know!", TOOLS)
    assert r.tool_calls


def test_a_payload_with_a_trailing_comma_is_repaired():
    sloppy = '{"tool_calls":[{"name":"ask_user","arguments":{"questions":[{"prompt":"Level?"}],}}],}'
    assert parse_reply(sloppy, TOOLS).tool_calls


def test_prose_is_never_repaired_into_a_tool_call():
    for text in [
        "Photosynthesis has two stages.",
        "The set {a, b, c} is closed under the operation.",
        "Here is config:\n\n```json\n{\"model\":\"claude-code\"}\n```\n",
    ]:
        r = parse_reply(text, TOOLS)
        assert r.tool_calls is None, f"prose became a tool call: {text!r}"
        assert r.error is None, f"prose triggered a retry: {text!r}"


def test_prose_that_merely_mentions_tool_calls_stays_prose():
    # A tutoring answer explaining the relay itself must not be hijacked.
    text = "The relay converts the reply into tool_calls before DeepTutor sees it."
    r = parse_reply(text, TOOLS)
    assert r.tool_calls is None
    assert r.error is None
