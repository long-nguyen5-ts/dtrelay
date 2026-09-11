"""Regression: prose containing a code fence is NOT a tool call.

Observed live: a tutoring answer that included a fenced example tripped
_candidate(), which searched for a fence anywhere in the reply. The parse
failed, the relay burned a ~60s corrective retry, and the user waited twice as
long for the same answer.
"""
from dtrelay.translate import parse_reply

TOOLS = [{
    "type": "function",
    "function": {
        "name": "search_kb",
        "parameters": {"type": "object", "properties": {"q": {"type": "string"}},
                       "required": ["q"]},
    },
}]

ANSWER_WITH_CODE = """Photosynthesis has two stages. Here is the net equation:

```
6CO2 + 6H2O + light -> C6H12O6 + 6O2
```

The Calvin cycle then fixes the carbon."""

ANSWER_WITH_EMPTY_FENCE = "Consider this:\n\n```\n```\n\nThat is the shape of it."

ANSWER_WITH_JSON_EXAMPLE = """You can configure it like this:

```json
{"model": "claude-code", "stream": true}
```

Set that in the catalog."""

REAL_CALL = '```json\n{"tool_calls":[{"name":"search_kb","arguments":{"q":"x"}}]}\n```'
REAL_CALL_BARE = '{"tool_calls":[{"name":"search_kb","arguments":{"q":"x"}}]}'


def test_prose_containing_a_code_fence_is_content():
    r = parse_reply(ANSWER_WITH_CODE, TOOLS)
    assert r.error is None, "must not trigger a corrective retry"
    assert r.tool_calls is None
    assert r.content == ANSWER_WITH_CODE


def test_prose_with_an_empty_fence_is_content():
    r = parse_reply(ANSWER_WITH_EMPTY_FENCE, TOOLS)
    assert r.error is None
    assert r.content == ANSWER_WITH_EMPTY_FENCE


def test_prose_containing_a_json_example_is_content():
    # The fence holds valid JSON, but it is not a tool_calls payload and the
    # reply is clearly prose.
    r = parse_reply(ANSWER_WITH_JSON_EXAMPLE, TOOLS)
    assert r.error is None
    assert r.tool_calls is None


def test_a_real_fenced_tool_call_still_parses():
    assert parse_reply(REAL_CALL, TOOLS).tool_calls is not None


def test_a_real_bare_tool_call_still_parses():
    assert parse_reply(REAL_CALL_BARE, TOOLS).tool_calls is not None


def test_a_whole_reply_that_is_a_broken_fence_still_errors():
    # Reply is nothing but a fence, so it WAS trying to call a tool.
    r = parse_reply('```json\n{"tool_calls":[{oops}]}\n```', TOOLS)
    assert r.error
