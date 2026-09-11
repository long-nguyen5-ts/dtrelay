"""Both failure modes of reply parsing, pinned together.

These are mirror images and a fix for one must not reintroduce the other:

  A. prose that merely CONTAINS a fence must not be read as a tool call
     (cost a ~60s corrective retry on every answer with a code block)
  B. a tool call wrapped in ANY prose must not be shown to the user
     (leaked raw JSON into the chat)
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

CALL = '{"tool_calls":[{"name":"search_kb","arguments":{"q":"x"}}]}'


def call_of(text):
    return parse_reply(text, TOOLS).tool_calls


# --- B: tool calls must be recognised however they are wrapped -------------

def test_clean_fenced_call():
    assert call_of(f"```json\n{CALL}\n```")


def test_bare_call():
    assert call_of(CALL)


def test_call_with_a_preamble():
    assert call_of(f"Let me look that up.\n\n```json\n{CALL}\n```")


def test_call_with_trailing_text():
    assert call_of(f"```json\n{CALL}\n```\n\nSearching now.")


def test_bare_call_embedded_in_a_sentence():
    assert call_of(f"Sure. {CALL}")


def test_unfenced_call_with_preamble():
    assert call_of(f"One moment.\n{CALL}")


# --- A: prose must stay prose ---------------------------------------------

def test_prose_with_a_code_fence_is_prose():
    txt = "Two stages:\n\n```\n6CO2 + 6H2O -> C6H12O6\n```\n\nThat is the net equation."
    r = parse_reply(txt, TOOLS)
    assert r.tool_calls is None and r.error is None
    assert r.content == txt


def test_prose_with_an_empty_fence_is_prose():
    txt = "Consider:\n\n```\n```\n\nDone."
    r = parse_reply(txt, TOOLS)
    assert r.tool_calls is None and r.error is None


def test_prose_with_an_unrelated_json_example_is_prose():
    txt = 'Configure it:\n\n```json\n{"model":"claude-code","stream":true}\n```\n\nThen save.'
    r = parse_reply(txt, TOOLS)
    assert r.tool_calls is None and r.error is None


def test_plain_prose_is_prose():
    r = parse_reply("Photosynthesis needs light to split water.", TOOLS)
    assert r.tool_calls is None and r.error is None


# --- malformed attempts still earn a corrective retry ---------------------

def test_malformed_tool_call_is_an_error():
    r = parse_reply('```json\n{"tool_calls":[{oops}]}\n```', TOOLS)
    assert r.error


def test_unknown_tool_is_an_error():
    r = parse_reply('{"tool_calls":[{"name":"nope","arguments":{}}]}', TOOLS)
    assert r.error


def test_bad_arguments_are_an_error():
    r = parse_reply('{"tool_calls":[{"name":"search_kb","arguments":{"q":9}}]}', TOOLS)
    assert r.error
