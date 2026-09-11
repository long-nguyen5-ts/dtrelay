"""A tool-call payload must never be rendered as the assistant's message.

When a tool call cannot be salvaged (bad enum, unparseable payload) the relay
degrades to prose. Degrading must not mean pasting the raw JSON into the chat:
that is what the user sees as "the tutor replied with JSON".
"""
from dtrelay.translate import strip_payload

FENCED = '```json\n{"tool_calls":[{"name":"ask_user","arguments":{"q":1}}]}\n```'


def test_a_bare_payload_leaves_no_content():
    assert strip_payload(FENCED) == ""


def test_prose_around_a_payload_survives():
    text = f"Let me get set up.\n\n{FENCED}\n\nOne moment."
    out = strip_payload(text)
    assert "tool_calls" not in out
    assert "{" not in out
    assert "Let me get set up." in out
    assert "One moment." in out


def test_ordinary_prose_is_untouched():
    text = "Photosynthesis has two stages, the light reactions and the Calvin cycle."
    assert strip_payload(text) == text


def test_prose_with_an_unrelated_code_block_is_untouched():
    text = "The equation:\n\n```\n6CO2 + 6H2O -> C6H12O6\n```\n\nThat is it."
    assert strip_payload(text) == text


def test_an_unparseable_payload_is_still_removed():
    # The 21:17 failure: a payload that would not parse at all.
    text = '```json\n{"tool_calls":[{"name":"ask_user","arguments":{broken\n```'
    out = strip_payload(text)
    assert "tool_calls" not in out
    assert out == ""
