from dtrelay.stream_gate import HOLDBACK, StreamGate


def feed(gate, text, size=5):
    out = []
    for i in range(0, len(text), size):
        out.append(gate.accept(text[i:i + size]))
    return "".join(out)


def test_short_text_is_held_back_entirely():
    # Nothing is emitted until there is more than the holdback window, so a
    # payload opener can never escape before it is recognised.
    gate = StreamGate()
    assert feed(gate, "short") == ""


def test_long_prose_streams_progressively():
    gate = StreamGate()
    text = "Photosynthesis converts light into chemical energy. " * 4
    streamed = feed(gate, text)
    assert streamed, "long prose must emit before the run finishes"
    assert text.startswith(streamed)
    assert len(streamed) == len(text) - HOLDBACK


def test_flush_returns_exactly_the_remainder():
    gate = StreamGate()
    text = "Photosynthesis converts light into chemical energy. " * 4
    streamed = feed(gate, text)
    assert streamed + gate.flush(text) == text, "no loss, no duplication"


def test_a_payload_stops_emission():
    gate = StreamGate()
    payload = '{"tool_calls":[{"name":"search_kb","arguments":{"q":"x"}}]}'
    assert feed(gate, payload) == ""
    assert gate.stopped


def test_a_preamble_then_payload_never_emits_the_payload():
    gate = StreamGate()
    streamed = feed(gate, 'Let me look that up.\n\n```json\n{"tool_calls":[{"name":"s"}]}\n```')
    assert "tool_calls" not in streamed
    assert "{" not in streamed
    assert "`" not in streamed


def test_a_long_preamble_emits_prose_but_stops_at_the_fence():
    gate = StreamGate()
    preamble = "I need to check your notes before answering that question properly. "
    streamed = feed(gate, preamble + '```json\n{"tool_calls":[]}\n```')
    assert streamed
    assert preamble.startswith(streamed), "only prose, and only a prefix of it"
    assert "`" not in streamed


def test_prose_containing_a_fence_stops_but_flush_repairs_it():
    # A code block in an answer halts streaming (it might be a payload), and
    # flush must then deliver the rest without duplicating what was sent.
    gate = StreamGate()
    text = "The net equation is shown here in full detail below:\n\n```\n6CO2\n```\n\nDone."
    streamed = feed(gate, text)
    assert streamed + gate.flush(text) == text
