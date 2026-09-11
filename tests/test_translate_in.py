from dtrelay.translate import looks_like_json, parse_reply

TOOLS = [{
    "type": "function",
    "function": {
        "name": "search_kb",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
}]

FENCED = '```json\n{"tool_calls":[{"name":"search_kb","arguments":{"query":"x"}}]}\n```'
BARE = '{"tool_calls":[{"name":"search_kb","arguments":{"query":"x"}}]}'


def test_parses_a_fenced_tool_call():
    r = parse_reply(FENCED, TOOLS)
    assert r.error is None
    assert r.tool_calls[0]["function"]["name"] == "search_kb"


def test_arguments_are_a_json_encoded_string():
    r = parse_reply(FENCED, TOOLS)
    assert r.tool_calls[0]["function"]["arguments"] == '{"query": "x"}'


def test_parses_a_bare_tool_call():
    assert parse_reply(BARE, TOOLS).tool_calls is not None


def test_each_call_gets_a_unique_id():
    r = parse_reply(
        '{"tool_calls":[{"name":"search_kb","arguments":{"query":"a"}},'
        '{"name":"search_kb","arguments":{"query":"b"}}]}',
        TOOLS,
    )
    ids = [c["id"] for c in r.tool_calls]
    assert len(set(ids)) == 2
    assert all(i.startswith("call_") for i in ids)


def test_prose_is_returned_as_content():
    r = parse_reply("The answer is 4.", TOOLS)
    assert r.tool_calls is None
    assert r.content == "The answer is 4."


def test_unknown_tool_name_is_an_error():
    r = parse_reply('{"tool_calls":[{"name":"nope","arguments":{}}]}', TOOLS)
    assert r.tool_calls is None
    assert "nope" in r.error


def test_arguments_failing_schema_are_an_error():
    r = parse_reply(
        '{"tool_calls":[{"name":"search_kb","arguments":{"query":123}}]}', TOOLS
    )
    assert r.tool_calls is None
    assert r.error


def test_missing_required_argument_is_an_error():
    r = parse_reply('{"tool_calls":[{"name":"search_kb","arguments":{}}]}', TOOLS)
    assert r.tool_calls is None
    assert r.error


def test_malformed_json_in_a_fence_is_an_error():
    r = parse_reply('```json\n{"tool_calls":[{oops}]}\n```', TOOLS)
    assert r.tool_calls is None
    assert r.error


def test_prose_that_merely_mentions_json_is_not_a_tool_call():
    r = parse_reply("You could call search_kb with a query.", TOOLS)
    assert r.tool_calls is None
    assert r.error is None


def test_looks_like_json_detects_fence_and_brace():
    assert looks_like_json("```json")
    assert looks_like_json("  {")
    assert not looks_like_json("The answer")
