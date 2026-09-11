from dtrelay.translate import contract_system_prompt, render_messages, to_prompt

TOOLS = [{
    "type": "function",
    "function": {
        "name": "search_kb",
        "description": "Search the knowledge base",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
}]


def test_contract_names_the_tool_and_the_json_shape():
    p = contract_system_prompt(TOOLS)
    assert "search_kb" in p
    assert "Search the knowledge base" in p
    assert '"tool_calls"' in p
    assert "executed by the CALLER" in p


def test_contract_without_tools_forbids_tool_calls():
    p = contract_system_prompt([])
    assert "no tools are available" in p.lower()


def test_tool_result_messages_render_with_their_name():
    out = render_messages([
        {"role": "tool", "name": "search_kb", "content": "3 hits"},
    ])
    assert "TOOL RESULT (search_kb): 3 hits" in out


def test_user_and_assistant_turns_are_labelled():
    out = render_messages([
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ])
    assert "USER: hello" in out
    assert "ASSISTANT: hi" in out


def test_system_messages_render_as_instructions():
    out = render_messages([{"role": "system", "content": "be terse"}])
    assert "INSTRUCTIONS: be terse" in out


def test_to_prompt_strips_a_lone_user_turn_to_bare_text():
    assert to_prompt([{"role": "user", "content": "what is 2+2?"}]) == "what is 2+2?"


def test_to_prompt_labels_multi_message_deltas():
    out = to_prompt([
        {"role": "tool", "name": "search_kb", "content": "3 hits"},
        {"role": "user", "content": "summarise"},
    ])
    assert "TOOL RESULT (search_kb): 3 hits" in out
    assert "USER: summarise" in out
