"""Opt-in: exercises the REAL claude CLI. Run with:
    DTRELAY_SMOKE=1 python3 -m pytest tests/test_smoke_real.py -v -s
"""
import os

import pytest

from dtrelay.config import load_settings
from dtrelay.runner import run
from dtrelay.translate import contract_system_prompt, parse_reply

pytestmark = pytest.mark.skipif(
    not os.environ.get("DTRELAY_SMOKE"), reason="set DTRELAY_SMOKE=1 to run"
)

TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}]


def test_real_cli_emits_a_valid_tool_call():
    s = load_settings()
    result = run(s, "What is the weather in Hanoi? Use the tool.", None,
                 contract_system_prompt(TOOLS))
    assert not result.is_error, result.text
    parsed = parse_reply(result.text, TOOLS)
    assert parsed.error is None, f"{parsed.error}\nraw: {result.text}"
    assert parsed.tool_calls[0]["function"]["name"] == "get_weather"
    assert result.session_id


def test_real_cli_resumes_with_a_delta():
    s = load_settings()
    first = run(s, "Remember the number 41. Reply OK.", None,
                contract_system_prompt([]))
    assert not first.is_error, first.text
    second = run(s, "Add one to the number you remembered. Reply with digits only.",
                 first.session_id, contract_system_prompt([]))
    assert not second.is_error, second.text
    assert "42" in second.text
