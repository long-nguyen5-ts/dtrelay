"""Stand-in for the `claude` binary: emits stream-json without touching a model.

Behaviour is driven by env vars so one script covers every test case:
  FAKE_RESULT     text the result event carries
  FAKE_SESSION    session id to report
  FAKE_EXIT       process exit code
  FAKE_IS_ERROR   "1" to set is_error on the result event
  FAKE_DELTAS     "|"-separated partial text chunks emitted before the result
  FAKE_ECHO_ARGV  path to write argv to, so tests can assert on flags
  FAKE_ECHO_STDIN path to write the received prompt to
"""
import json
import os
import sys


def main():
    argv_path = os.environ.get("FAKE_ECHO_ARGV")
    if argv_path:
        with open(argv_path, "w") as f:
            f.write("\n".join(sys.argv[1:]))
    prompt = sys.stdin.read()
    stdin_path = os.environ.get("FAKE_ECHO_STDIN")
    if stdin_path:
        with open(stdin_path, "w") as f:
            f.write(prompt)

    exit_code = int(os.environ.get("FAKE_EXIT", "0"))
    if exit_code:
        sys.stderr.write("fake claude failure\n")
        sys.exit(exit_code)

    for chunk in filter(None, os.environ.get("FAKE_DELTAS", "").split("|")):
        print(json.dumps({
            "type": "stream_event",
            "event": {
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": chunk},
            },
        }), flush=True)

    print(json.dumps({
        "type": "result",
        "result": os.environ.get("FAKE_RESULT", "ok"),
        "session_id": os.environ.get("FAKE_SESSION", "sid_fake"),
        "is_error": os.environ.get("FAKE_IS_ERROR") == "1",
        "duration_ms": 12,
        "num_turns": 1,
        "total_cost_usd": 0.01,
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }), flush=True)


if __name__ == "__main__":
    main()
