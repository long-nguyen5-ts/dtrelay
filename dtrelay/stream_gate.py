"""Decides, chunk by chunk, what may be forwarded to the client.

Streaming is one-way: text already sent cannot be retracted. A tool-call
payload must therefore never be emitted even by a fragment, while ordinary
prose should appear as it is produced -- a tutoring answer can take a minute,
and a spinner for that long is its own defect.

The gate holds back a trailing window so a payload opener is recognised before
it could be forwarded, and stops permanently once one is seen.
"""
from __future__ import annotations

from dtrelay.translate import payload_has_begun

# Wide enough to cover an opening fence plus the start of a tool_calls key
# before either could be emitted.
HOLDBACK = 48


class StreamGate:
    def __init__(self, holdback: int = HOLDBACK):
        self.holdback = holdback
        self.buf = ""
        self.sent = ""
        self.stopped = False

    def accept(self, chunk: str) -> str:
        """Take a delta; return the text that is safe to emit now (may be "")."""
        self.buf += chunk
        if self.stopped:
            return ""
        if payload_has_begun(self.buf):
            self.stopped = True
            return ""
        safe = self.buf[: -self.holdback] if len(self.buf) > self.holdback else ""
        if len(safe) <= len(self.sent):
            return ""
        out = safe[len(self.sent):]
        self.sent = safe
        return out

    def flush(self, final_text: str) -> str:
        """The remainder to emit once the reply is known to be prose.

        When the final text continues what was streamed, only the tail is
        returned. After a corrective retry the text is replaced wholesale and
        no longer extends what was sent; the full text is returned then, which
        may repeat a short preamble but never drops content.
        """
        if final_text.startswith(self.sent):
            return final_text[len(self.sent):]
        return final_text
