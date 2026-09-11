"""Per-conversation serialisation plus a global cap.

One subscription backs every run, so the global semaphore is what keeps a
fan-out turn from melting the plan window. The session lock is acquired FIRST
so a queued conversation never holds a global slot while waiting on itself.
"""
from __future__ import annotations

import threading
from contextlib import contextmanager


class Limiter:
    def __init__(self, max_concurrent: int):
        self._sem = threading.Semaphore(max(1, int(max_concurrent)))
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()
        self._in_flight = 0

    @property
    def in_flight(self) -> int:
        return self._in_flight

    def _lock_for(self, key: str) -> threading.Lock:
        with self._guard:
            return self._locks.setdefault(key, threading.Lock())

    @contextmanager
    def slot(self, session_key: str):
        lock = self._lock_for(session_key)
        lock.acquire()
        try:
            self._sem.acquire()
            with self._guard:
                self._in_flight += 1
            try:
                yield
            finally:
                with self._guard:
                    self._in_flight -= 1
                self._sem.release()
        finally:
            lock.release()
