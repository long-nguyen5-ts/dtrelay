import threading
import time

from dtrelay.limits import Limiter


def test_same_session_runs_are_serialised():
    limiter = Limiter(max_concurrent=4)
    order = []

    def work(tag, hold):
        with limiter.slot("sess_A"):
            order.append(f"{tag}-in")
            time.sleep(hold)
            order.append(f"{tag}-out")

    t1 = threading.Thread(target=work, args=("a", 0.2))
    t2 = threading.Thread(target=work, args=("b", 0.0))
    t1.start(); time.sleep(0.05); t2.start()
    t1.join(); t2.join()
    assert order == ["a-in", "a-out", "b-in", "b-out"]


def test_global_cap_limits_distinct_sessions():
    limiter = Limiter(max_concurrent=2)
    peak = {"v": 0}
    lock = threading.Lock()

    def work(i):
        with limiter.slot(f"sess_{i}"):
            with lock:
                peak["v"] = max(peak["v"], limiter.in_flight)
            time.sleep(0.1)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(6)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert peak["v"] <= 2


def test_slot_is_released_on_exception():
    limiter = Limiter(max_concurrent=1)
    try:
        with limiter.slot("sess_A"):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert limiter.in_flight == 0
    with limiter.slot("sess_A"):
        assert limiter.in_flight == 1
