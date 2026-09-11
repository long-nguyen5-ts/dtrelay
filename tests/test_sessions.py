from dtrelay.sessions import SessionStore, canon, fingerprint_chain


def M(role, content, **kw):
    return {"role": role, "content": content, **kw}


def test_canon_ignores_irrelevant_keys():
    assert canon(M("user", "hi")) == canon(M("user", "hi", extra="ignored"))


def test_canon_distinguishes_role_and_text():
    assert canon(M("user", "hi")) != canon(M("assistant", "hi"))
    assert canon(M("user", "hi")) != canon(M("user", "ho"))


def test_chain_is_prefix_stable():
    a = [M("system", "s"), M("user", "u1")]
    b = a + [M("assistant", "a1"), M("user", "u2")]
    assert fingerprint_chain(b)[:2] == fingerprint_chain(a)


def test_lookup_returns_longest_known_prefix(tmp_path):
    store = SessionStore(tmp_path / "s.json", ttl_hours=72)
    convo = [M("system", "s"), M("user", "u1"), M("assistant", "a1")]
    store.remember(convo, "sid_A")

    sid, n = store.lookup(convo + [M("user", "u2")])
    assert sid == "sid_A"
    assert n == 3


def test_lookup_misses_on_a_branch(tmp_path):
    store = SessionStore(tmp_path / "s.json", ttl_hours=72)
    store.remember([M("system", "s"), M("user", "u1")], "sid_A")

    sid, n = store.lookup([M("system", "s"), M("user", "DIFFERENT")])
    assert sid is None
    assert n == 0


def test_lookup_misses_when_system_prompt_changes(tmp_path):
    store = SessionStore(tmp_path / "s.json", ttl_hours=72)
    store.remember([M("system", "s"), M("user", "u1")], "sid_A")

    sid, _ = store.lookup([M("system", "DIFFERENT"), M("user", "u1")])
    assert sid is None


def test_remember_survives_reload(tmp_path):
    p = tmp_path / "s.json"
    convo = [M("user", "u1")]
    SessionStore(p, ttl_hours=72).remember(convo, "sid_A")

    sid, n = SessionStore(p, ttl_hours=72).lookup(convo)
    assert (sid, n) == ("sid_A", 1)


def test_reap_drops_expired_entries(tmp_path):
    p = tmp_path / "s.json"
    store = SessionStore(p, ttl_hours=0)
    store.remember([M("user", "u1")], "sid_A")
    assert store.reap() == 1
    assert store.lookup([M("user", "u1")]) == (None, 0)
