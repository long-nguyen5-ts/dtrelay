from fastapi.testclient import TestClient

from dtrelay.config import Settings
from dtrelay.embeddings import EMBED_DIM
from dtrelay.server import create_app
from tests.test_server import FakeRunner

import pytest


def client(tmp_path):
    return TestClient(create_app(Settings(state_dir=tmp_path), runner=FakeRunner()))


def test_models_lists_both_services(tmp_path):
    ids = [m["id"] for m in client(tmp_path).get("/v1/models").json()["data"]]
    assert "claude-code" in ids
    assert "bge-small-en-v1.5" in ids


def test_empty_input_is_a_400(tmp_path):
    r = client(tmp_path).post("/v1/embeddings", json={"input": []})
    assert r.status_code == 400


def test_token_array_input_is_a_400(tmp_path):
    r = client(tmp_path).post("/v1/embeddings", json={"input": [[1, 2, 3]]})
    assert r.status_code == 400


@pytest.mark.slow
def test_batch_returns_one_indexed_vector_per_input(tmp_path):
    r = client(tmp_path).post("/v1/embeddings", json={"input": ["alpha", "beta", "gamma"]})
    body = r.json()
    assert r.status_code == 200
    assert body["object"] == "list"
    assert [d["index"] for d in body["data"]] == [0, 1, 2]
    assert all(len(d["embedding"]) == EMBED_DIM for d in body["data"])
