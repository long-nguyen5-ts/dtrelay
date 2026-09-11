import pytest

from dtrelay.embeddings import EMBED_DIM, EMBED_MODEL_ID, embed_texts, normalize_input


def test_normalize_accepts_a_bare_string():
    assert normalize_input("hello") == ["hello"]


def test_normalize_accepts_a_list():
    assert normalize_input(["a", "b"]) == ["a", "b"]


def test_normalize_rejects_empty():
    with pytest.raises(ValueError):
        normalize_input([])
    with pytest.raises(ValueError):
        normalize_input("")


def test_normalize_decodes_token_id_arrays():
    # OpenAI clients may send pre-tokenized input; we cannot detokenize, so
    # it must fail loudly rather than embed nonsense.
    with pytest.raises(ValueError):
        normalize_input([[1, 2, 3]])


@pytest.mark.slow
def test_embeddings_have_the_declared_dimension():
    vecs = embed_texts(["photosynthesis converts light into chemical energy"])
    assert len(vecs) == 1
    assert len(vecs[0]) == EMBED_DIM


@pytest.mark.slow
def test_semantically_close_texts_score_higher_than_distant_ones():
    a, b, c = embed_texts([
        "a cat sat on the mat",
        "a kitten rested on the rug",
        "quarterly revenue exceeded forecasts",
    ])

    def dot(u, v):
        return sum(x * y for x, y in zip(u, v))

    assert dot(a, b) > dot(a, c), "related sentences must embed closer than unrelated ones"


def test_model_id_is_stable():
    assert EMBED_MODEL_ID == "bge-small-en-v1.5"
