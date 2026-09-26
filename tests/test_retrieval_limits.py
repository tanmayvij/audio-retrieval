from contextlib import nullcontext
from unittest.mock import Mock

import pytest

from app.services import retrieval


@pytest.fixture
def dependencies(monkeypatch):
    connection = object()
    rows = [{"chunk_id": str(index), "cosine_similarity": 1 - index / 100} for index in range(60)]
    semantic = Mock(side_effect=lambda connection, vector, limit: rows[:limit])
    lexical = Mock(side_effect=lambda connection, query, limit: rows[:limit])
    exact = Mock(side_effect=lambda connection, pattern, limit: rows[:limit])
    embed = Mock(return_value=[0.0] * 768)
    monkeypatch.setattr(retrieval, "connect", lambda: nullcontext(connection))
    monkeypatch.setattr(retrieval, "embed_query", embed)
    monkeypatch.setattr(retrieval, "semantic_candidates", semantic)
    monkeypatch.setattr(retrieval, "lexical_candidates", lexical)
    monkeypatch.setattr(retrieval, "exact_candidates", exact)
    return semantic, lexical, exact, embed


@pytest.mark.parametrize("mode", ["semantic", "hybrid", "exact"])
def test_limit_preserves_default_prefix(mode, dependencies):
    default = retrieval.search("question", mode)
    expanded = retrieval.search("question", mode, limit=5)
    assert len(default) == 3
    assert len(expanded) == 5
    assert default == expanded[:3]
    semantic, lexical, exact, embed = dependencies
    if mode == "exact":
        embed.assert_not_called()
        assert exact.call_args.args[-1] == 5
    else:
        assert semantic.call_args.args[-1] == 50
        if mode == "hybrid":
            assert lexical.call_args.args[-1] == 50


@pytest.mark.parametrize("mode", ["semantic", "hybrid"])
def test_candidate_depth_grows_with_limit(mode, dependencies):
    assert len(retrieval.search("question", mode, limit=60)) == 60
    semantic, lexical, _, _ = dependencies
    assert semantic.call_args.args[-1] == 60
    if mode == "hybrid":
        assert lexical.call_args.args[-1] == 60


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, "5"])
def test_invalid_limit_fails_before_model_or_database(limit, dependencies):
    with pytest.raises(ValueError, match="positive integer"):
        retrieval.search("question", "semantic", limit=limit)
    dependencies[3].assert_not_called()
