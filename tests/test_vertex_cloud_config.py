"""Cloud Vertex branch regression tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError


def test_embedding_normalization_unit_length():
    from src.embeddings import _l2_normalize

    vector = _l2_normalize([3.0, 4.0])
    assert vector == [0.6, 0.8]


def test_embedding_dimension_rejects_unsupported_value():
    from src.config import Settings

    with pytest.raises(ValidationError):
        Settings(EMBEDDING_DIMENSION=1024)
