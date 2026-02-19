import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from MVP.backend.chroma_db.embedder import embed_documents


# -------------------------
# Test: normal input
# -------------------------
@patch("MVP.backend.vector_store.embedder.get_embedding_model")
def test_embed_documents_normal(mock_get_model):
    mock_model = MagicMock()

    mock_model.encode.return_value = np.array([
        [0.1] * 384,
        [0.2] * 384
    ], dtype=np.float32)

    mock_get_model.return_value = mock_model

    chunks = ["Hello world", "Another chunk"]

    result = embed_documents(chunks)

    assert len(result) == 2
    assert len(result[0]) == 384
    mock_model.encode.assert_called_once()


# -------------------------
# Test: empty input
# -------------------------
def test_embed_documents_empty():
    result = embed_documents([])
    assert result == []


# -------------------------
# Test: filtering empty chunks
# -------------------------
@patch("MVP.backend.vector_store.embedder.get_embedding_model")
def test_embed_documents_filtering(mock_get_model):
    mock_model = MagicMock()

    mock_model.encode.return_value = np.array([
        [0.1] * 384,
        [0.2] * 384
    ], dtype=np.float32)

    mock_get_model.return_value = mock_model

    chunks = ["Valid chunk", "", "   ", "Another valid chunk"]

    result = embed_documents(chunks)

    assert len(result) == 2

    called_chunks = mock_model.encode.call_args[0][0]
    assert called_chunks == ["Valid chunk", "Another valid chunk"]


# -------------------------
# Test: exception handling
# -------------------------
@patch("MVP.backend.vector_store.embedder.get_embedding_model")
def test_embed_documents_exception(mock_get_model):
    mock_model = MagicMock()
    mock_model.encode.side_effect = Exception("Embedding failed")

    mock_get_model.return_value = mock_model

    with pytest.raises(Exception, match="Embedding failed"):
        embed_documents(["This will fail"])
