import pytest
from MVP.backend.ingestion.chunk_docs import chunk_documents

def test_basic_chunking():
    docs = ["A" * 1000]
    chunks = chunk_documents(docs, chunk_size=200, overlap=50)

    assert len(chunks) > 0
    assert all(len(chunk) <= 200 for chunk in chunks)

def test_overlap_logic():
    text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" * 10
    chunks = chunk_documents([text], chunk_size=100, overlap=20)

    assert chunks[0][-20:] == chunks[1][:20]

def test_empty_document_skipped():
    chunks = chunk_documents(["", "   "])
    assert chunks == []

def test_invalid_overlap():
    with pytest.raises(ValueError):
        chunk_documents(["test"], chunk_size=100, overlap=100)
