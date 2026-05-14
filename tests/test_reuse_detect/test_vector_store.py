"""Tests for the FAISSVectorStore."""

import tempfile
from pathlib import Path

import pytest

from reuse_detect.models import (
    BlockFingerprint,
    ChangeType,
    CodeBlock,
    CodeEmbedding,
    CodeFeatures,
    IndexSource,
)
from reuse_detect.vector_store import FAISSVectorStore


def make_fingerprint(vector: list[float], content_hash: str = "hash") -> BlockFingerprint:
    return BlockFingerprint(
        block=CodeBlock(
            file_path=Path("test.py"),
            start_line=1,
            end_line=3,
            content="def foo():\n    pass",
            change_type=ChangeType.ADDED,
            function_name="foo",
        ),
        features=CodeFeatures(),
        embedding=CodeEmbedding(vector=vector, dim=len(vector), model_id="test"),
        content_hash=content_hash,
    )


class TestFAISSVectorStore:
    """Test the FAISS vector store."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = FAISSVectorStore(
            store_path=Path(self.tmpdir), dimension=4
        )

    def test_upsert_and_query(self):
        fp = make_fingerprint([1.0, 0.0, 0.0, 0.0], "hash1")
        self.store.upsert(fp, "id1", IndexSource.LOCAL_CODEBASE)

        results = self.store.query([1.0, 0.0, 0.0, 0.0], top_k=5)
        assert len(results) == 1
        assert results[0][0] == "id1"
        assert results[0][2] == IndexSource.LOCAL_CODEBASE

    def test_query_returns_nearest(self):
        fp1 = make_fingerprint([1.0, 0.0, 0.0, 0.0], "hash1")
        fp2 = make_fingerprint([0.0, 1.0, 0.0, 0.0], "hash2")
        fp3 = make_fingerprint([0.9, 0.1, 0.0, 0.0], "hash3")

        self.store.upsert(fp1, "id1", IndexSource.LOCAL_CODEBASE)
        self.store.upsert(fp2, "id2", IndexSource.GITHUB_LIBRARY)
        self.store.upsert(fp3, "id3", IndexSource.LOCAL_CODEBASE)

        results = self.store.query([1.0, 0.0, 0.0, 0.0], top_k=2)
        assert len(results) == 2
        # id1 should be closest (exact match)
        assert results[0][0] == "id1"

    def test_source_filtering(self):
        fp1 = make_fingerprint([1.0, 0.0, 0.0, 0.0], "hash1")
        fp2 = make_fingerprint([0.9, 0.1, 0.0, 0.0], "hash2")

        self.store.upsert(fp1, "id1", IndexSource.LOCAL_CODEBASE)
        self.store.upsert(fp2, "id2", IndexSource.GITHUB_LIBRARY)

        # Query only GitHub sources
        results = self.store.query(
            [1.0, 0.0, 0.0, 0.0],
            top_k=5,
            sources=[IndexSource.GITHUB_LIBRARY],
        )
        assert len(results) == 1
        assert results[0][2] == IndexSource.GITHUB_LIBRARY

    def test_top_k_limit(self):
        for i in range(10):
            fp = make_fingerprint([float(i) / 10, 0.0, 0.0, 0.0], f"hash{i}")
            self.store.upsert(fp, f"id{i}", IndexSource.LOCAL_CODEBASE)

        results = self.store.query([1.0, 0.0, 0.0, 0.0], top_k=3)
        assert len(results) <= 3

    def test_delete(self):
        fp = make_fingerprint([1.0, 0.0, 0.0, 0.0], "hash1")
        self.store.upsert(fp, "id1", IndexSource.LOCAL_CODEBASE)
        assert self.store.size == 1

        self.store.delete("id1")
        results = self.store.query([1.0, 0.0, 0.0, 0.0], top_k=5)
        assert len(results) == 0

    def test_upsert_replaces_existing(self):
        fp1 = make_fingerprint([1.0, 0.0, 0.0, 0.0], "hash1")
        fp2 = make_fingerprint([0.0, 1.0, 0.0, 0.0], "hash2")

        self.store.upsert(fp1, "id1", IndexSource.LOCAL_CODEBASE)
        self.store.upsert(fp2, "id1", IndexSource.LOCAL_CODEBASE)

        # Should only have one entry
        assert self.store.size == 1

    def test_save_and_load(self):
        fp = make_fingerprint([1.0, 0.0, 0.0, 0.0], "hash1")
        self.store.upsert(fp, "id1", IndexSource.LOCAL_CODEBASE)
        self.store.save()

        # Create new store and load
        new_store = FAISSVectorStore(
            store_path=Path(self.tmpdir), dimension=4
        )
        new_store.load()

        results = new_store.query([1.0, 0.0, 0.0, 0.0], top_k=5)
        assert len(results) == 1
        assert results[0][0] == "id1"

    def test_empty_store_query(self):
        results = self.store.query([1.0, 0.0, 0.0, 0.0], top_k=5)
        assert results == []
