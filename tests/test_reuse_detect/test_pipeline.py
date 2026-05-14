"""Tests for the pipeline orchestrator."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from reuse_detect.config import DetectionConfig
from reuse_detect.models import Decision
from reuse_detect.pipeline import detect_reusable_code


class MockEmbeddingProvider:
    """Mock embedding provider."""

    @property
    def model_id(self) -> str:
        return "mock"

    @property
    def dimension(self) -> int:
        return 4

    def embed(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3, 0.4]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


class TestPipelineEmptyDiff:
    """Test pipeline behavior with empty diffs."""

    def test_empty_diff_returns_pass(self):
        config = DetectionConfig()
        provider = MockEmbeddingProvider()

        with patch(
            "reuse_detect.change_extractor.ChangeExtractor.extract",
            return_value=[],
        ):
            decision, suggestions = detect_reusable_code(
                config=config,
                embedding_provider=provider,
                repo_root=Path("."),
            )

        assert decision == Decision.PASS
        assert suggestions == []
