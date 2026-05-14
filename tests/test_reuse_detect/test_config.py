"""Tests for DetectionConfig."""

import tempfile
from pathlib import Path

import pytest

from reuse_detect.config import DetectionConfig


class TestDetectionConfig:
    """Test configuration loading and defaults."""

    def test_default_values(self):
        config = DetectionConfig()
        assert config.block_threshold == 0.85
        assert config.warn_threshold == 0.70
        assert config.min_block_lines == 3
        assert config.top_k == 50
        assert config.max_suggestions_per_block == 3
        assert config.max_llm_calls_per_commit == 20
        assert config.llm_timeout_seconds == 10.0
        assert config.similarity_threshold == 0.75
        assert config.index_refresh_interval_hours == 24
        assert "*.py" in config.include_patterns
        assert "test_*" in config.exclude_patterns

    def test_from_file_nonexistent(self):
        config = DetectionConfig.from_file(Path("/nonexistent/path.yml"))
        assert config.block_threshold == 0.85  # defaults

    def test_from_file_with_overrides(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yml", delete=False
        ) as f:
            f.write("block_threshold: 0.90\nmin_block_lines: 5\n")
            f.flush()

            config = DetectionConfig.from_file(Path(f.name))
            assert config.block_threshold == 0.90
            assert config.min_block_lines == 5
            # Other values remain default
            assert config.warn_threshold == 0.70

    def test_to_file_and_reload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yml"
            config = DetectionConfig(block_threshold=0.92, top_k=100)
            config.to_file(path)

            loaded = DetectionConfig.from_file(path)
            assert loaded.block_threshold == 0.92
            assert loaded.top_k == 100
