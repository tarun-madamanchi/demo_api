"""Tests for the ChangeExtractor."""

from pathlib import Path

import pytest

from reuse_detect.change_extractor import ChangeExtractor
from reuse_detect.config import DetectionConfig
from reuse_detect.models import ChangeType, CodeBlock


class TestIsTrivial:
    """Test trivial block detection."""

    def setup_method(self):
        self.config = DetectionConfig(min_block_lines=3)
        self.extractor = ChangeExtractor(self.config)

    def test_block_below_min_lines_is_trivial(self):
        block = CodeBlock(
            file_path=Path("test.py"),
            start_line=1,
            end_line=2,
            content="def foo():\n    pass",
            change_type=ChangeType.ADDED,
        )
        assert self.extractor.is_trivial(block) is True

    def test_block_at_min_lines_is_not_trivial(self):
        block = CodeBlock(
            file_path=Path("test.py"),
            start_line=1,
            end_line=3,
            content="def foo():\n    x = 1\n    return x",
            change_type=ChangeType.ADDED,
        )
        assert self.extractor.is_trivial(block) is False

    def test_comments_only_is_trivial(self):
        block = CodeBlock(
            file_path=Path("test.py"),
            start_line=1,
            end_line=5,
            content="# comment 1\n# comment 2\n# comment 3\n# comment 4\n# comment 5",
            change_type=ChangeType.ADDED,
        )
        assert self.extractor.is_trivial(block) is True

    def test_whitespace_only_is_trivial(self):
        block = CodeBlock(
            file_path=Path("test.py"),
            start_line=1,
            end_line=5,
            content="\n\n\n\n\n",
            change_type=ChangeType.ADDED,
        )
        assert self.extractor.is_trivial(block) is True

    def test_mixed_comments_and_code_is_not_trivial(self):
        block = CodeBlock(
            file_path=Path("test.py"),
            start_line=1,
            end_line=4,
            content="# comment\ndef foo():\n    x = 1\n    return x",
            change_type=ChangeType.ADDED,
        )
        assert self.extractor.is_trivial(block) is False


class TestNormalize:
    """Test source normalization."""

    def setup_method(self):
        self.config = DetectionConfig()
        self.extractor = ChangeExtractor(self.config)

    def test_idempotence(self):
        source = "  def foo():\n      return 1\n\n\n"
        once = self.extractor.normalize(source)
        twice = self.extractor.normalize(once)
        assert once == twice

    def test_strips_trailing_whitespace(self):
        source = "def foo():   \n    return 1   "
        result = self.extractor.normalize(source)
        for line in result.splitlines():
            assert line == line.rstrip()

    def test_normalizes_line_endings(self):
        source = "def foo():\r\n    return 1\r"
        result = self.extractor.normalize(source)
        assert "\r" not in result

    def test_collapses_multiple_blank_lines(self):
        source = "def foo():\n\n\n\n    return 1"
        result = self.extractor.normalize(source)
        assert "\n\n\n" not in result

    def test_removes_leading_trailing_blanks(self):
        source = "\n\ndef foo():\n    return 1\n\n"
        result = self.extractor.normalize(source)
        assert not result.startswith("\n")
        assert not result.endswith("\n")


class TestShouldProcessFile:
    """Test file pattern matching."""

    def test_python_file_included(self):
        config = DetectionConfig(
            include_patterns=["*.py"], exclude_patterns=["test_*"]
        )
        extractor = ChangeExtractor(config)
        assert extractor._should_process_file(Path("src/main.py")) is True

    def test_test_file_excluded(self):
        config = DetectionConfig(
            include_patterns=["*.py"], exclude_patterns=["test_*"]
        )
        extractor = ChangeExtractor(config)
        assert extractor._should_process_file(Path("test_main.py")) is False

    def test_non_python_file_excluded(self):
        config = DetectionConfig(
            include_patterns=["*.py"], exclude_patterns=[]
        )
        extractor = ChangeExtractor(config)
        assert extractor._should_process_file(Path("readme.md")) is False

    def test_conftest_excluded(self):
        config = DetectionConfig(
            include_patterns=["*.py"], exclude_patterns=["conftest.py"]
        )
        extractor = ChangeExtractor(config)
        assert extractor._should_process_file(Path("conftest.py")) is False
