"""Tests for the FeatureEmbeddingBuilder."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from reuse_detect.feature_builder import FeatureEmbeddingBuilder
from reuse_detect.models import ChangeType, CodeBlock, CodeFeatures


class MockEmbeddingProvider:
    """Mock embedding provider for testing."""

    def __init__(self, dimension: int = 8):
        self._dimension = dimension

    @property
    def model_id(self) -> str:
        return "mock-model"

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> list[float]:
        # Return a deterministic vector based on text hash
        import hashlib

        h = hashlib.md5(text.encode()).hexdigest()
        return [int(h[i : i + 2], 16) / 255.0 for i in range(0, self._dimension * 2, 2)]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


class TestFeatureExtraction:
    """Test AST feature extraction."""

    def setup_method(self):
        self.provider = MockEmbeddingProvider()
        self.builder = FeatureEmbeddingBuilder(self.provider)

    def test_extracts_function_signature(self):
        source = "def add(a: int, b: int) -> int:\n    return a + b"
        features = self.builder._extract_features(source)
        assert len(features.function_signatures) == 1
        assert "add" in features.function_signatures[0]
        assert "int" in features.function_signatures[0]

    def test_extracts_class_hierarchy(self):
        source = "class MyClass(BaseClass):\n    pass"
        features = self.builder._extract_features(source)
        assert len(features.class_hierarchy) == 1
        assert "MyClass" in features.class_hierarchy[0]
        assert "BaseClass" in features.class_hierarchy[0]

    def test_extracts_decorators(self):
        source = "@staticmethod\ndef foo():\n    pass"
        features = self.builder._extract_features(source)
        assert "staticmethod" in features.decorators

    def test_extracts_control_flow(self):
        source = "def foo(x):\n    if x > 0:\n        for i in range(x):\n            pass\n    return x"
        features = self.builder._extract_features(source)
        assert "IF" in features.control_flow_pattern
        assert "FOR" in features.control_flow_pattern
        assert "RET" in features.control_flow_pattern

    def test_extracts_import_patterns(self):
        source = "import os\nfrom pathlib import Path\ndef foo():\n    pass"
        features = self.builder._extract_features(source)
        assert len(features.import_patterns) == 2

    def test_extracts_parameter_types(self):
        source = "def foo(x: int, y: str) -> bool:\n    return True"
        features = self.builder._extract_features(source)
        assert "int" in features.parameter_types
        assert "str" in features.parameter_types
        assert features.return_type == "bool"

    def test_structure_hash_rename_invariant(self):
        source1 = "def foo(x):\n    y = x + 1\n    return y"
        source2 = "def bar(a):\n    b = a + 1\n    return b"
        features1 = self.builder._extract_features(source1)
        features2 = self.builder._extract_features(source2)
        assert features1.ast_structure_hash == features2.ast_structure_hash

    def test_structure_hash_comment_invariant(self):
        source1 = "def foo(x):\n    return x + 1"
        source2 = "def foo(x):\n    # add one\n    return x + 1"
        features1 = self.builder._extract_features(source1)
        features2 = self.builder._extract_features(source2)
        assert features1.ast_structure_hash == features2.ast_structure_hash

    def test_unparseable_code_returns_empty_features(self):
        source = "this is not valid python {{{}"
        features = self.builder._extract_features(source)
        assert features.function_signatures == []
        assert features.ast_structure_hash != ""  # Still gets a hash


class TestBuild:
    """Test the full build method."""

    def setup_method(self):
        self.provider = MockEmbeddingProvider()
        self.builder = FeatureEmbeddingBuilder(self.provider)

    def test_build_produces_fingerprint(self):
        block = CodeBlock(
            file_path=Path("test.py"),
            start_line=1,
            end_line=3,
            content="def add(a, b):\n    return a + b",
            change_type=ChangeType.ADDED,
            function_name="add",
        )
        fp = self.builder.build(block)
        assert fp.block == block
        assert fp.content_hash != ""
        assert fp.embedding.dim == 8
        assert len(fp.embedding.vector) == 8
        assert fp.features.function_signatures != []

    def test_build_batch(self):
        blocks = [
            CodeBlock(
                file_path=Path("test.py"),
                start_line=1,
                end_line=2,
                content=f"def func{i}():\n    return {i}",
                change_type=ChangeType.ADDED,
                function_name=f"func{i}",
            )
            for i in range(3)
        ]
        fps = self.builder.build_batch(blocks)
        assert len(fps) == 3
        for fp in fps:
            assert fp.embedding.dim == 8
