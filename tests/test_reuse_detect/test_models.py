"""Tests for core data models."""

from pathlib import Path

from reuse_detect.models import (
    ChangeType,
    CodeBlock,
    CodeEmbedding,
    CodeFeatures,
    BlockFingerprint,
    Decision,
    IndexSource,
    ReuseSuggestion,
    SourceTaggedCandidate,
    ValidatedMatch,
)


class TestCodeBlock:
    def test_line_count(self):
        block = CodeBlock(
            file_path=Path("test.py"),
            start_line=1,
            end_line=10,
            content="def foo():\n    pass",
            change_type=ChangeType.ADDED,
        )
        assert block.line_count == 10

    def test_function_name(self):
        block = CodeBlock(
            file_path=Path("test.py"),
            start_line=1,
            end_line=3,
            content="def foo():\n    pass",
            change_type=ChangeType.MODIFIED,
            function_name="foo",
        )
        assert block.function_name == "foo"
        assert block.class_name is None


class TestEnums:
    def test_change_type_values(self):
        assert ChangeType.ADDED.value == "added"
        assert ChangeType.MODIFIED.value == "modified"

    def test_index_source_values(self):
        assert IndexSource.GITHUB_LIBRARY.value == "github_library"
        assert IndexSource.LOCAL_CODEBASE.value == "local_codebase"

    def test_decision_values(self):
        assert Decision.PASS.value == "pass"
        assert Decision.WARN.value == "warn"
        assert Decision.BLOCK.value == "block"


class TestCodeFeatures:
    def test_default_factory(self):
        features = CodeFeatures()
        assert features.function_signatures == []
        assert features.class_hierarchy == []
        assert features.import_patterns == []
        assert features.decorators == []
        assert features.control_flow_pattern == ""
        assert features.parameter_types == []
        assert features.return_type is None
        assert features.ast_structure_hash == ""


class TestBlockFingerprint:
    def test_construction(self):
        block = CodeBlock(
            file_path=Path("test.py"),
            start_line=1,
            end_line=5,
            content="def foo():\n    return 1",
            change_type=ChangeType.ADDED,
        )
        fp = BlockFingerprint(
            block=block,
            features=CodeFeatures(),
            embedding=CodeEmbedding(vector=[0.1, 0.2], dim=2, model_id="test"),
            content_hash="abc123",
        )
        assert fp.content_hash == "abc123"
        assert fp.embedding.dim == 2
