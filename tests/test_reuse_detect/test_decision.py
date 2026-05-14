"""Tests for the DecisionRenderer."""

from pathlib import Path

import pytest

from reuse_detect.config import DetectionConfig
from reuse_detect.decision import DecisionRenderer
from reuse_detect.models import (
    ChangeType,
    CodeBlock,
    Decision,
    IndexSource,
    ValidatedMatch,
)


def make_block() -> CodeBlock:
    return CodeBlock(
        file_path=Path("src/utils.py"),
        start_line=10,
        end_line=20,
        content="def helper():\n    return 42",
        change_type=ChangeType.MODIFIED,
        function_name="helper",
    )


def make_match(combined_score: float) -> ValidatedMatch:
    return ValidatedMatch(
        source=IndexSource.LOCAL_CODEBASE,
        indexed_id="local:lib/utils.py:hash123",
        structural_score=0.8,
        semantic_score=0.7,
        llm_confidence=0.9,
        combined_score=combined_score,
        llm_rationale="These functions are logically equivalent",
        import_path="from lib.utils import helper",
        existing_code="def helper():\n    return 42",
    )


class TestDecisionLogic:
    """Test threshold-based decision logic."""

    def setup_method(self):
        self.config = DetectionConfig(block_threshold=0.85, warn_threshold=0.70)
        self.renderer = DecisionRenderer(self.config)

    def test_block_when_score_above_block_threshold(self):
        block = make_block()
        validated = [make_match(0.90)]
        decision, suggestions = self.renderer.decide_and_render(block, validated)
        assert decision == Decision.BLOCK
        assert len(suggestions) > 0

    def test_warn_when_score_between_thresholds(self):
        block = make_block()
        validated = [make_match(0.75)]
        decision, suggestions = self.renderer.decide_and_render(block, validated)
        assert decision == Decision.WARN
        assert len(suggestions) > 0

    def test_pass_when_score_below_warn_threshold(self):
        block = make_block()
        validated = [make_match(0.50)]
        decision, suggestions = self.renderer.decide_and_render(block, validated)
        assert decision == Decision.PASS
        assert len(suggestions) == 0

    def test_pass_when_no_validated_matches(self):
        block = make_block()
        decision, suggestions = self.renderer.decide_and_render(block, [])
        assert decision == Decision.PASS
        assert len(suggestions) == 0

    def test_block_at_exact_threshold(self):
        block = make_block()
        validated = [make_match(0.85)]
        decision, _ = self.renderer.decide_and_render(block, validated)
        assert decision == Decision.BLOCK

    def test_warn_at_exact_threshold(self):
        block = make_block()
        validated = [make_match(0.70)]
        decision, _ = self.renderer.decide_and_render(block, validated)
        assert decision == Decision.WARN


class TestSuggestionRendering:
    """Test suggestion rendering."""

    def setup_method(self):
        self.config = DetectionConfig(max_suggestions_per_block=2)
        self.renderer = DecisionRenderer(self.config)

    def test_limits_suggestions_per_block(self):
        block = make_block()
        validated = [make_match(0.90), make_match(0.88), make_match(0.86)]
        _, suggestions = self.renderer.decide_and_render(block, validated)
        assert len(suggestions) <= 2

    def test_suggestion_has_all_required_fields(self):
        block = make_block()
        validated = [make_match(0.90)]
        _, suggestions = self.renderer.decide_and_render(block, validated)
        assert len(suggestions) == 1
        s = suggestions[0]
        assert s.original_code_location != ""
        assert s.existing_code_location != ""
        assert s.import_statement != ""
        assert s.usage_example != ""
        assert 0.0 <= s.confidence <= 1.0
        assert s.explanation != ""

    def test_suggestions_sorted_by_confidence(self):
        block = make_block()
        validated = [make_match(0.75), make_match(0.90), make_match(0.80)]
        _, suggestions = self.renderer.decide_and_render(block, validated)
        # Should be sorted descending by confidence
        for i in range(len(suggestions) - 1):
            assert suggestions[i].confidence >= suggestions[i + 1].confidence
