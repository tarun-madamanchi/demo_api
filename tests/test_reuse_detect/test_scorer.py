"""Tests for the HybridScorerLLMValidator."""

from pathlib import Path

import pytest

from reuse_detect.config import DetectionConfig
from reuse_detect.models import (
    BlockFingerprint,
    ChangeType,
    CodeBlock,
    CodeEmbedding,
    CodeFeatures,
    IndexSource,
    SourceTaggedCandidate,
)
from reuse_detect.scorer import HybridScorerLLMValidator


class MockLLMProvider:
    """Mock LLM provider for testing."""

    def __init__(self, confidence: float = 0.8, rationale: str = "Similar logic"):
        self.confidence = confidence
        self.rationale = rationale
        self.call_count = 0

    def check_logic_equivalence(
        self, source_code: str, candidate_code: str, timeout: float
    ) -> tuple[float, str]:
        self.call_count += 1
        return self.confidence, self.rationale


def make_fingerprint(
    content: str = "def foo():\n    return 1",
    vector: list[float] | None = None,
    ast_hash: str = "abc",
    control_flow: str = "RET",
    function_name: str = "foo",
) -> BlockFingerprint:
    """Helper to create a BlockFingerprint for testing."""
    if vector is None:
        vector = [0.1, 0.2, 0.3, 0.4]

    return BlockFingerprint(
        block=CodeBlock(
            file_path=Path("test.py"),
            start_line=1,
            end_line=3,
            content=content,
            change_type=ChangeType.ADDED,
            function_name=function_name,
        ),
        features=CodeFeatures(
            function_signatures=[f"def {function_name}()"],
            ast_structure_hash=ast_hash,
            control_flow_pattern=control_flow,
            token_sequence=["NAME", "CALL", "RET"],
        ),
        embedding=CodeEmbedding(vector=vector, dim=len(vector), model_id="test"),
        content_hash="hash123",
    )


class TestStructuralSimilarity:
    """Test structural similarity computation."""

    def setup_method(self):
        self.config = DetectionConfig()
        self.scorer = HybridScorerLLMValidator(self.config)

    def test_identical_fingerprints_score_1(self):
        fp = make_fingerprint()
        score = self.scorer._compute_structural_similarity(fp, fp)
        assert score == 1.0

    def test_different_fingerprints_score_less_than_1(self):
        fp1 = make_fingerprint(ast_hash="abc", control_flow="IF|RET")
        fp2 = make_fingerprint(ast_hash="xyz", control_flow="FOR|RET")
        score = self.scorer._compute_structural_similarity(fp1, fp2)
        assert 0.0 <= score < 1.0

    def test_score_in_range(self):
        fp1 = make_fingerprint()
        fp2 = make_fingerprint(ast_hash="different")
        score = self.scorer._compute_structural_similarity(fp1, fp2)
        assert 0.0 <= score <= 1.0


class TestSemanticSimilarity:
    """Test semantic similarity computation."""

    def setup_method(self):
        self.config = DetectionConfig()
        self.scorer = HybridScorerLLMValidator(self.config)

    def test_identical_vectors_score_1(self):
        fp = make_fingerprint(vector=[1.0, 0.0, 0.0, 0.0])
        score = self.scorer._compute_semantic_similarity(fp, fp)
        assert abs(score - 1.0) < 1e-6

    def test_orthogonal_vectors_score_0(self):
        fp1 = make_fingerprint(vector=[1.0, 0.0, 0.0, 0.0])
        fp2 = make_fingerprint(vector=[0.0, 1.0, 0.0, 0.0])
        score = self.scorer._compute_semantic_similarity(fp1, fp2)
        assert abs(score) < 1e-6

    def test_empty_vectors_score_0(self):
        fp1 = make_fingerprint(vector=[])
        fp2 = make_fingerprint(vector=[])
        score = self.scorer._compute_semantic_similarity(fp1, fp2)
        assert score == 0.0


class TestCombinedScore:
    """Test combined score computation."""

    def setup_method(self):
        self.config = DetectionConfig()
        self.scorer = HybridScorerLLMValidator(self.config)

    def test_all_zeros_gives_zero(self):
        score = self.scorer._compute_combined_score(0.0, 0.0, 0.0)
        assert score == 0.0

    def test_all_ones_gives_one(self):
        score = self.scorer._compute_combined_score(1.0, 1.0, 1.0)
        assert abs(score - 1.0) < 1e-6

    def test_score_always_in_range(self):
        for s in [0.0, 0.25, 0.5, 0.75, 1.0]:
            for e in [0.0, 0.25, 0.5, 0.75, 1.0]:
                for l in [0.0, 0.25, 0.5, 0.75, 1.0]:
                    score = self.scorer._compute_combined_score(s, e, l)
                    assert 0.0 <= score <= 1.0


class TestLLMCallCap:
    """Test LLM call cap enforcement."""

    def test_respects_max_calls(self):
        config = DetectionConfig(max_llm_calls_per_commit=2)
        llm = MockLLMProvider()
        scorer = HybridScorerLLMValidator(config, llm)

        fp = make_fingerprint()
        candidates = [
            SourceTaggedCandidate(
                source=IndexSource.LOCAL_CODEBASE,
                indexed_id=f"id_{i}",
                distance=0.1,
                fingerprint=make_fingerprint(function_name=f"func{i}"),
            )
            for i in range(5)
        ]

        scorer.score_and_validate(fp, candidates)
        assert llm.call_count == 2  # Capped at max_llm_calls_per_commit

    def test_reset_counter(self):
        config = DetectionConfig(max_llm_calls_per_commit=1)
        llm = MockLLMProvider()
        scorer = HybridScorerLLMValidator(config, llm)

        fp = make_fingerprint()
        candidates = [
            SourceTaggedCandidate(
                source=IndexSource.LOCAL_CODEBASE,
                indexed_id="id_0",
                distance=0.1,
                fingerprint=make_fingerprint(),
            )
        ]

        scorer.score_and_validate(fp, candidates)
        assert llm.call_count == 1

        scorer.reset_llm_counter()
        scorer.score_and_validate(fp, candidates)
        assert llm.call_count == 2


class TestScoreAndValidate:
    """Test the full score_and_validate method."""

    def test_produces_validated_matches(self):
        config = DetectionConfig()
        llm = MockLLMProvider(confidence=0.9)
        scorer = HybridScorerLLMValidator(config, llm)

        fp = make_fingerprint(vector=[1.0, 0.0, 0.0, 0.0])
        candidates = [
            SourceTaggedCandidate(
                source=IndexSource.GITHUB_LIBRARY,
                indexed_id="github:repo:hash",
                distance=0.1,
                fingerprint=make_fingerprint(vector=[0.9, 0.1, 0.0, 0.0]),
            )
        ]

        results = scorer.score_and_validate(fp, candidates)
        assert len(results) == 1
        assert results[0].source == IndexSource.GITHUB_LIBRARY
        assert 0.0 <= results[0].combined_score <= 1.0
        assert results[0].llm_confidence == 0.9

    def test_without_llm_provider(self):
        config = DetectionConfig()
        scorer = HybridScorerLLMValidator(config, llm_provider=None)

        fp = make_fingerprint(vector=[1.0, 0.0, 0.0, 0.0])
        candidates = [
            SourceTaggedCandidate(
                source=IndexSource.LOCAL_CODEBASE,
                indexed_id="local:test:hash",
                distance=0.2,
                fingerprint=make_fingerprint(vector=[0.8, 0.2, 0.0, 0.0]),
            )
        ]

        results = scorer.score_and_validate(fp, candidates)
        assert len(results) == 1
        assert results[0].llm_confidence == 0.0
        assert results[0].combined_score >= 0.0
