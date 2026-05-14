"""Stage 5: Hybrid Scorer and LLM Validator.

Computes structural similarity, embedding cosine, and LLM logic-equivalence
check for each candidate. Produces a combined score.
"""

import hashlib
import logging
from typing import Protocol

import numpy as np

from .config import DetectionConfig
from .models import (
    BlockFingerprint,
    IndexSource,
    SourceTaggedCandidate,
    ValidatedMatch,
)

logger = logging.getLogger(__name__)


class LLMProvider(Protocol):
    """Protocol for LLM API providers."""

    def check_logic_equivalence(
        self, source_code: str, candidate_code: str, timeout: float
    ) -> tuple[float, str]:
        """Check if two code blocks are logically equivalent.

        Returns (confidence, rationale) where confidence is in [0.0, 1.0].
        """
        ...


class GitHubModelsLLMProvider:
    """LLM provider using GitHub Models API for logic-equivalence checks."""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_token: str | None = None,
    ):
        self._model = model
        self._api_token = api_token

    def check_logic_equivalence(
        self, source_code: str, candidate_code: str, timeout: float
    ) -> tuple[float, str]:
        """Check logic equivalence via GitHub Models API."""
        import httpx

        prompt = (
            "Compare these two code blocks and determine if they are logically equivalent "
            "(i.e., they accomplish the same task/behavior, even if written differently).\n\n"
            f"Code Block A:\n```python\n{source_code}\n```\n\n"
            f"Code Block B:\n```python\n{candidate_code}\n```\n\n"
            "Respond with a JSON object containing:\n"
            '- "confidence": a float between 0.0 and 1.0 indicating how confident you are '
            "they are logically equivalent\n"
            '- "rationale": a brief explanation of your assessment'
        )

        try:
            response = httpx.post(
                "https://models.github.ai/inference/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._api_token}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self._model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                },
                timeout=timeout,
            )
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]

            # Parse the response
            import json

            # Try to extract JSON from the response
            try:
                result = json.loads(content)
            except json.JSONDecodeError:
                # Try to find JSON in the response
                import re

                match = re.search(r"\{[^}]+\}", content)
                if match:
                    result = json.loads(match.group())
                else:
                    return 0.0, "Failed to parse LLM response"

            confidence = float(result.get("confidence", 0.0))
            rationale = result.get("rationale", "No rationale provided")
            return max(0.0, min(1.0, confidence)), rationale

        except Exception as e:
            logger.warning("LLM call failed: %s", e)
            return 0.0, f"LLM unavailable: {e}"


class HybridScorerLLMValidator:
    """Computes AST + embedding cosine + LLM logic-equivalence in one pass."""

    # Weights for combining scores
    STRUCTURAL_WEIGHT = 0.3
    SEMANTIC_WEIGHT = 0.3
    LLM_WEIGHT = 0.4

    def __init__(
        self,
        config: DetectionConfig,
        llm_provider: LLMProvider | None = None,
    ):
        self.config = config
        self.llm_provider = llm_provider
        self._llm_calls_made = 0

    @property
    def llm_calls_remaining(self) -> int:
        """Number of LLM calls remaining for this commit."""
        return max(
            0, self.config.max_llm_calls_per_commit - self._llm_calls_made
        )

    def reset_llm_counter(self) -> None:
        """Reset the LLM call counter (call at start of each commit)."""
        self._llm_calls_made = 0

    def score_and_validate(
        self,
        fp: BlockFingerprint,
        candidates: list[SourceTaggedCandidate],
    ) -> list[ValidatedMatch]:
        """For each candidate: structural score, semantic score, LLM logic check.

        Emit ValidatedMatch with a combined score in [0.0, 1.0].
        """
        validated: list[ValidatedMatch] = []

        for candidate in candidates:
            # Compute structural similarity
            structural_score = self._compute_structural_similarity(
                fp, candidate.fingerprint
            )

            # Compute semantic similarity (1 - cosine distance)
            semantic_score = self._compute_semantic_similarity(
                fp, candidate.fingerprint
            )

            # LLM logic-equivalence check (if budget allows)
            llm_confidence = 0.0
            llm_rationale = "LLM not invoked"

            if (
                self.llm_provider
                and self._llm_calls_made < self.config.max_llm_calls_per_commit
            ):
                try:
                    llm_confidence, llm_rationale = (
                        self.llm_provider.check_logic_equivalence(
                            fp.block.content,
                            candidate.fingerprint.block.content,
                            timeout=self.config.llm_timeout_seconds,
                        )
                    )
                    self._llm_calls_made += 1
                except Exception as e:
                    logger.warning("LLM validation failed: %s", e)
                    llm_rationale = f"LLM fallback: {e}"

            # Compute combined score
            combined_score = self._compute_combined_score(
                structural_score, semantic_score, llm_confidence
            )

            # Determine import path from candidate metadata
            import_path = self._derive_import_path(candidate)

            validated.append(
                ValidatedMatch(
                    source=candidate.source,
                    indexed_id=candidate.indexed_id,
                    structural_score=structural_score,
                    semantic_score=semantic_score,
                    llm_confidence=llm_confidence,
                    combined_score=combined_score,
                    llm_rationale=llm_rationale,
                    import_path=import_path,
                    existing_code=candidate.fingerprint.block.content,
                )
            )

        return validated

    def _compute_structural_similarity(
        self, fp1: BlockFingerprint, fp2: BlockFingerprint
    ) -> float:
        """Compute structural similarity from AST features.

        This is rename-invariant and comment-invariant because it uses
        the AST structure hash and normalized token sequences.
        """
        score = 0.0
        total_weight = 0.0

        # AST structure hash comparison (highest weight - rename/comment invariant)
        if fp1.features.ast_structure_hash and fp2.features.ast_structure_hash:
            if (
                fp1.features.ast_structure_hash
                == fp2.features.ast_structure_hash
            ):
                score += 0.4
            total_weight += 0.4

        # Control flow pattern similarity
        if (
            fp1.features.control_flow_pattern
            or fp2.features.control_flow_pattern
        ):
            cf_sim = self._string_similarity(
                fp1.features.control_flow_pattern,
                fp2.features.control_flow_pattern,
            )
            score += 0.2 * cf_sim
            total_weight += 0.2

        # Token sequence similarity (normalized, so rename-invariant)
        if fp1.features.token_sequence or fp2.features.token_sequence:
            ts_sim = self._sequence_similarity(
                fp1.features.token_sequence, fp2.features.token_sequence
            )
            score += 0.2 * ts_sim
            total_weight += 0.2

        # Function signature structure similarity
        if (
            fp1.features.function_signatures
            or fp2.features.function_signatures
        ):
            sig_sim = self._list_similarity(
                fp1.features.function_signatures,
                fp2.features.function_signatures,
            )
            score += 0.1 * sig_sim
            total_weight += 0.1

        # Decorator similarity
        if fp1.features.decorators or fp2.features.decorators:
            dec_sim = self._list_similarity(
                fp1.features.decorators, fp2.features.decorators
            )
            score += 0.1 * dec_sim
            total_weight += 0.1

        if total_weight == 0:
            return 0.0

        return min(1.0, max(0.0, score / total_weight))

    def _compute_semantic_similarity(
        self, fp1: BlockFingerprint, fp2: BlockFingerprint
    ) -> float:
        """Compute semantic similarity via embedding cosine distance."""
        if not fp1.embedding.vector or not fp2.embedding.vector:
            return 0.0

        v1 = np.array(fp1.embedding.vector, dtype=np.float32)
        v2 = np.array(fp2.embedding.vector, dtype=np.float32)

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        cosine_sim = float(np.dot(v1, v2) / (norm1 * norm2))
        # Clamp to [0, 1] (cosine can be negative for dissimilar vectors)
        return max(0.0, min(1.0, cosine_sim))

    def _compute_combined_score(
        self,
        structural: float,
        semantic: float,
        llm_confidence: float,
    ) -> float:
        """Combine scores into a single value in [0.0, 1.0].

        If LLM was not invoked (confidence=0), redistribute its weight
        to structural and semantic equally.
        """
        if llm_confidence == 0.0 and self.LLM_WEIGHT > 0:
            # Redistribute LLM weight
            adjusted_structural_weight = self.STRUCTURAL_WEIGHT + (
                self.LLM_WEIGHT / 2
            )
            adjusted_semantic_weight = self.SEMANTIC_WEIGHT + (
                self.LLM_WEIGHT / 2
            )
            combined = (
                structural * adjusted_structural_weight
                + semantic * adjusted_semantic_weight
            )
        else:
            combined = (
                structural * self.STRUCTURAL_WEIGHT
                + semantic * self.SEMANTIC_WEIGHT
                + llm_confidence * self.LLM_WEIGHT
            )

        return max(0.0, min(1.0, combined))

    def _string_similarity(self, s1: str, s2: str) -> float:
        """Simple string similarity based on common subsequence ratio."""
        if not s1 and not s2:
            return 1.0
        if not s1 or not s2:
            return 0.0

        # Use longest common subsequence ratio
        len1, len2 = len(s1), len(s2)
        max_len = max(len1, len2)
        if max_len == 0:
            return 1.0

        # Simple character-level overlap
        common = sum(1 for c1, c2 in zip(s1, s2) if c1 == c2)
        return common / max_len

    def _sequence_similarity(
        self, seq1: list[str], seq2: list[str]
    ) -> float:
        """Compute similarity between two token sequences."""
        if not seq1 and not seq2:
            return 1.0
        if not seq1 or not seq2:
            return 0.0

        # Jaccard similarity on token sets
        set1 = set(seq1)
        set2 = set(seq2)
        intersection = len(set1 & set2)
        union = len(set1 | set2)

        if union == 0:
            return 1.0
        return intersection / union

    def _list_similarity(
        self, list1: list[str], list2: list[str]
    ) -> float:
        """Compute similarity between two lists of strings."""
        if not list1 and not list2:
            return 1.0
        if not list1 or not list2:
            return 0.0

        set1 = set(list1)
        set2 = set(list2)
        intersection = len(set1 & set2)
        union = len(set1 | set2)

        if union == 0:
            return 1.0
        return intersection / union

    def _derive_import_path(self, candidate: SourceTaggedCandidate) -> str:
        """Derive an import path from the candidate metadata."""
        fp = candidate.fingerprint
        file_path = str(fp.block.file_path)

        # Convert file path to module path
        module_path = file_path.replace("/", ".").replace("\\", ".")
        if module_path.endswith(".py"):
            module_path = module_path[:-3]

        if fp.block.function_name:
            return f"from {module_path} import {fp.block.function_name}"
        elif fp.block.class_name:
            return f"from {module_path} import {fp.block.class_name}"
        else:
            return f"import {module_path}"
