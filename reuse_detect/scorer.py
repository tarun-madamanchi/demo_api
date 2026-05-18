"""Stage 5: Hybrid Scorer and LLM Validator.

Computes structural similarity, embedding cosine, and LLM logic-equivalence
check for each candidate. Produces a combined score.
"""

import hashlib
import logging
from typing import Protocol

import numpy as np

from .config import DetectionConfig
from .models import BlockFingerprint, IndexSource, SourceTaggedCandidate, ValidatedMatch

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
    """LLM provider using GitHub Models API for logic-equivalence checks.

    Automatically disables itself if the API is unreachable (SSL/network)
    to avoid repeated slow timeouts.
    """

    def __init__(
        self,
        model: str = "gpt-4.1",
        api_token: str | None = None,
    ):
        self._model = model
        self._api_token = api_token
        self._api_disabled = False

    def check_logic_equivalence(
        self, source_code: str, candidate_code: str, timeout: float
    ) -> tuple[float, str]:
        """Check logic equivalence via GitHub Models API."""
        import httpx

        if self._api_disabled:
            return 0.0, "LLM API disabled (network unreachable)"

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
                verify=False,
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
            error_str = str(e)
            if any(
                kw in error_str
                for kw in ("SSL", "CERTIFICATE_VERIFY", "ConnectError", "timed out")
            ):
                if not self._api_disabled:
                    logger.warning(
                        "LLM API unreachable (corporate SSL proxy): %s. "
                        "Disabling LLM validation for this session.",
                        e,
                    )
                    self._api_disabled = True
            else:
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
        return max(0, self.config.max_llm_calls_per_commit - self._llm_calls_made)

    def reset_llm_counter(self) -> None:
        """Reset the LLM call counter (call at start of each commit)."""
        self._llm_calls_made = 0

    # Max candidates to send to LLM per block (pre-filter the rest)
    MAX_LLM_CANDIDATES_PER_BLOCK = 5

    def score_and_validate(
        self,
        fp: BlockFingerprint,
        candidates: list[SourceTaggedCandidate],
    ) -> list[ValidatedMatch]:
        """For each candidate: structural score, semantic score, LLM logic check.

        Pre-filters candidates: only the top N by structural+semantic score
        are sent to the LLM. This avoids exhausting rate limits on low-quality
        candidates.

        Emit ValidatedMatch with a combined score in [0.0, 1.0].
        """
        # Phase 1: Score ALL candidates with structural + semantic (cheap, local)
        pre_scored: list[tuple[SourceTaggedCandidate, float, float, bool]] = []
        for candidate in candidates:
            structural_score = self._compute_structural_similarity(
                fp, candidate.fingerprint
            )
            semantic_score = self._compute_semantic_similarity(
                fp, candidate.fingerprint
            )
            has_semantic = bool(
                fp.embedding.vector and candidate.fingerprint.embedding.vector
            )
            pre_scored.append((candidate, structural_score, semantic_score, has_semantic))

        # Phase 2: Sort by preliminary score (structural + semantic) descending
        pre_scored.sort(
            key=lambda x: (x[1] + x[2]) / 2.0,
            reverse=True,
        )

        # Phase 3: Only call LLM on the top candidates
        validated: list[ValidatedMatch] = []
        for idx, (candidate, structural_score, semantic_score, has_semantic) in enumerate(pre_scored):
            llm_confidence = 0.0
            llm_rationale = "LLM not invoked (pre-filtered)"

            # Only invoke LLM on top candidates
            should_call_llm = (
                self.llm_provider
                and self._llm_calls_made < self.config.max_llm_calls_per_commit
                and idx < self.MAX_LLM_CANDIDATES_PER_BLOCK
            )

            if should_call_llm:
                llm_confidence, llm_rationale = self._call_llm_with_retry(
                    fp.block.content,
                    candidate.fingerprint.block.content,
                )

            # Compute combined score
            combined_score = self._compute_combined_score(
                structural_score, semantic_score, llm_confidence, has_semantic
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

    def _call_llm_with_retry(
        self,
        source_code: str,
        candidate_code: str,
        max_retries: int = 1,
    ) -> tuple[float, str]:
        """Call LLM with retry on 429 rate-limit errors.

        Respects the Retry-After header from the API response.
        Uses a shorter wait (5s) to avoid blocking the pre-commit too long.
        If still rate-limited after retry, returns 0 gracefully.
        """
        import time

        for attempt in range(max_retries + 1):
            try:
                confidence, rationale = self.llm_provider.check_logic_equivalence(
                    source_code,
                    candidate_code,
                    timeout=self.config.llm_timeout_seconds,
                )
                self._llm_calls_made += 1
                return confidence, rationale
            except Exception as e:
                error_str = str(e)
                if "429" in error_str and attempt < max_retries:
                    # Wait a short time before retry (don't block pre-commit too long)
                    wait_time = 5
                    logger.info(
                        "Rate limited (attempt %d/%d). Waiting %ds before retry...",
                        attempt + 1,
                        max_retries + 1,
                        wait_time,
                    )
                    time.sleep(wait_time)
                    continue
                elif "429" in error_str:
                    # Rate limited and out of retries — stop calling LLM entirely
                    # to avoid wasting time on further 429s
                    self._llm_calls_made = self.config.max_llm_calls_per_commit
                    logger.warning(
                        "Rate limit exhausted. Disabling LLM for remaining candidates."
                    )
                    return 0.0, "LLM rate-limited (budget exhausted)"
                else:
                    logger.warning("LLM validation failed: %s", e)
                    return 0.0, f"LLM fallback: {e}"

        return 0.0, "LLM retries exhausted"

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
            if fp1.features.ast_structure_hash == fp2.features.ast_structure_hash:
                score += 0.4
            total_weight += 0.4

        # Control flow pattern similarity
        if fp1.features.control_flow_pattern or fp2.features.control_flow_pattern:
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
        if fp1.features.function_signatures or fp2.features.function_signatures:
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
        """Compute semantic similarity via embedding cosine distance.

        If the candidate's embedding is empty (retrieved from store),
        we skip this component and rely on structural + LLM scoring.
        """
        if not fp1.embedding.vector or not fp2.embedding.vector:
            return 0.0

        v1 = np.array(fp1.embedding.vector, dtype=np.float32)
        v2 = np.array(fp2.embedding.vector, dtype=np.float32)

        # Vectors must be same dimension
        if len(v1) != len(v2):
            return 0.0

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
        has_semantic: bool = True,
    ) -> float:
        """Combine scores into a single value in [0.0, 1.0].

        When the LLM returns high confidence (>0.7), it's the strongest
        signal — it means the code is logically equivalent. In that case,
        boost the LLM weight so the combined score reflects the LLM's
        assessment more strongly.

        If LLM was not invoked (confidence=0), redistribute its weight
        proportionally between structural and semantic.
        """
        s_weight = self.STRUCTURAL_WEIGHT
        e_weight = self.SEMANTIC_WEIGHT if has_semantic else 0.0
        l_weight = self.LLM_WEIGHT

        # If semantic is unavailable, split its weight
        if not has_semantic:
            s_weight += self.SEMANTIC_WEIGHT * 0.5
            l_weight += self.SEMANTIC_WEIGHT * 0.5

        # If LLM is not invoked, redistribute proportionally
        if llm_confidence == 0.0 and l_weight > 0:
            if has_semantic:
                s_weight += l_weight * 0.5
                e_weight += l_weight * 0.5
            else:
                s_weight += l_weight
            l_weight = 0.0
        elif llm_confidence >= 0.7:
            # LLM is highly confident — boost its weight as it's the
            # most reliable signal for logic equivalence
            if llm_confidence >= 0.9:
                # Near-identical code: LLM dominates the score
                boost = 0.25
            else:
                boost = 0.15
            s_weight -= boost * 0.5
            e_weight -= boost * 0.5
            l_weight += boost

        # Normalize weights to sum to 1.0
        total_weight = s_weight + e_weight + l_weight
        if total_weight > 0:
            s_weight /= total_weight
            e_weight /= total_weight
            l_weight /= total_weight

        combined = (
            structural * s_weight + semantic * e_weight + llm_confidence * l_weight
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

    def _sequence_similarity(self, seq1: list[str], seq2: list[str]) -> float:
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

    def _list_similarity(self, list1: list[str], list2: list[str]) -> float:
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
