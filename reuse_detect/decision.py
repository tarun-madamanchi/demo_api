"""Stage 6: Decision and Suggestion Rendering.

Applies deterministic thresholds to produce pass/warn/block decisions
and renders reuse suggestions with import paths and usage examples.
"""

import logging

from .config import DetectionConfig
from .models import CodeBlock, Decision, ReuseSuggestion, ValidatedMatch

logger = logging.getLogger(__name__)


class DecisionRenderer:
    """Deterministic thresholding + suggestion rendering."""

    def __init__(self, config: DetectionConfig):
        self.config = config
        self.block_threshold = config.block_threshold
        self.warn_threshold = config.warn_threshold
        self.max_suggestions_per_block = config.max_suggestions_per_block

    def decide_and_render(
        self,
        block: CodeBlock,
        validated: list[ValidatedMatch],
    ) -> tuple[Decision, list[ReuseSuggestion]]:
        """Apply thresholds to produce a Decision and render suggestions.

        Decision logic:
        - BLOCK if max combined_score >= block_threshold
        - WARN if max combined_score >= warn_threshold (but < block_threshold)
        - PASS if max combined_score < warn_threshold
        """
        if not validated:
            return Decision.PASS, []

        # Find the highest combined score
        max_score = max(v.combined_score for v in validated)

        # Determine decision
        if max_score >= self.block_threshold:
            decision = Decision.BLOCK
        elif max_score >= self.warn_threshold:
            decision = Decision.WARN
        else:
            decision = Decision.PASS

        # Render suggestions for WARN and BLOCK decisions
        suggestions: list[ReuseSuggestion] = []
        if decision in (Decision.WARN, Decision.BLOCK):
            # Sort by combined score descending
            sorted_matches = sorted(
                validated, key=lambda v: v.combined_score, reverse=True
            )

            # Limit to max_suggestions_per_block
            top_matches = sorted_matches[: self.max_suggestions_per_block]

            for match in top_matches:
                suggestion = self._render_suggestion(block, match)
                suggestions.append(suggestion)

        return decision, suggestions

    def decide_and_render_all(
        self,
        blocks: list[CodeBlock],
        all_validated: list[ValidatedMatch],
    ) -> tuple[Decision, list[ReuseSuggestion]]:
        """Decide across all blocks and aggregate suggestions.

        The overall decision is the most severe across all blocks.
        """
        if not all_validated:
            return Decision.PASS, []

        overall_decision = Decision.PASS
        all_suggestions: list[ReuseSuggestion] = []

        # Group validated matches by block (using indexed_id prefix or just use all)
        # For simplicity, treat all validated matches together
        max_score = max(v.combined_score for v in all_validated)

        if max_score >= self.block_threshold:
            overall_decision = Decision.BLOCK
        elif max_score >= self.warn_threshold:
            overall_decision = Decision.WARN

        if overall_decision in (Decision.WARN, Decision.BLOCK):
            sorted_matches = sorted(
                all_validated, key=lambda v: v.combined_score, reverse=True
            )

            # Limit total suggestions
            max_total = len(blocks) * self.max_suggestions_per_block
            top_matches = sorted_matches[:max_total]

            for match in top_matches:
                # Find the corresponding block (use first block as fallback)
                block = blocks[0] if blocks else None
                if block:
                    suggestion = self._render_suggestion(block, match)
                    all_suggestions.append(suggestion)

        return overall_decision, all_suggestions

    def _render_suggestion(
        self, block: CodeBlock, match: ValidatedMatch
    ) -> ReuseSuggestion:
        """Render a single ReuseSuggestion with all required fields."""
        original_location = f"{block.file_path}:{block.start_line}-{block.end_line}"

        # Derive existing code location from the match
        existing_location = match.indexed_id

        # Build usage example
        usage_example = self._build_usage_example(match)

        return ReuseSuggestion(
            original_code_location=original_location,
            existing_code_location=existing_location,
            import_statement=match.import_path,
            usage_example=usage_example,
            confidence=match.combined_score,
            explanation=match.llm_rationale,
            diff_preview=self._build_diff_preview(block, match),
        )

    def _build_usage_example(self, match: ValidatedMatch) -> str:
        """Build a usage example from the match."""
        # Extract the function/class name from the import path
        parts = match.import_path.split("import ")
        if len(parts) > 1:
            name = parts[1].strip()
            return (
                f"{match.import_path}\n\n# Use {name} instead of duplicating the logic"
            )
        return f"{match.import_path}\n\n# Reuse the existing implementation"

    def _build_diff_preview(self, block: CodeBlock, match: ValidatedMatch) -> str:
        """Build a diff preview showing what would change."""
        lines = [
            f"- # Your code in {block.file_path}:{block.start_line}",
            f"+ # Existing code: {match.indexed_id}",
            f"+ {match.import_path}",
        ]
        return "\n".join(lines)
