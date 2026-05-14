"""Stage 7: Pre-commit Response.

Writes output to stderr and controls the process exit code.
"""

import sys
from io import StringIO

from .models import Decision, ReuseSuggestion


class PrecommitResponse:
    """Writes output and controls the exit code."""

    def __init__(self, output_stream=None):
        """Initialize with an optional output stream (defaults to stderr)."""
        self.output_stream = output_stream or sys.stderr

    def respond(
        self, decision: Decision, suggestions: list[ReuseSuggestion]
    ) -> int:
        """Write suggestions to stderr. Return 0 on pass/warn, non-zero on block.

        - BLOCK: writes suggestions to stderr, returns exit code 1
        - WARN: writes suggestions to stderr, returns exit code 0
        - PASS: exits silently with code 0
        """
        if decision == Decision.PASS:
            return 0

        if decision == Decision.BLOCK:
            self._write_block_message(suggestions)
            return 1

        if decision == Decision.WARN:
            self._write_warn_message(suggestions)
            return 0

        return 0

    def _write_block_message(self, suggestions: list[ReuseSuggestion]) -> None:
        """Write a BLOCK message with suggestions to stderr."""
        self.output_stream.write(
            "\n" + "=" * 70 + "\n"
        )
        self.output_stream.write(
            "🚫 COMMIT BLOCKED: Duplicate code detected\n"
        )
        self.output_stream.write("=" * 70 + "\n\n")
        self.output_stream.write(
            "The following code duplicates existing library/codebase code.\n"
            "Please reuse the existing implementation instead.\n\n"
        )

        self._write_suggestions(suggestions)

        self.output_stream.write(
            "\nTo proceed, refactor your code to use the suggested imports.\n"
            "To override, use: git commit --no-verify\n"
        )

    def _write_warn_message(self, suggestions: list[ReuseSuggestion]) -> None:
        """Write a WARN message with suggestions to stderr."""
        self.output_stream.write(
            "\n" + "-" * 70 + "\n"
        )
        self.output_stream.write(
            "⚠️  WARNING: Potential code duplication detected\n"
        )
        self.output_stream.write("-" * 70 + "\n\n")
        self.output_stream.write(
            "Consider reusing the following existing implementations:\n\n"
        )

        self._write_suggestions(suggestions)

        self.output_stream.write(
            "\nCommit will proceed, but consider refactoring.\n"
        )

    def _write_suggestions(self, suggestions: list[ReuseSuggestion]) -> None:
        """Write formatted suggestions."""
        for i, suggestion in enumerate(suggestions, 1):
            self.output_stream.write(f"  [{i}] {suggestion.original_code_location}\n")
            self.output_stream.write(
                f"      Matches: {suggestion.existing_code_location}\n"
            )
            self.output_stream.write(
                f"      Confidence: {suggestion.confidence:.0%}\n"
            )
            self.output_stream.write(
                f"      Import: {suggestion.import_statement}\n"
            )
            self.output_stream.write(
                f"      Usage:\n        {suggestion.usage_example}\n"
            )
            self.output_stream.write(
                f"      Reason: {suggestion.explanation}\n"
            )
            self.output_stream.write("\n")
