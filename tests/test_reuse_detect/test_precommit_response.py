"""Tests for the PrecommitResponse."""

from io import StringIO

import pytest

from reuse_detect.models import Decision, ReuseSuggestion
from reuse_detect.precommit_response import PrecommitResponse


def make_suggestion() -> ReuseSuggestion:
    return ReuseSuggestion(
        original_code_location="src/utils.py:10-20",
        existing_code_location="lib/common.py:5-15",
        import_statement="from lib.common import helper",
        usage_example="from lib.common import helper\n\n# Use helper instead",
        confidence=0.92,
        explanation="These functions perform the same computation",
        diff_preview="- your code\n+ existing code",
    )


class TestExitCodes:
    """Test exit code correctness."""

    def test_pass_returns_0(self):
        output = StringIO()
        response = PrecommitResponse(output_stream=output)
        code = response.respond(Decision.PASS, [])
        assert code == 0

    def test_warn_returns_0(self):
        output = StringIO()
        response = PrecommitResponse(output_stream=output)
        code = response.respond(Decision.WARN, [make_suggestion()])
        assert code == 0

    def test_block_returns_nonzero(self):
        output = StringIO()
        response = PrecommitResponse(output_stream=output)
        code = response.respond(Decision.BLOCK, [make_suggestion()])
        assert code != 0


class TestStderrOutput:
    """Test stderr output formatting."""

    def test_pass_produces_no_output(self):
        output = StringIO()
        response = PrecommitResponse(output_stream=output)
        response.respond(Decision.PASS, [])
        assert output.getvalue() == ""

    def test_warn_writes_suggestions(self):
        output = StringIO()
        response = PrecommitResponse(output_stream=output)
        suggestion = make_suggestion()
        response.respond(Decision.WARN, [suggestion])
        text = output.getvalue()
        assert "WARNING" in text
        assert suggestion.import_statement in text
        assert suggestion.usage_example in text

    def test_block_writes_suggestions(self):
        output = StringIO()
        response = PrecommitResponse(output_stream=output)
        suggestion = make_suggestion()
        response.respond(Decision.BLOCK, [suggestion])
        text = output.getvalue()
        assert "BLOCKED" in text
        assert suggestion.import_statement in text
        assert suggestion.usage_example in text

    def test_block_contains_override_hint(self):
        output = StringIO()
        response = PrecommitResponse(output_stream=output)
        response.respond(Decision.BLOCK, [make_suggestion()])
        text = output.getvalue()
        assert "--no-verify" in text
