"""Core data models and enums for the reuse detection pipeline."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class ChangeType(Enum):
    """Type of change detected in the diff."""

    ADDED = "added"
    MODIFIED = "modified"


class IndexSource(Enum):
    """Source tag for vector store entries."""

    GITHUB_LIBRARY = "github_library"
    LOCAL_CODEBASE = "local_codebase"


class Decision(Enum):
    """Pipeline decision outcome."""

    PASS = "pass"
    WARN = "warn"
    BLOCK = "block"


@dataclass
class CodeBlock:
    """A discrete unit of code that can be analyzed for reuse."""

    file_path: Path
    start_line: int
    end_line: int
    content: str
    change_type: ChangeType
    function_name: str | None = None
    class_name: str | None = None

    @property
    def line_count(self) -> int:
        """Number of lines in the code block."""
        return self.end_line - self.start_line + 1


@dataclass
class CodeFeatures:
    """Structural features extracted from a code block via AST analysis."""

    function_signatures: list[str] = field(default_factory=list)
    class_hierarchy: list[str] = field(default_factory=list)
    import_patterns: list[str] = field(default_factory=list)
    decorators: list[str] = field(default_factory=list)
    control_flow_pattern: str = ""
    parameter_types: list[str] = field(default_factory=list)
    return_type: str | None = None
    docstring: str | None = None
    ast_structure_hash: str = ""
    token_sequence: list[str] = field(default_factory=list)


@dataclass
class CodeEmbedding:
    """Dense semantic vector for a code block."""

    vector: list[float] = field(default_factory=list)
    dim: int = 0
    model_id: str = ""


@dataclass
class BlockFingerprint:
    """AST features AND dense embedding for a single CodeBlock."""

    block: CodeBlock
    features: CodeFeatures
    embedding: CodeEmbedding
    content_hash: str = ""


@dataclass
class SourceTaggedCandidate:
    """An ANN hit carrying its source tag."""

    source: IndexSource
    indexed_id: str
    distance: float
    fingerprint: BlockFingerprint


@dataclass
class ValidatedMatch:
    """A candidate after scoring and LLM validation."""

    source: IndexSource
    indexed_id: str
    structural_score: float
    semantic_score: float
    llm_confidence: float
    combined_score: float
    llm_rationale: str
    import_path: str
    existing_code: str


@dataclass
class ReuseSuggestion:
    """A suggestion to reuse existing code."""

    original_code_location: str
    existing_code_location: str
    import_statement: str
    usage_example: str
    confidence: float
    explanation: str
    diff_preview: str = ""
