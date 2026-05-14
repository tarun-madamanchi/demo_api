"""Stage 2: Change Extractor.

Parses git diff, expands to whole functions/classes, drops trivial blocks,
and normalizes source code.
"""

import ast
import fnmatch
import logging
import re
import subprocess
import textwrap
from pathlib import Path

from .config import DetectionConfig
from .models import ChangeType, CodeBlock

logger = logging.getLogger(__name__)


class ChangeExtractor:
    """Parses git diff, expands to whole functions/classes, normalizes source."""

    def __init__(self, config: DetectionConfig, repo_root: Path | None = None):
        self.config = config
        self.repo_root = repo_root or Path(".")

    def extract(self, base_ref: str = "HEAD") -> list[CodeBlock]:
        """Extract all meaningful code blocks from the staged diff.

        Parses the staged git diff, identifies changed files, expands hunks
        to complete function/class definitions, and filters trivial blocks.
        """
        diff_output = self._get_staged_diff(base_ref)
        if not diff_output.strip():
            logger.info("No staged changes found (git diff --staged is empty)")
            return []

        changed_files = self._parse_diff_files(diff_output)
        logger.info("Found %d changed files in staged diff", len(changed_files))
        blocks: list[CodeBlock] = []

        for file_path, changed_lines, change_type in changed_files:
            if not self._should_process_file(file_path):
                logger.debug("Skipping %s (excluded by patterns)", file_path)
                continue

            try:
                file_blocks = self._extract_blocks_from_file(
                    file_path, changed_lines, change_type
                )
                logger.debug(
                    "Extracted %d blocks from %s (changed lines: %s)",
                    len(file_blocks),
                    file_path,
                    changed_lines[:10],
                )
                blocks.extend(file_blocks)
            except (SyntaxError, OSError) as e:
                logger.warning(
                    "Failed to parse %s: %s (lines %s)",
                    file_path,
                    e,
                    changed_lines,
                )
                continue

        # Filter trivial blocks
        non_trivial = [b for b in blocks if not self.is_trivial(b)]
        logger.info(
            "Extracted %d blocks total, %d non-trivial",
            len(blocks),
            len(non_trivial),
        )
        return non_trivial

    def is_trivial(self, block: CodeBlock) -> bool:
        """Return True if block is too small or comments/whitespace-only."""
        if block.line_count < self.config.min_block_lines:
            return True

        # Check if content is only comments and whitespace
        lines = block.content.strip().splitlines()
        meaningful_lines = [
            line for line in lines if line.strip() and not line.strip().startswith("#")
        ]
        if not meaningful_lines:
            return True

        return False

    def normalize(self, source: str) -> str:
        """Normalize source code to remove formatting variations.

        Normalization is idempotent: normalize(normalize(x)) == normalize(x).
        """
        # Dedent the source
        source = textwrap.dedent(source)

        # Normalize line endings
        source = source.replace("\r\n", "\n").replace("\r", "\n")

        # Remove trailing whitespace from each line
        lines = [line.rstrip() for line in source.splitlines()]

        # Remove leading/trailing blank lines
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()

        # Collapse multiple blank lines into one
        normalized_lines: list[str] = []
        prev_blank = False
        for line in lines:
            if not line.strip():
                if not prev_blank:
                    normalized_lines.append("")
                prev_blank = True
            else:
                normalized_lines.append(line)
                prev_blank = False

        return "\n".join(normalized_lines)

    def _get_staged_diff(self, base_ref: str) -> str:
        """Run git diff --staged and return the output."""
        try:
            result = subprocess.run(
                ["git", "diff", "--staged", "--unified=0", base_ref],
                capture_output=True,
                text=True,
                cwd=str(self.repo_root),
                check=True,
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            # If base_ref doesn't exist (e.g., initial commit), try without it
            if "unknown revision" in (e.stderr or ""):
                logger.debug(
                    "Base ref '%s' not found, trying diff without base ref",
                    base_ref,
                )
                result = subprocess.run(
                    ["git", "diff", "--staged", "--unified=0"],
                    capture_output=True,
                    text=True,
                    cwd=str(self.repo_root),
                )
                return result.stdout
            # If there's nothing staged, git may return non-zero
            logger.debug("git diff --staged failed: %s", e.stderr)
            return ""

    def _parse_diff_files(
        self, diff_output: str
    ) -> list[tuple[Path, list[int], ChangeType]]:
        """Parse diff output to extract changed files and line numbers."""
        files: list[tuple[Path, list[int], ChangeType]] = []
        current_file: str | None = None
        changed_lines: list[int] = []
        is_new_file = False

        for line in diff_output.splitlines():
            if line.startswith("diff --git"):
                if current_file is not None:
                    change_type = (
                        ChangeType.ADDED if is_new_file else ChangeType.MODIFIED
                    )
                    files.append((Path(current_file), changed_lines, change_type))
                # Extract file path from "diff --git a/path b/path"
                parts = line.split(" b/", 1)
                current_file = parts[1] if len(parts) > 1 else None
                changed_lines = []
                is_new_file = False
            elif line.startswith("new file mode"):
                is_new_file = True
            elif line.startswith("@@"):
                # Parse hunk header: @@ -old,count +new,count @@
                match = re.search(r"\+(\d+)(?:,(\d+))?", line)
                if match:
                    start = int(match.group(1))
                    count = int(match.group(2)) if match.group(2) else 1
                    changed_lines.extend(range(start, start + count))

        # Don't forget the last file
        if current_file is not None:
            change_type = ChangeType.ADDED if is_new_file else ChangeType.MODIFIED
            files.append((Path(current_file), changed_lines, change_type))

        return files

    def _should_process_file(self, file_path: Path) -> bool:
        """Check if a file matches include/exclude patterns."""
        filename = file_path.name
        # Use forward slashes for cross-platform pattern matching
        filepath_str = str(file_path).replace("\\", "/")

        # Check exclude patterns first
        for pattern in self.config.exclude_patterns:
            if fnmatch.fnmatch(filename, pattern) or fnmatch.fnmatch(
                filepath_str, pattern
            ):
                return False

        # Check include patterns
        if self.config.include_patterns:
            for pattern in self.config.include_patterns:
                if fnmatch.fnmatch(filename, pattern) or fnmatch.fnmatch(
                    filepath_str, pattern
                ):
                    return True
            return False

        return True

    def _extract_blocks_from_file(
        self, file_path: Path, changed_lines: list[int], change_type: ChangeType
    ) -> list[CodeBlock]:
        """Extract complete function/class blocks that overlap with changed lines."""
        full_path = self.repo_root / file_path
        source = full_path.read_text(encoding="utf-8")

        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            logger.warning("AST parse failed for %s: %s", file_path, e)
            raise

        blocks: list[CodeBlock] = []
        changed_set = set(changed_lines)

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                # Get the full range of the node
                start_line = node.lineno
                end_line = node.end_lineno or node.lineno

                # Check if any changed line falls within this node
                node_lines = set(range(start_line, end_line + 1))
                if not node_lines.intersection(changed_set):
                    continue

                # Extract the complete definition
                source_lines = source.splitlines()
                block_content = "\n".join(source_lines[start_line - 1 : end_line])

                # Normalize the content
                normalized_content = self.normalize(block_content)

                function_name = (
                    node.name
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    else None
                )
                class_name = node.name if isinstance(node, ast.ClassDef) else None

                blocks.append(
                    CodeBlock(
                        file_path=file_path,
                        start_line=start_line,
                        end_line=end_line,
                        content=normalized_content,
                        change_type=change_type,
                        function_name=function_name,
                        class_name=class_name,
                    )
                )

        # Remove parent blocks when a more specific child block exists.
        # If a class contains a function that was also extracted, drop the class
        # block — the function is a more precise unit for comparison.
        if len(blocks) > 1:
            blocks = self._remove_parent_blocks(blocks)

        return blocks

    def _remove_parent_blocks(self, blocks: list[CodeBlock]) -> list[CodeBlock]:
        """Remove blocks that fully contain other blocks.

        When a class and its method are both extracted, keep only the method
        since it's a more precise unit for reuse comparison.
        """
        # Sort by size (smallest first)
        sorted_blocks = sorted(blocks, key=lambda b: b.line_count)
        result: list[CodeBlock] = []

        for block in sorted_blocks:
            # Check if this block fully contains any already-accepted block
            is_parent = any(
                block.start_line <= other.start_line
                and block.end_line >= other.end_line
                and block is not other
                for other in sorted_blocks
                if other.line_count < block.line_count
            )
            if not is_parent:
                result.append(block)

        return result
