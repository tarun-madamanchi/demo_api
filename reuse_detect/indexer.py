"""Offline: Unified Indexer.

Sole writer to the unified source-tagged vector store.
Fetches configured GitHub common libraries, walks the local microservice
repo, builds a BlockFingerprint per function/class, and upserts rows
tagged LOCAL or GITHUB.
"""

import ast
import hashlib
import json
import logging
import shutil
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path

from .change_extractor import ChangeExtractor
from .config import DetectionConfig
from .feature_builder import FeatureEmbeddingBuilder
from .models import ChangeType, CodeBlock, IndexSource
from .vector_store import FAISSVectorStore

logger = logging.getLogger(__name__)


class UnifiedIndexer:
    """Sole writer to the unified source-tagged vector store.

    Fetches configured GitHub common libraries, walks the local microservice
    repo, builds a BlockFingerprint per function/class, and upserts rows
    tagged LOCAL or GITHUB.

    Runs on install, on a schedule, and incrementally on local repo change.
    """

    def __init__(
        self,
        config: DetectionConfig,
        builder: FeatureEmbeddingBuilder,
        store: FAISSVectorStore,
        api_token: str | None = None,
    ):
        self.config = config
        self.builder = builder
        self.store = store
        self.api_token = api_token
        self._last_github_refresh: datetime | None = None
        self._metadata_path = config.cache_dir / "index_metadata.json"

    def reindex_github(self, repositories: list[str] | None = None) -> int:
        """Fetch/refresh GitHub repos and upsert fingerprints with source=GITHUB.

        Args:
            repositories: List of GitHub repo URLs to index. If None, uses
                all configured repos (extracting URLs from GitHubRepoConfig).

        Returns the number of fingerprints indexed.
        """
        if repositories is not None:
            repos = repositories
        elif self.config.github_repositories:
            repos = [r.url for r in self.config.github_repositories]
        else:
            repos = []

        if not repos:
            logger.info("No GitHub repositories configured for indexing")
            return 0

        total_indexed = 0

        for repo_url in repos:
            try:
                blocks = self._fetch_github_repo_blocks(repo_url)
                if not blocks:
                    continue

                # Build fingerprints in batches
                fingerprints = self.builder.build_batch(blocks)

                for fp in fingerprints:
                    indexed_id = f"github:{repo_url}:{fp.content_hash}"
                    self.store.upsert(fp, indexed_id, IndexSource.GITHUB_LIBRARY)
                    total_indexed += 1

            except Exception as e:
                logger.warning(
                    "Failed to index GitHub repo %s: %s. "
                    "Falling back to cached index.",
                    repo_url,
                    e,
                )
                continue

        if total_indexed > 0:
            self.store.save()
            self._update_metadata("github_refresh", datetime.now().isoformat())

        logger.info("Indexed %d fingerprints from GitHub repositories", total_indexed)
        return total_indexed

    def reindex_local(self, repo_root: Path | None = None) -> int:
        """Walk local microservice repo and upsert fingerprints with source=LOCAL.

        Returns the number of fingerprints indexed.
        """
        root = repo_root or Path(".")
        total_indexed = 0

        python_files = self._find_python_files(root)

        for file_path in python_files:
            try:
                blocks = self._extract_blocks_from_file(file_path, root)
                if not blocks:
                    continue

                fingerprints = self.builder.build_batch(blocks)

                for fp in fingerprints:
                    relative_path = file_path.relative_to(root)
                    indexed_id = f"local:{relative_path}:{fp.content_hash}"
                    self.store.upsert(fp, indexed_id, IndexSource.LOCAL_CODEBASE)
                    total_indexed += 1

            except Exception as e:
                logger.warning("Failed to index %s: %s", file_path, e)
                continue

        if total_indexed > 0:
            self.store.save()
            self._update_metadata("local_refresh", datetime.now().isoformat())

        logger.info("Indexed %d fingerprints from local codebase", total_indexed)
        return total_indexed

    def reindex_incremental_local(self, changed_paths: list[Path]) -> int:
        """Upsert only fingerprints for changed local files.

        Returns the number of fingerprints indexed.
        """
        total_indexed = 0

        for file_path in changed_paths:
            if not file_path.exists():
                # File was deleted, remove from index
                self._remove_file_from_index(file_path)
                continue

            if not self._should_index_file(file_path):
                continue

            try:
                blocks = self._extract_blocks_from_file(file_path, file_path.parent)
                if not blocks:
                    continue

                fingerprints = self.builder.build_batch(blocks)

                for fp in fingerprints:
                    indexed_id = f"local:{file_path}:{fp.content_hash}"
                    self.store.upsert(fp, indexed_id, IndexSource.LOCAL_CODEBASE)
                    total_indexed += 1

            except Exception as e:
                logger.warning("Failed to incrementally index %s: %s", file_path, e)
                continue

        if total_indexed > 0:
            self.store.save()

        logger.info("Incrementally indexed %d fingerprints", total_indexed)
        return total_indexed

    def should_refresh_github(self) -> bool:
        """Check if the GitHub index needs refreshing based on interval."""
        metadata = self._load_metadata()
        last_refresh_str = metadata.get("github_refresh")

        if not last_refresh_str:
            return True

        try:
            last_refresh = datetime.fromisoformat(last_refresh_str)
            interval = timedelta(hours=self.config.index_refresh_interval_hours)
            return datetime.now() - last_refresh > interval
        except (ValueError, TypeError):
            return True

    def rebuild_from_scratch(self) -> None:
        """Delete corrupted cache and rebuild from scratch."""
        logger.warning("Rebuilding index from scratch")

        # Clear the store
        store_path = self.config.cache_dir
        if store_path.exists():
            shutil.rmtree(store_path)
        store_path.mkdir(parents=True, exist_ok=True)

        # Re-initialize store
        self.store._reset()

        # Reindex everything
        self.reindex_local()
        if self.config.github_repositories:
            self.reindex_github()

    def _fetch_github_repo_blocks(self, repo_url: str) -> list[CodeBlock]:
        """Fetch a GitHub repository and extract code blocks.

        Uses the GitHub API with authentication to fetch repository contents.
        """
        try:
            import httpx

            # Parse repo URL to get owner/repo
            parts = repo_url.rstrip("/").split("/")
            if len(parts) >= 2:
                owner, repo = parts[-2], parts[-1]
            else:
                logger.warning("Invalid repo URL format: %s", repo_url)
                return []

            # Build headers with authentication
            headers: dict[str, str] = {"Accept": "application/vnd.github.v3+json"}
            if self.api_token:
                headers["Authorization"] = f"Bearer {self.api_token}"
            else:
                logger.warning(
                    "No API token available for GitHub API calls. "
                    "Private repos will not be accessible."
                )

            # Use GitHub API to list Python files
            api_url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/main?recursive=1"
            response = httpx.get(api_url, headers=headers, timeout=30.0)

            if response.status_code == 401:
                logger.error(
                    "GitHub API authentication failed for %s. Check GITHUB_TOKEN.",
                    repo_url,
                )
                return []
            elif response.status_code == 403:
                logger.error(
                    "GitHub API access forbidden for %s. Token may lack repo scope.",
                    repo_url,
                )
                return []
            elif response.status_code == 404:
                logger.error(
                    "GitHub repo not found: %s. Check URL and token permissions.",
                    repo_url,
                )
                return []
            elif response.status_code != 200:
                logger.warning(
                    "GitHub API returned %d for %s",
                    response.status_code,
                    repo_url,
                )
                return []

            tree = response.json().get("tree", [])
            python_files = [
                item["path"]
                for item in tree
                if item["path"].endswith(".py") and item["type"] == "blob"
            ]

            logger.info(
                "Found %d Python files in %s/%s", len(python_files), owner, repo
            )

            blocks: list[CodeBlock] = []
            for file_path in python_files:
                content_url = (
                    f"https://raw.githubusercontent.com/{owner}/{repo}/main/{file_path}"
                )
                # raw.githubusercontent.com also needs auth for private repos
                content_headers: dict[str, str] = {}
                if self.api_token:
                    content_headers["Authorization"] = f"token {self.api_token}"

                content_response = httpx.get(
                    content_url, headers=content_headers, timeout=30.0
                )
                if content_response.status_code == 200:
                    file_blocks = self._parse_source_to_blocks(
                        content_response.text, Path(file_path)
                    )
                    blocks.extend(file_blocks)
                else:
                    logger.debug(
                        "Failed to fetch %s (status %d)",
                        file_path,
                        content_response.status_code,
                    )

            logger.info(
                "Extracted %d code blocks from %s/%s", len(blocks), owner, repo
            )
            return blocks

        except Exception as e:
            logger.warning("Failed to fetch GitHub repo %s: %s", repo_url, e)
            return []

    def _parse_source_to_blocks(self, source: str, file_path: Path) -> list[CodeBlock]:
        """Parse source code into CodeBlock objects for functions and classes."""
        blocks: list[CodeBlock] = []

        try:
            tree = ast.parse(source)
        except SyntaxError:
            return blocks

        source_lines = source.splitlines()

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                start_line = node.lineno
                end_line = node.end_lineno or node.lineno

                block_content = "\n".join(source_lines[start_line - 1 : end_line])

                # Skip trivial blocks
                if end_line - start_line + 1 < self.config.min_block_lines:
                    continue

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
                        content=block_content,
                        change_type=ChangeType.ADDED,
                        function_name=function_name,
                        class_name=class_name,
                    )
                )

        return blocks

    def _find_python_files(self, root: Path) -> list[Path]:
        """Find all Python files in the repository, respecting patterns."""
        import fnmatch

        python_files: list[Path] = []

        for file_path in root.rglob("*.py"):
            relative = file_path.relative_to(root)
            filename = file_path.name

            # Check exclude patterns
            excluded = False
            for pattern in self.config.exclude_patterns:
                if fnmatch.fnmatch(filename, pattern) or fnmatch.fnmatch(
                    str(relative), pattern
                ):
                    excluded = True
                    break

            if excluded:
                continue

            # Check include patterns
            if self.config.include_patterns:
                included = False
                for pattern in self.config.include_patterns:
                    if fnmatch.fnmatch(filename, pattern) or fnmatch.fnmatch(
                        str(relative), pattern
                    ):
                        included = True
                        break
                if not included:
                    continue

            python_files.append(file_path)

        return python_files

    def _extract_blocks_from_file(self, file_path: Path, root: Path) -> list[CodeBlock]:
        """Extract code blocks from a single file."""
        try:
            source = file_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            logger.warning("Cannot read %s: %s", file_path, e)
            return []

        relative_path = file_path.relative_to(root) if root else file_path
        return self._parse_source_to_blocks(source, relative_path)

    def _should_index_file(self, file_path: Path) -> bool:
        """Check if a file should be indexed based on patterns."""
        import fnmatch

        filename = file_path.name

        for pattern in self.config.exclude_patterns:
            if fnmatch.fnmatch(filename, pattern):
                return False

        if self.config.include_patterns:
            for pattern in self.config.include_patterns:
                if fnmatch.fnmatch(filename, pattern):
                    return True
            return False

        return True

    def _remove_file_from_index(self, file_path: Path) -> None:
        """Remove all entries for a file from the index."""
        prefix = f"local:{file_path}"
        ids_to_remove = [
            id_ for id_ in self.store._id_to_position.keys() if id_.startswith(prefix)
        ]
        for id_ in ids_to_remove:
            self.store.delete(id_)

    def _load_metadata(self) -> dict:
        """Load index metadata from disk."""
        if not self._metadata_path.exists():
            return {}
        try:
            return json.loads(self._metadata_path.read_text())
        except (json.JSONDecodeError, OSError):
            return {}

    def _update_metadata(self, key: str, value: str) -> None:
        """Update a metadata field and persist."""
        metadata = self._load_metadata()
        metadata[key] = value
        self._metadata_path.parent.mkdir(parents=True, exist_ok=True)
        self._metadata_path.write_text(json.dumps(metadata))
