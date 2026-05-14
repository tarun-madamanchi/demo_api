"""CLI entry point for the reuse detection tool.

Provides the `reuse-detect` command with support for:
- --staged: Run detection on staged changes (pre-commit mode)
- --reindex: Manually trigger re-indexing
- --config: Path to configuration file
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from .config import DetectionConfig
from .dependency_checker import DependencyChecker
from .feature_builder import GitHubModelsEmbeddingProvider
from .indexer import UnifiedIndexer
from .local_embedding import LocalEmbeddingProvider
from .models import Decision
from .pipeline import detect_reusable_code, get_indexable_repos, run_precommit
from .precommit_response import PrecommitResponse
from .scorer import GitHubModelsLLMProvider
from .vector_store import FAISSVectorStore

# Load .env from the package directory (where the token lives),
# then also try the repo root .env as a fallback.
_package_dir = Path(__file__).resolve().parent
load_dotenv(_package_dir / ".env")
load_dotenv()  # Also check CWD/.env as fallback


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog="reuse-detect",
        description="Detect duplicate or reusable code before committing.",
    )

    parser.add_argument(
        "--staged",
        action="store_true",
        help="Run detection on staged changes (pre-commit mode)",
    )

    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Manually trigger re-indexing of the vector store",
    )

    parser.add_argument(
        "--reindex-local",
        action="store_true",
        help="Re-index only the local codebase",
    )

    parser.add_argument(
        "--reindex-github",
        action="store_true",
        help="Re-index only GitHub repositories",
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path(".reuse-detect.yml"),
        help="Path to configuration file (default: .reuse-detect.yml)",
    )

    parser.add_argument(
        "--base-ref",
        type=str,
        default="HEAD",
        help="Base git reference for diff comparison (default: HEAD)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the CLI."""
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
        stream=sys.stderr,
    )


def get_api_token() -> str | None:
    """Get the GitHub API token from environment."""
    return os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")


def main(argv: list[str] | None = None) -> int:
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args(argv)

    setup_logging(args.verbose)

    # Load configuration
    config = DetectionConfig.from_file(args.config)

    # Get API token
    api_token = get_api_token()

    if not api_token:
        logging.getLogger(__name__).warning(
            "No GITHUB_TOKEN found. Embedding API calls will fail. "
            "Falling back to local embedding provider."
        )

    # Initialize providers — use GitHub Models API if token available,
    # otherwise fall back to local offline embeddings.
    if api_token:
        embedding_provider = GitHubModelsEmbeddingProvider(api_token=api_token)
    else:
        embedding_provider = LocalEmbeddingProvider()

    llm_provider = GitHubModelsLLMProvider(api_token=api_token) if api_token else None

    # Handle reindex commands
    if args.reindex or args.reindex_local or args.reindex_github:
        return _handle_reindex(config, embedding_provider, args, api_token=api_token)

    # Default: run staged detection (pre-commit mode)
    if args.staged or not any([args.reindex, args.reindex_local, args.reindex_github]):
        return run_precommit(
            config=config,
            embedding_provider=embedding_provider,
            llm_provider=llm_provider,
            base_ref=args.base_ref,
            api_token=api_token,
        )

    return 0


def _handle_reindex(
    config: DetectionConfig,
    embedding_provider,
    args,
    api_token: str | None = None,
) -> int:
    """Handle re-indexing commands.

    GitHub re-indexing is dependency-gated: only repos whose package is
    declared in pyproject.toml are indexed. This prevents unusable
    suggestions from libraries the project cannot import.
    """
    from .feature_builder import FeatureEmbeddingBuilder

    builder = FeatureEmbeddingBuilder(embedding_provider)
    store = FAISSVectorStore(
        store_path=config.cache_dir / "vectors",
        dimension=embedding_provider.dimension,
    )
    store.load()

    indexer = UnifiedIndexer(config, builder, store, api_token=api_token)

    if args.reindex:
        # Full reindex: local + dependency-gated GitHub repos
        indexer.reindex_local()
        indexable_repos = get_indexable_repos(config)
        if indexable_repos:
            repo_urls = [r.url for r in indexable_repos]
            indexer.reindex_github(repo_urls)
            print(
                f"Re-indexing complete. Indexed {len(repo_urls)} GitHub repos "
                f"(of {len(config.github_repositories)} configured).",
                file=sys.stderr,
            )
        else:
            print(
                "Re-indexing complete. No GitHub repos matched project dependencies.",
                file=sys.stderr,
            )
    elif args.reindex_local:
        indexer.reindex_local()
        print("Local re-indexing complete.", file=sys.stderr)
    elif args.reindex_github:
        indexable_repos = get_indexable_repos(config)
        if indexable_repos:
            repo_urls = [r.url for r in indexable_repos]
            indexer.reindex_github(repo_urls)
            print(
                f"GitHub re-indexing complete. Indexed {len(repo_urls)} repos "
                f"(of {len(config.github_repositories)} configured).",
                file=sys.stderr,
            )
        else:
            print(
                "No GitHub repos matched project dependencies. Nothing to index.",
                file=sys.stderr,
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
