"""Pipeline orchestrator.

Wires stages 2-7 together: extract → build → retrieve → score → decide → respond.
Handles the per-block loop with LLM call cap tracking across all blocks.
"""

import logging
from pathlib import Path

from .candidate_retriever import CandidateRetriever
from .change_extractor import ChangeExtractor
from .config import DetectionConfig
from .decision import DecisionRenderer
from .dependency_checker import DependencyChecker
from .feature_builder import EmbeddingProvider, FeatureEmbeddingBuilder
from .indexer import UnifiedIndexer
from .models import (
    BlockFingerprint,
    Decision,
    IndexSource,
    ReuseSuggestion,
    ValidatedMatch,
)
from .precommit_response import PrecommitResponse
from .scorer import HybridScorerLLMValidator, LLMProvider
from .vector_store import FAISSVectorStore

logger = logging.getLogger(__name__)


def get_indexable_repos(config: DetectionConfig, repo_root: Path | None = None) -> list:
    """Filter configured GitHub repos through DependencyChecker.

    Only repos whose package is declared in pyproject.toml are returned.
    This implements the dependency-gated library indexing from the design.
    """
    if not config.github_repositories:
        return []

    root = repo_root or Path(".")
    pyproject_path = root / config.pyproject_path

    checker = DependencyChecker(pyproject_path)
    return checker.filter_indexable_repos(config.github_repositories)


def detect_reusable_code(
    config: DetectionConfig,
    embedding_provider: EmbeddingProvider,
    llm_provider: LLMProvider | None = None,
    repo_root: Path | None = None,
    base_ref: str = "HEAD",
) -> tuple[Decision, list[ReuseSuggestion]]:
    """Main entry point for reuse detection on pre-commit.

    Orchestrates the full pipeline:
    1. Filter GitHub repos through DependencyChecker
    2. Extract changed code blocks from staged diff
    3. Build fingerprints (AST features + embeddings)
    4. Retrieve candidates from vector store
    5. Score and validate with hybrid scorer + LLM
    6. Decide (pass/warn/block) and render suggestions

    Returns (Decision, list[ReuseSuggestion]).
    """
    root = repo_root or Path(".")

    # Dependency-gated library indexing: only index repos that are
    # declared as dependencies in pyproject.toml
    indexable_repos = get_indexable_repos(config, root)
    logger.info(
        "Dependency check: %d of %d configured repos are project dependencies",
        len(indexable_repos),
        len(config.github_repositories),
    )

    # Initialize components
    extractor = ChangeExtractor(config, repo_root=root)
    builder = FeatureEmbeddingBuilder(embedding_provider)
    store = FAISSVectorStore(
        store_path=config.cache_dir / "vectors",
        dimension=embedding_provider.dimension,
    )
    store.load()

    retriever = CandidateRetriever(store)
    scorer = HybridScorerLLMValidator(config, llm_provider)
    renderer = DecisionRenderer(config)

    logger.info(
        "Vector store loaded with %d entries", store.size
    )

    # Stage 2: Extract changed blocks
    try:
        blocks = extractor.extract(base_ref=base_ref)
    except Exception as e:
        logger.error("Failed to extract changes: %s", e)
        return Decision.PASS, []

    # Handle empty diff
    if not blocks:
        logger.info("No meaningful code changes detected. PASS.")
        return Decision.PASS, []

    # Filter trivial blocks (already done in extract, but be explicit)
    blocks = [b for b in blocks if not extractor.is_trivial(b)]

    if not blocks:
        logger.info("All blocks are trivial. PASS.")
        return Decision.PASS, []

    all_validated: list[ValidatedMatch] = []
    scorer.reset_llm_counter()

    # Stages 3-5: For each block
    for block in blocks:
        try:
            # Stage 3: Build fingerprint
            fp = builder.build(block)

            # Stage 4: Retrieve candidates
            candidates = retriever.query(
                fp,
                sources=[IndexSource.LOCAL_CODEBASE, IndexSource.GITHUB_LIBRARY],
                top_k=config.top_k,
            )

            if not candidates:
                continue

            # Stage 5: Score and validate
            validated = scorer.score_and_validate(fp, candidates)
            all_validated.extend(validated)

        except SyntaxError as e:
            logger.warning(
                "Skipping unparseable block %s:%d-%d: %s",
                block.file_path,
                block.start_line,
                block.end_line,
                e,
            )
            continue
        except Exception as e:
            logger.warning(
                "Error processing block %s:%d-%d: %s",
                block.file_path,
                block.start_line,
                block.end_line,
                e,
            )
            continue

    # Stage 6: Decision
    if not all_validated:
        return Decision.PASS, []

    decision, suggestions = renderer.decide_and_render_all(blocks, all_validated)

    return decision, suggestions


def ensure_index_fresh(
    config: DetectionConfig,
    embedding_provider: EmbeddingProvider,
    repo_root: Path | None = None,
    api_token: str | None = None,
) -> None:
    """Ensure the vector store index is up-to-date before running detection.

    Checks if the GitHub index needs refreshing based on the configured
    interval and triggers re-indexing of dependency-gated repos if needed.
    """
    root = repo_root or Path(".")

    builder = FeatureEmbeddingBuilder(embedding_provider)
    store = FAISSVectorStore(
        store_path=config.cache_dir / "vectors",
        dimension=embedding_provider.dimension,
    )
    store.load()

    indexer = UnifiedIndexer(config, builder, store, api_token=api_token)

    # Always reindex local on first run or if store is empty
    if store.size == 0:
        logger.info("Vector store is empty. Running full initial indexing.")
        indexer.reindex_local(root)
        indexable_repos = get_indexable_repos(config, root)
        if indexable_repos:
            repo_urls = [r.url for r in indexable_repos]
            logger.info("Indexing %d GitHub repos: %s", len(repo_urls), repo_urls)
            indexed = indexer.reindex_github(repo_urls)
            if indexed == 0:
                logger.warning(
                    "GitHub indexing returned 0 blocks. Check token and repo access."
                )
        else:
            logger.warning(
                "No GitHub repos matched project dependencies. "
                "Configured repos: %s",
                [r.url for r in config.github_repositories],
            )
        return

    # Check if GitHub index needs refreshing
    if indexer.should_refresh_github():
        indexable_repos = get_indexable_repos(config, root)
        if indexable_repos:
            repo_urls = [r.url for r in indexable_repos]
            logger.info("Refreshing GitHub index for %d repos", len(repo_urls))
            indexer.reindex_github(repo_urls)


def run_precommit(
    config: DetectionConfig,
    embedding_provider: EmbeddingProvider,
    llm_provider: LLMProvider | None = None,
    repo_root: Path | None = None,
    base_ref: str = "HEAD",
    api_token: str | None = None,
) -> int:
    """Run the full pre-commit pipeline and return the exit code.

    This is the entry point called by the CLI.
    Returns 0 for PASS/WARN, non-zero for BLOCK.
    """
    # Ensure index is fresh before detection
    try:
        ensure_index_fresh(config, embedding_provider, repo_root, api_token=api_token)
    except Exception as e:
        logger.error(
            "Index refresh failed: %s. Detection will use existing index "
            "(which may be empty).",
            e,
        )

    decision, suggestions = detect_reusable_code(
        config=config,
        embedding_provider=embedding_provider,
        llm_provider=llm_provider,
        repo_root=repo_root,
        base_ref=base_ref,
    )

    # Stage 7: Pre-commit response
    response = PrecommitResponse()
    return response.respond(decision, suggestions)
