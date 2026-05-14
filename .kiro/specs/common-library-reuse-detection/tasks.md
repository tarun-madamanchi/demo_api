# Implementation Plan: Common Library Reuse Detection

## Overview

This plan implements an agent-driven pipeline that detects duplicate or reusable code before a developer commits changes. The pipeline runs as a pre-commit hook combining AST structural analysis, semantic embeddings (GitHub Models API), vector search (FAISS/ChromaDB), and LLM validation. Implementation proceeds bottom-up: configuration and data models first, then each pipeline stage in order, then the offline indexer, and finally the CLI entry point and pre-commit integration.

## Tasks

- [x] 1. Set up project structure and configuration
  - [x] 1.1 Create package structure and DetectionConfig dataclass
    - Create `reuse_detect/` package with `__init__.py`
    - Create `reuse_detect/config.py` with the `DetectionConfig` dataclass including all fields: `github_repositories`, `similarity_threshold`, `block_threshold`, `warn_threshold`, `min_block_lines`, `cache_dir`, `index_refresh_interval_hours`, `exclude_patterns`, `include_patterns`, `max_suggestions_per_block`, `top_k`, `llm_timeout_seconds`, `max_llm_calls_per_commit`
    - Implement config loading from a YAML/TOML file with defaults fallback
    - _Requirements: 11.1, 11.4_

  - [x] 1.2 Define core data models and enums
    - Create `reuse_detect/models.py` with `CodeBlock`, `ChangeType`, `CodeFeatures`, `CodeEmbedding`, `BlockFingerprint`, `IndexSource`, `SourceTaggedCandidate`, `ValidatedMatch`, `Decision`, `ReuseSuggestion` dataclasses and enums
    - _Requirements: 1.1, 2.1, 3.1, 4.4, 5.4_

  - [x]* 1.3 Write unit tests for configuration and data models
    - Test DetectionConfig defaults and overrides
    - Test dataclass construction and field validation
    - _Requirements: 11.1, 11.4_

- [x] 2. Implement ChangeExtractor (Stage 2)
  - [x] 2.1 Implement git diff parsing and CodeBlock extraction
    - Create `reuse_detect/change_extractor.py` with `ChangeExtractor` class
    - Implement `extract()` method that runs `git diff --staged`, parses hunks, and expands partial hunks to complete function/class definitions using the Python `ast` module
    - Implement file filtering based on `include_patterns` and `exclude_patterns` from config
    - _Requirements: 1.1, 1.4, 11.2, 11.3_

  - [x] 2.2 Implement trivial block filtering and normalization
    - Implement `is_trivial()` method checking `min_block_lines` threshold and comments/whitespace-only content
    - Implement source normalization (strip formatting variations, consistent whitespace)
    - Ensure normalization is idempotent
    - _Requirements: 1.2, 1.3, 1.5_

  - [ ]* 2.3 Write property test for trivial block exclusion
    - **Property 8: Trivial Block Exclusion**
    - **Validates: Requirements 1.2, 1.3**

  - [ ]* 2.4 Write property test for function/class expansion
    - **Property 17: Function and Class Expansion**
    - **Validates: Requirement 1.4**

  - [ ]* 2.5 Write property test for normalization idempotence
    - **Property 18: Normalization Idempotence**
    - **Validates: Requirement 1.5**

  - [ ]* 2.6 Write property test for pattern-based file filtering
    - **Property 19: Pattern-Based File Filtering**
    - **Validates: Requirements 11.2, 11.3**

- [x] 3. Implement FeatureEmbeddingBuilder (Stage 3)
  - [x] 3.1 Implement AST feature extraction
    - Create `reuse_detect/feature_builder.py` with `FeatureEmbeddingBuilder` class
    - Implement single-pass AST walk extracting: function signatures, class hierarchy, import patterns, decorators, control flow patterns, parameter types, return types, AST structure hash
    - _Requirements: 2.1, 2.2_

  - [x] 3.2 Implement embedding API integration
    - Implement `build()` method that calls GitHub Models API (`models.github.ai`) for dense semantic embedding
    - Implement `build_batch()` method for batched embedding calls (offline indexer path)
    - Combine AST features and embedding into a `BlockFingerprint`
    - _Requirements: 2.3, 2.4_

  - [x]* 3.3 Write unit tests for AST feature extraction
    - Test extraction of known Python patterns (functions, classes, decorators, control flow)
    - Test structure hash consistency
    - _Requirements: 2.1, 2.2_

- [x] 4. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. Implement CandidateRetriever (Stage 4)
  - [x] 5.1 Implement vector store abstraction and FAISS/ChromaDB backend
    - Create `reuse_detect/vector_store.py` with a `VectorStore` protocol/abstract class
    - Implement FAISS-based backend with source-tagged metadata storage
    - Support upsert, delete, and ANN query operations
    - _Requirements: 3.1, 3.2, 3.3_

  - [x] 5.2 Implement CandidateRetriever query logic
    - Create `reuse_detect/candidate_retriever.py` with `CandidateRetriever` class
    - Implement `query()` method issuing a single ANN query against the unified store
    - Return up to `top_k` `SourceTaggedCandidate` results tagged with `IndexSource`
    - _Requirements: 3.1, 3.2, 3.3_

  - [ ]* 5.3 Write property test for top-K retrieval bound
    - **Property 12: Top-K Retrieval Bound**
    - **Validates: Requirement 3.1**

  - [ ]* 5.4 Write property test for source coverage
    - **Property 15: Source Coverage**
    - **Validates: Requirements 9.3, 3.2**

- [x] 6. Implement HybridScorerLLMValidator (Stage 5)
  - [x] 6.1 Implement structural and semantic scoring
    - Create `reuse_detect/scorer.py` with `HybridScorerLLMValidator` class
    - Implement structural similarity computation from AST features (rename-invariant, comment-invariant)
    - Implement semantic similarity via embedding cosine distance
    - _Requirements: 4.1, 4.2, 8.4, 8.5_

  - [x] 6.2 Implement LLM logic-equivalence validation
    - Implement LLM call to GitHub Models API for logic-equivalence check
    - Implement LLM call cap tracking (max_llm_calls_per_commit)
    - Implement timeout handling and fallback to structural+semantic only
    - _Requirements: 4.3, 9.4, 10.4_

  - [x] 6.3 Implement combined score computation
    - Combine structural, semantic, and LLM confidence into a single `combined_score` in [0.0, 1.0]
    - Produce `ValidatedMatch` objects with all scoring fields
    - _Requirements: 4.4, 4.5_

  - [ ]* 6.4 Write property test for score range invariant
    - **Property 6: Score Range Invariant**
    - **Validates: Requirement 4.5**

  - [ ]* 6.5 Write property test for self-match identity
    - **Property 2: Self-Match Identity**
    - **Validates: Requirement 8.2**

  - [ ]* 6.6 Write property test for symmetry
    - **Property 3: Symmetry**
    - **Validates: Requirement 8.3**

  - [ ]* 6.7 Write property test for rename invariance
    - **Property 4: Rename Invariance**
    - **Validates: Requirement 8.4**

  - [ ]* 6.8 Write property test for comment invariance
    - **Property 5: Comment Invariance**
    - **Validates: Requirement 8.5**

  - [ ]* 6.9 Write property test for LLM call cap
    - **Property 16: LLM Call Cap**
    - **Validates: Requirement 9.4**

- [x] 7. Implement DecisionRenderer (Stage 6)
  - [x] 7.1 Implement threshold-based decision logic
    - Create `reuse_detect/decision.py` with `DecisionRenderer` class
    - Implement `decide_and_render()` applying block_threshold and warn_threshold to produce PASS/WARN/BLOCK
    - Limit suggestions per block to `max_suggestions_per_block`
    - _Requirements: 5.1, 5.2, 5.3, 5.5_

  - [x] 7.2 Implement ReuseSuggestion rendering
    - Render suggestions with all required fields: original code location, existing code location, import statement, usage example, confidence score, explanation
    - _Requirements: 5.4_

  - [ ]* 7.3 Write property test for threshold-based decision
    - **Property 7: Threshold-Based Decision**
    - **Validates: Requirements 5.1, 5.2, 5.3**

  - [ ]* 7.4 Write property test for suggestion count bound
    - **Property 10: Suggestion Count Bound**
    - **Validates: Requirement 5.5**

  - [ ]* 7.5 Write property test for suggestion field completeness
    - **Property 14: Suggestion Field Completeness**
    - **Validates: Requirement 5.4**

- [x] 8. Implement PrecommitResponse (Stage 7)
  - [x] 8.1 Implement exit code logic and stderr output
    - Create `reuse_detect/precommit_response.py` with `PrecommitResponse` class
    - Implement `respond()` returning non-zero exit code on BLOCK, 0 on WARN/PASS
    - Format human-readable stderr output with import paths and usage examples for WARN/BLOCK
    - Exit silently on PASS
    - _Requirements: 6.1, 6.2, 6.3, 6.4_

  - [ ]* 8.2 Write property test for exit code correctness
    - **Property 13: Exit Code Correctness**
    - **Validates: Requirements 6.1, 6.2, 6.3**

  - [ ]* 8.3 Write property test for stderr output contains suggestions
    - **Property 21: Stderr Output Contains Suggestions**
    - **Validates: Requirement 6.4**

- [x] 9. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 10. Implement UnifiedIndexer (Offline)
  - [x] 10.1 Implement GitHub repository indexing
    - Create `reuse_detect/indexer.py` with `UnifiedIndexer` class
    - Implement `reindex_github()` that fetches configured repos, builds BlockFingerprints for all functions/classes, and upserts rows tagged GITHUB into the vector store
    - Implement refresh interval check (`index_refresh_interval_hours`)
    - _Requirements: 7.1, 7.4_

  - [x] 10.2 Implement local codebase indexing
    - Implement `reindex_local()` that walks the local repo, builds BlockFingerprints, and upserts rows tagged LOCAL
    - Implement `reindex_incremental_local()` for changed-files-only re-indexing
    - _Requirements: 7.2, 7.3_

  - [x] 10.3 Implement error handling and resilience for indexing
    - Handle GitHub unavailability with cached index fallback
    - Handle missing cache with GitHub-skip fallback
    - Handle index corruption with delete-and-rebuild
    - Ensure UnifiedIndexer is the sole writer (no concurrent writes)
    - _Requirements: 7.5, 10.1, 10.2, 10.5_

  - [ ]* 10.4 Write unit tests for indexer error handling
    - Test fallback behavior when GitHub is unavailable
    - Test incremental re-indexing correctness
    - Test corruption recovery
    - _Requirements: 10.1, 10.2, 10.5_

- [x] 11. Implement pipeline orchestration and error handling
  - [x] 11.1 Implement main pipeline orchestrator
    - Create `reuse_detect/pipeline.py` with `detect_reusable_code()` function
    - Wire stages 2-7 together: extract → build → retrieve → score → decide → respond
    - Implement per-block loop with LLM call cap tracking across all blocks
    - Handle empty diff with PASS decision
    - _Requirements: 9.1, 9.2, 9.4, 10.6_

  - [x] 11.2 Implement error handling for unparseable blocks
    - Skip unparseable blocks with logging (file path, line numbers)
    - Continue processing remaining blocks
    - _Requirements: 10.3_

  - [ ]* 11.3 Write property test for pipeline completeness
    - **Property 9: Pipeline Completeness**
    - **Validates: Requirement 9.1**

  - [ ]* 11.4 Write property test for total output bound
    - **Property 11: Total Output Bound**
    - **Validates: Requirement 9.2**

  - [ ]* 11.5 Write property test for unparseable block resilience
    - **Property 20: Unparseable Block Resilience**
    - **Validates: Requirement 10.3**

- [x] 12. Implement CLI entry point and pre-commit hook integration
  - [x] 12.1 Implement CLI entry point
    - Create `reuse_detect/cli.py` with `reuse-detect` command using argparse or click
    - Support `--staged` flag for pre-commit mode
    - Support `--reindex` flag for manual re-indexing
    - Load configuration and invoke the pipeline orchestrator
    - _Requirements: 11.1, 11.4_

  - [x] 12.2 Implement pre-commit hook configuration
    - Create `.pre-commit-hooks.yaml` for pre-commit framework integration
    - Create `setup.py` or `pyproject.toml` entry point for `reuse-detect` command
    - Document installation in README
    - _Requirements: 6.1, 6.2, 6.3_

  - [ ]* 12.3 Write integration tests for end-to-end pipeline
    - Test full pipeline with a mock git repository containing known duplicates
    - Test PASS, WARN, and BLOCK scenarios end-to-end
    - Verify exit codes and stderr output
    - _Requirements: 6.1, 6.2, 6.3, 6.4_

- [x] 13. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties from the design document
- Unit tests validate specific examples and edge cases
- The implementation language is Python as specified in the design
- GitHub Models API (`models.github.ai`) is used for both embeddings and LLM chat completions
- FAISS is the primary vector store backend; ChromaDB is an alternative
