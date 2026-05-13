# Requirements Document

## Introduction

This document specifies the requirements for the Common Library Reuse Detection feature. The system is an agent-driven pipeline that detects duplicate or reusable code before a developer commits changes. It integrates as a git pre-commit hook and CI/PR review step, combining AST structural analysis, semantic embeddings, vector search, and LLM validation to identify code that duplicates logic already available in common libraries or the local microservice codebase.

## Glossary

- **Pipeline**: The end-to-end detection workflow comprising 7 online stages and 1 offline indexer
- **Change_Extractor**: The component (Stage 2) that parses the git diff, expands hunks to whole functions/classes, and normalizes source code
- **Feature_Embedding_Builder**: The component (Stage 3) that performs a single AST walk and a single embedding call per code block to produce structural features and a dense vector
- **Candidate_Retriever**: The component (Stage 4) that issues a single ANN query against the unified vector store and returns source-tagged candidates
- **Hybrid_Scorer**: The component (Stage 5) that computes structural similarity, embedding cosine, and LLM logic-equivalence for each candidate
- **Decision_Renderer**: The component (Stage 6) that applies deterministic thresholds to produce pass/warn/block decisions and renders reuse suggestions
- **Pre_Commit_Hook**: The component (Stage 7) that writes output to stderr and controls the process exit code
- **Unified_Indexer**: The offline component that populates the vector store with fingerprints from GitHub common libraries and the local microservice codebase
- **CodeBlock**: A discrete unit of changed code (function or class) extracted from the staged diff
- **BlockFingerprint**: The combined AST features and dense embedding vector for a single CodeBlock
- **ValidatedMatch**: A candidate that has been scored structurally, semantically, and validated by the LLM
- **Decision**: The outcome of the pipeline for a commit: pass, warn, or block
- **ReuseSuggestion**: A rendered suggestion containing import path, usage example, and explanation
- **Vector_Store**: The local FAISS or ChromaDB index holding source-tagged BlockFingerprint rows
- **IndexSource**: A tag indicating whether a vector store row originates from a GitHub common library or the local codebase

## Requirements

### Requirement 1: Change Extraction

**User Story:** As a developer, I want the system to automatically identify meaningful code changes from my staged diff, so that only relevant code blocks are analyzed for reuse.

#### Acceptance Criteria

1. WHEN a pre-commit hook is triggered, THE Change_Extractor SHALL parse the staged git diff and produce a list of CodeBlock objects representing changed functions and classes
2. WHEN a CodeBlock contains fewer lines than the configured min_block_lines threshold, THE Change_Extractor SHALL classify the block as trivial and exclude it from further analysis
3. WHEN a CodeBlock consists entirely of comments or whitespace, THE Change_Extractor SHALL classify the block as trivial and exclude it from further analysis
4. WHEN extracting a partial function or class from a diff hunk, THE Change_Extractor SHALL expand the extraction to include the complete function or class definition
5. WHEN producing CodeBlock objects, THE Change_Extractor SHALL normalize the source code to remove formatting variations before downstream processing

### Requirement 2: Feature and Embedding Construction

**User Story:** As a developer, I want the system to build a rich fingerprint of each code block combining structural and semantic signals, so that matches can be found even when code is written differently.

#### Acceptance Criteria

1. WHEN a CodeBlock is provided, THE Feature_Embedding_Builder SHALL perform exactly one AST parse and one embedding API call to produce a BlockFingerprint
2. WHEN building a BlockFingerprint, THE Feature_Embedding_Builder SHALL extract structural features including function signatures, class hierarchy, import patterns, decorators, control flow patterns, parameter types, return types, and AST structure hash
3. WHEN building a BlockFingerprint, THE Feature_Embedding_Builder SHALL produce a dense semantic embedding vector via the GitHub Models API
4. WHEN processing multiple CodeBlocks for the offline indexer, THE Feature_Embedding_Builder SHALL support batched embedding calls to reduce API round-trips

### Requirement 3: Candidate Retrieval

**User Story:** As a developer, I want the system to efficiently find potential matches from both common libraries and the local codebase in a single query, so that detection is fast and comprehensive.

#### Acceptance Criteria

1. WHEN a BlockFingerprint is provided, THE Candidate_Retriever SHALL issue a single ANN query against the unified Vector_Store and return up to top_k SourceTaggedCandidate results
2. WHEN querying the Vector_Store, THE Candidate_Retriever SHALL return candidates from both LOCAL and GITHUB sources in a single result set without requiring separate queries
3. WHEN returning candidates, THE Candidate_Retriever SHALL tag each candidate with its IndexSource so downstream stages can distinguish between local and library matches

### Requirement 4: Hybrid Scoring and LLM Validation

**User Story:** As a developer, I want the system to combine structural, semantic, and behavioral analysis to validate matches, so that suggestions are accurate and false positives are minimized.

#### Acceptance Criteria

1. WHEN scoring a candidate, THE Hybrid_Scorer SHALL compute a structural similarity score based on AST features
2. WHEN scoring a candidate, THE Hybrid_Scorer SHALL compute a semantic similarity score based on embedding cosine distance
3. WHEN scoring a candidate, THE Hybrid_Scorer SHALL invoke the LLM to determine logic-equivalence between the staged code and the candidate
4. WHEN all scoring signals are available, THE Hybrid_Scorer SHALL produce a combined_score that integrates structural, semantic, and LLM confidence values into a single ValidatedMatch
5. WHEN the combined_score for any ValidatedMatch is computed, THE Hybrid_Scorer SHALL ensure the score falls within the range [0.0, 1.0]

### Requirement 5: Decision and Suggestion Rendering

**User Story:** As a developer, I want the system to make a clear pass/warn/block decision and provide actionable reuse suggestions, so that I know exactly what to do when duplication is detected.

#### Acceptance Criteria

1. WHEN the highest combined_score among all ValidatedMatch results meets or exceeds the block_threshold, THE Decision_Renderer SHALL produce a BLOCK decision
2. WHEN the highest combined_score is below block_threshold but meets or exceeds the warn_threshold, THE Decision_Renderer SHALL produce a WARN decision
3. WHEN the highest combined_score is below the warn_threshold, THE Decision_Renderer SHALL produce a PASS decision
4. WHEN rendering a ReuseSuggestion, THE Decision_Renderer SHALL include the original code location, existing code location, a valid import statement, a usage example, confidence score, and explanation
5. WHEN producing suggestions for a single CodeBlock, THE Decision_Renderer SHALL limit the number of suggestions to max_suggestions_per_block from the configuration

### Requirement 6: Pre-commit Hook Response

**User Story:** As a developer, I want the pre-commit hook to block my commit when significant duplication is found and let it through otherwise, so that duplicated code never enters the repository undetected.

#### Acceptance Criteria

1. WHEN the Decision is BLOCK, THE Pre_Commit_Hook SHALL exit with a non-zero exit code so that git aborts the commit
2. WHEN the Decision is WARN, THE Pre_Commit_Hook SHALL write suggestions to stderr and exit with code 0 so that the commit proceeds
3. WHEN the Decision is PASS, THE Pre_Commit_Hook SHALL exit with code 0 without writing suggestions
4. WHEN writing suggestions to stderr, THE Pre_Commit_Hook SHALL format the output as a human-readable message listing the top reuse suggestions with import paths and usage examples

### Requirement 7: Unified Indexing

**User Story:** As a developer, I want the system to maintain an up-to-date index of both common libraries and local code, so that detection always reflects the current state of available reusable code.

#### Acceptance Criteria

1. WHEN indexing GitHub repositories, THE Unified_Indexer SHALL fetch configured repositories, build BlockFingerprints for all functions and classes, and upsert rows tagged with source GITHUB into the Vector_Store
2. WHEN indexing the local microservice codebase, THE Unified_Indexer SHALL walk the repository, build BlockFingerprints for all functions and classes, and upsert rows tagged with source LOCAL into the Vector_Store
3. WHEN local files change, THE Unified_Indexer SHALL support incremental re-indexing of only the changed files rather than a full rebuild
4. WHEN the index_refresh_interval_hours has elapsed since the last GitHub index update, THE Unified_Indexer SHALL trigger a refresh of the GitHub library index
5. THE Unified_Indexer SHALL be the sole writer to the Vector_Store to prevent concurrent write conflicts

### Requirement 8: Scoring Correctness

**User Story:** As a developer, I want the similarity scoring to be mathematically sound and deterministic, so that I can trust the results are consistent and fair.

#### Acceptance Criteria

1. THE Hybrid_Scorer SHALL produce identical results for the same codebase state and configuration across multiple runs (determinism)
2. WHEN a CodeBlock is compared against itself, THE Hybrid_Scorer SHALL produce a similarity score of 1.0 (self-match)
3. WHEN comparing CodeBlock A against CodeBlock B, THE Hybrid_Scorer SHALL produce the same similarity score as when comparing B against A (symmetry)
4. WHEN all variables in a CodeBlock are renamed consistently, THE Hybrid_Scorer SHALL produce the same structural similarity score as the original (rename invariance)
5. WHEN comments are added to or removed from a CodeBlock, THE Hybrid_Scorer SHALL produce the same structural similarity score as the original (comment invariance)

### Requirement 9: Pipeline Completeness and Bounds

**User Story:** As a developer, I want assurance that every meaningful change is analyzed and that output is bounded, so that nothing is silently skipped and output remains manageable.

#### Acceptance Criteria

1. THE Pipeline SHALL analyze every changed CodeBlock that has lines greater than or equal to min_block_lines (completeness)
2. THE Pipeline SHALL produce a total number of suggestions less than or equal to the number of analyzed blocks multiplied by max_suggestions_per_block (bounded output)
3. WHEN querying the Vector_Store for a non-trivial CodeBlock, THE Candidate_Retriever SHALL query against both LOCAL and GITHUB sources (source coverage)
4. WHEN the number of LLM calls would exceed max_llm_calls_per_commit, THE Pipeline SHALL stop issuing LLM calls and use only structural and semantic scores for remaining candidates

### Requirement 10: Error Handling and Resilience

**User Story:** As a developer, I want the system to handle failures gracefully without blocking my workflow unnecessarily, so that infrastructure issues do not prevent me from committing valid code.

#### Acceptance Criteria

1. IF a GitHub repository is unavailable during indexing, THEN THE Unified_Indexer SHALL fall back to the cached index and log a warning
2. IF no cached index exists and GitHub is unavailable, THEN THE Unified_Indexer SHALL skip the GitHub source and continue with local codebase only
3. IF a CodeBlock cannot be parsed by the AST parser, THEN THE Change_Extractor SHALL skip the block, log the parse error with file path and line numbers, and continue processing remaining blocks
4. IF the LLM service times out or is unavailable, THEN THE Hybrid_Scorer SHALL fall back to structural and semantic scores only and log a warning
5. IF the Vector_Store index is corrupted, THEN THE Unified_Indexer SHALL delete the corrupted cache and rebuild from scratch
6. IF the staged diff is empty, THEN THE Pipeline SHALL return a PASS decision with an informative message

### Requirement 11: Configuration

**User Story:** As a developer, I want to configure detection thresholds, file patterns, and resource limits, so that the system adapts to my project's needs.

#### Acceptance Criteria

1. THE Pipeline SHALL read configuration values for block_threshold, warn_threshold, min_block_lines, top_k, max_suggestions_per_block, max_llm_calls_per_commit, and llm_timeout_seconds from a DetectionConfig object
2. WHEN exclude_patterns are configured, THE Change_Extractor SHALL skip files matching those patterns during extraction
3. WHEN include_patterns are configured, THE Change_Extractor SHALL only process files matching those patterns
4. THE Pipeline SHALL use default configuration values when no explicit configuration is provided
