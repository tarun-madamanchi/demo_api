"""Stage 4: Candidate Retriever.

Issues a single ANN query against the unified source-tagged vector store
and returns candidates from both LOCAL and GITHUB sources.
"""

import logging
from pathlib import Path

from .models import (
    BlockFingerprint,
    ChangeType,
    CodeBlock,
    CodeEmbedding,
    CodeFeatures,
    IndexSource,
    SourceTaggedCandidate,
)
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


class CandidateRetriever:
    """Single ANN query against the unified source-tagged vector store."""

    def __init__(self, store: VectorStore):
        self.store = store

    def query(
        self,
        fp: BlockFingerprint,
        sources: list[IndexSource] | None = None,
        top_k: int = 50,
    ) -> list[SourceTaggedCandidate]:
        """Return top_k candidates from ALL requested sources in one call.

        Issues a single ANN query against the unified vector store and
        returns source-tagged candidates with full fingerprint data
        reconstructed from stored metadata.
        """
        if sources is None:
            sources = [IndexSource.LOCAL_CODEBASE, IndexSource.GITHUB_LIBRARY]

        results = self.store.query(
            vector=fp.embedding.vector,
            top_k=top_k,
            sources=sources,
        )

        if not results:
            logger.debug("No candidates found in vector store (store may be empty)")
            return []

        candidates: list[SourceTaggedCandidate] = []
        for indexed_id, distance, source, metadata in results:
            # Skip self-matches (same content hash as the query)
            if metadata.get("content_hash") == fp.content_hash:
                continue

            # Reconstruct a full fingerprint from stored metadata
            candidate_fp = self._reconstruct_fingerprint(metadata)

            candidates.append(
                SourceTaggedCandidate(
                    source=source,
                    indexed_id=indexed_id,
                    distance=distance,
                    fingerprint=candidate_fp,
                )
            )

        logger.info(
            "Retrieved %d candidates (from %d raw results)",
            len(candidates),
            len(results),
        )
        return candidates

    def _reconstruct_fingerprint(self, metadata: dict) -> BlockFingerprint:
        """Reconstruct a BlockFingerprint from stored vector store metadata.

        This rebuilds the full fingerprint with actual content, features,
        and embedding data so the scorer can perform meaningful comparisons.
        """
        # Reconstruct the CodeBlock with real data
        block = CodeBlock(
            file_path=Path(metadata.get("file_path", "unknown")),
            start_line=metadata.get("start_line", 0),
            end_line=metadata.get("end_line", 0),
            content=metadata.get("content", ""),
            change_type=ChangeType.MODIFIED,
            function_name=metadata.get("function_name"),
            class_name=metadata.get("class_name"),
        )

        # Reconstruct CodeFeatures from stored structural data
        features = CodeFeatures(
            ast_structure_hash=metadata.get("ast_structure_hash", ""),
            control_flow_pattern=metadata.get("control_flow_pattern", ""),
            token_sequence=metadata.get("token_sequence", []),
            function_signatures=metadata.get("function_signatures", []),
            decorators=metadata.get("decorators", []),
        )

        # Reconstruct embedding — use the stored vector if available
        stored_vector = metadata.get("_vector", [])
        embedding = CodeEmbedding(
            vector=stored_vector,
            dim=len(stored_vector),
            model_id="stored",
        )

        return BlockFingerprint(
            block=block,
            features=features,
            embedding=embedding,
            content_hash=metadata.get("content_hash", ""),
        )
