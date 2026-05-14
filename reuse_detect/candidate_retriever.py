"""Stage 4: Candidate Retriever.

Issues a single ANN query against the unified source-tagged vector store
and returns candidates from both LOCAL and GITHUB sources.
"""

import logging
from pathlib import Path

from .models import (
    BlockFingerprint,
    CodeBlock,
    CodeEmbedding,
    CodeFeatures,
    ChangeType,
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
        returns source-tagged candidates.
        """
        if sources is None:
            sources = [IndexSource.LOCAL_CODEBASE, IndexSource.GITHUB_LIBRARY]

        results = self.store.query(
            vector=fp.embedding.vector,
            top_k=top_k,
            sources=sources,
        )

        candidates: list[SourceTaggedCandidate] = []
        for indexed_id, distance, source in results:
            # Create a minimal fingerprint for the candidate
            # In a full implementation, we'd retrieve the full fingerprint from the store
            candidate_fp = BlockFingerprint(
                block=CodeBlock(
                    file_path=Path("unknown"),
                    start_line=0,
                    end_line=0,
                    content="",
                    change_type=ChangeType.MODIFIED,
                ),
                features=CodeFeatures(),
                embedding=CodeEmbedding(),
                content_hash="",
            )

            candidates.append(
                SourceTaggedCandidate(
                    source=source,
                    indexed_id=indexed_id,
                    distance=distance,
                    fingerprint=candidate_fp,
                )
            )

        return candidates
