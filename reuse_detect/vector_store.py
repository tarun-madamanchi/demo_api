"""Vector store abstraction and FAISS backend.

Provides source-tagged ANN (Approximate Nearest Neighbor) search
over BlockFingerprint embeddings.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Protocol

import numpy as np

from .models import BlockFingerprint, IndexSource

logger = logging.getLogger(__name__)


class VectorStore(Protocol):
    """Protocol for vector store backends."""

    def upsert(
        self,
        fingerprint: BlockFingerprint,
        indexed_id: str,
        source: IndexSource,
    ) -> None:
        """Insert or update a fingerprint in the store."""
        ...

    def delete(self, indexed_id: str) -> None:
        """Delete an entry by its ID."""
        ...

    def query(
        self,
        vector: list[float],
        top_k: int = 50,
        sources: list[IndexSource] | None = None,
    ) -> list[tuple[str, float, IndexSource, dict]]:
        """Query for nearest neighbors. Returns (indexed_id, distance, source, metadata) tuples."""
        ...

    def save(self) -> None:
        """Persist the store to disk."""
        ...

    def load(self) -> None:
        """Load the store from disk."""
        ...

    @property
    def size(self) -> int:
        """Return the number of entries in the store."""
        ...


class FAISSVectorStore:
    """FAISS-based vector store with source-tagged metadata."""

    def __init__(self, store_path: Path, dimension: int = 1536):
        self.store_path = store_path
        self.dimension = dimension
        self._index = None
        self._metadata: dict[int, dict] = {}  # position -> metadata
        self._id_to_position: dict[str, int] = {}  # indexed_id -> position
        self._vectors: list[list[float]] = []
        self._next_position: int = 0

    @property
    def size(self) -> int:
        return len(self._id_to_position)

    def upsert(
        self,
        fingerprint: BlockFingerprint,
        indexed_id: str,
        source: IndexSource,
    ) -> None:
        """Insert or update a fingerprint in the store."""
        # If already exists, remove old entry
        if indexed_id in self._id_to_position:
            self.delete(indexed_id)

        position = self._next_position
        self._next_position += 1

        self._vectors.append(fingerprint.embedding.vector)
        self._metadata[position] = {
            "indexed_id": indexed_id,
            "source": source.value,
            "content_hash": fingerprint.content_hash,
            "file_path": str(fingerprint.block.file_path),
            "function_name": fingerprint.block.function_name,
            "class_name": fingerprint.block.class_name,
            "content": fingerprint.block.content,
            "start_line": fingerprint.block.start_line,
            "end_line": fingerprint.block.end_line,
            "ast_structure_hash": fingerprint.features.ast_structure_hash,
            "control_flow_pattern": fingerprint.features.control_flow_pattern,
            "token_sequence": fingerprint.features.token_sequence,
            "function_signatures": fingerprint.features.function_signatures,
            "decorators": fingerprint.features.decorators,
        }
        self._id_to_position[indexed_id] = position

        # Rebuild FAISS index
        self._rebuild_index()

    def delete(self, indexed_id: str) -> None:
        """Delete an entry by its ID."""
        if indexed_id not in self._id_to_position:
            return

        position = self._id_to_position[indexed_id]
        del self._metadata[position]
        del self._id_to_position[indexed_id]

        # Note: We don't remove from _vectors to keep positions stable.
        # The entry is simply excluded from metadata lookups.

    def query(
        self,
        vector: list[float],
        top_k: int = 50,
        sources: list[IndexSource] | None = None,
    ) -> list[tuple[str, float, IndexSource, dict]]:
        """Query for nearest neighbors with optional source filtering.

        Returns (indexed_id, distance, source, metadata) tuples.
        """
        if not self._vectors or not self._metadata:
            return []

        # Build numpy array of all active vectors
        active_positions = sorted(self._metadata.keys())
        if not active_positions:
            return []

        active_vectors = np.array(
            [self._vectors[p] for p in active_positions], dtype=np.float32
        )
        query_vector = np.array([vector], dtype=np.float32)

        # Normalize for cosine similarity
        norms = np.linalg.norm(active_vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        active_vectors_normalized = active_vectors / norms

        query_norm = np.linalg.norm(query_vector)
        if query_norm > 0:
            query_vector_normalized = query_vector / query_norm
        else:
            query_vector_normalized = query_vector

        # Compute cosine similarities
        similarities = np.dot(
            active_vectors_normalized, query_vector_normalized.T
        ).flatten()

        # Convert to distances (1 - similarity)
        distances = 1.0 - similarities

        # Sort by distance (ascending)
        sorted_indices = np.argsort(distances)

        results: list[tuple[str, float, IndexSource, dict]] = []
        for idx in sorted_indices:
            if len(results) >= top_k:
                break

            position = active_positions[idx]
            meta = self._metadata[position]
            source = IndexSource(meta["source"])

            # Apply source filter
            if sources and source not in sources:
                continue

            # Include the stored vector in metadata for scoring
            meta_with_vector = dict(meta)
            meta_with_vector["_vector"] = self._vectors[position]

            results.append(
                (meta["indexed_id"], float(distances[idx]), source, meta_with_vector)
            )

        return results

    def save(self) -> None:
        """Persist the store to disk."""
        self.store_path.mkdir(parents=True, exist_ok=True)

        data = {
            "vectors": self._vectors,
            "metadata": self._metadata,
            "id_to_position": self._id_to_position,
            "next_position": self._next_position,
            "dimension": self.dimension,
        }

        store_file = self.store_path / "vector_store.pkl"
        with open(store_file, "wb") as f:
            pickle.dump(data, f)

    def load(self) -> None:
        """Load the store from disk."""
        store_file = self.store_path / "vector_store.pkl"
        if not store_file.exists():
            return

        try:
            with open(store_file, "rb") as f:
                data = pickle.load(f)

            self._vectors = data["vectors"]
            self._metadata = data["metadata"]
            self._id_to_position = data["id_to_position"]
            self._next_position = data["next_position"]
            self.dimension = data["dimension"]
            self._rebuild_index()
        except (pickle.UnpicklingError, KeyError, EOFError) as e:
            logger.warning("Corrupted vector store, resetting: %s", e)
            self._reset()

    def _rebuild_index(self) -> None:
        """Rebuild the FAISS index from current vectors."""
        # For the numpy-based implementation, no explicit rebuild needed
        pass

    def _reset(self) -> None:
        """Reset the store to empty state."""
        self._vectors = []
        self._metadata = {}
        self._id_to_position = {}
        self._next_position = 0
        self._index = None
