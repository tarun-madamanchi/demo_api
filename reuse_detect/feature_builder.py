"""Stage 3: Feature and Embedding Builder.

Single-pass builder: one AST walk + one embedding call per block.
Produces structural features and a dense semantic embedding vector.
"""

import ast
import hashlib
import logging
from typing import Protocol

from .models import BlockFingerprint, CodeBlock, CodeEmbedding, CodeFeatures

logger = logging.getLogger(__name__)


class EmbeddingProvider(Protocol):
    """Protocol for embedding API providers."""

    def embed(self, text: str) -> list[float]:
        """Generate embedding vector for a single text."""
        ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embedding vectors for multiple texts."""
        ...

    @property
    def model_id(self) -> str:
        """Return the model identifier."""
        ...

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        ...


class GitHubModelsEmbeddingProvider:
    """Embedding provider using GitHub Models API."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_token: str | None = None,
    ):
        self._model = model
        self._api_token = api_token
        self._dimension = 1536  # Default for text-embedding-3-small

    @property
    def model_id(self) -> str:
        return self._model

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> list[float]:
        """Generate embedding via GitHub Models API."""
        import httpx

        response = httpx.post(
            "https://models.github.ai/inference/embeddings",
            headers={
                "Authorization": f"Bearer {self._api_token}",
                "Content-Type": "application/json",
            },
            json={"input": text, "model": self._model},
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json()
        return data["data"][0]["embedding"]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts in one API call."""
        import httpx

        response = httpx.post(
            "https://models.github.ai/inference/embeddings",
            headers={
                "Authorization": f"Bearer {self._api_token}",
                "Content-Type": "application/json",
            },
            json={"input": texts, "model": self._model},
            timeout=60.0,
        )
        response.raise_for_status()
        data = response.json()
        # Sort by index to maintain order
        sorted_data = sorted(data["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in sorted_data]


class FeatureEmbeddingBuilder:
    """Single-pass builder: one AST walk + one embedding call per block."""

    def __init__(self, embedding_provider: EmbeddingProvider):
        self.embedding_provider = embedding_provider

    def build(self, block: CodeBlock) -> BlockFingerprint:
        """Normalize, parse once, extract features, embed, return fingerprint."""
        # Extract structural features via AST
        features = self._extract_features(block.content)

        # Generate semantic embedding
        embedding = self._generate_embedding(block.content)

        # Compute content hash
        content_hash = hashlib.sha256(block.content.encode()).hexdigest()

        return BlockFingerprint(
            block=block,
            features=features,
            embedding=embedding,
            content_hash=content_hash,
        )

    def build_batch(self, blocks: list[CodeBlock]) -> list[BlockFingerprint]:
        """Batched variant for the offline indexer path.

        Uses batched embedding calls to reduce API round-trips.
        """
        if not blocks:
            return []

        # Extract features for all blocks (CPU-bound, no batching needed)
        features_list = [self._extract_features(b.content) for b in blocks]

        # Batch embedding call
        texts = [b.content for b in blocks]
        vectors = self.embedding_provider.embed_batch(texts)

        fingerprints: list[BlockFingerprint] = []
        for block, features, vector in zip(blocks, features_list, vectors):
            embedding = CodeEmbedding(
                vector=vector,
                dim=len(vector),
                model_id=self.embedding_provider.model_id,
            )
            content_hash = hashlib.sha256(block.content.encode()).hexdigest()
            fingerprints.append(
                BlockFingerprint(
                    block=block,
                    features=features,
                    embedding=embedding,
                    content_hash=content_hash,
                )
            )

        return fingerprints

    def _extract_features(self, source: str) -> CodeFeatures:
        """Perform a single AST walk to extract all structural features."""
        features = CodeFeatures()

        try:
            tree = ast.parse(source)
        except SyntaxError:
            # Return empty features for unparseable code
            features.ast_structure_hash = hashlib.md5(
                source.encode()
            ).hexdigest()
            return features

        # Walk the AST once, collecting all features
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                features.function_signatures.append(
                    self._get_function_signature(node)
                )
                if node.decorator_list:
                    for dec in node.decorator_list:
                        features.decorators.append(ast.unparse(dec))
                if node.returns:
                    features.return_type = ast.unparse(node.returns)
                # Extract parameter types
                for arg in node.args.args:
                    if arg.annotation:
                        features.parameter_types.append(
                            ast.unparse(arg.annotation)
                        )

            elif isinstance(node, ast.ClassDef):
                bases = [ast.unparse(b) for b in node.bases]
                features.class_hierarchy.append(
                    f"{node.name}({', '.join(bases)})"
                )
                if node.decorator_list:
                    for dec in node.decorator_list:
                        features.decorators.append(ast.unparse(dec))

            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                features.import_patterns.append(ast.unparse(node))

        # Extract control flow pattern
        features.control_flow_pattern = self._extract_control_flow(tree)

        # Extract docstring
        features.docstring = ast.get_docstring(tree)

        # Compute AST structure hash (rename-invariant, comment-invariant)
        features.ast_structure_hash = self._compute_structure_hash(tree)

        # Extract token sequence (normalized identifiers)
        features.token_sequence = self._extract_token_sequence(tree)

        return features

    def _get_function_signature(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> str:
        """Extract function signature as a string."""
        args = []
        for arg in node.args.args:
            if arg.annotation:
                args.append(f"{arg.arg}: {ast.unparse(arg.annotation)}")
            else:
                args.append(arg.arg)

        ret = f" -> {ast.unparse(node.returns)}" if node.returns else ""
        prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
        return f"{prefix}def {node.name}({', '.join(args)}){ret}"

    def _extract_control_flow(self, tree: ast.AST) -> str:
        """Extract a string representing the control flow structure."""
        patterns: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                patterns.append("IF")
            elif isinstance(node, ast.For):
                patterns.append("FOR")
            elif isinstance(node, ast.While):
                patterns.append("WHILE")
            elif isinstance(node, ast.Try):
                patterns.append("TRY")
            elif isinstance(node, ast.With):
                patterns.append("WITH")
            elif isinstance(node, ast.Return):
                patterns.append("RET")
            elif isinstance(node, ast.Yield):
                patterns.append("YIELD")
            elif isinstance(node, ast.Raise):
                patterns.append("RAISE")
        return "|".join(patterns)

    def _compute_structure_hash(self, tree: ast.AST) -> str:
        """Compute a hash of the AST structure, invariant to renames and comments.

        This hash captures the shape of the code (node types, nesting)
        but not the specific identifiers or literal values.
        """
        structure = self._ast_to_structure_string(tree)
        return hashlib.md5(structure.encode()).hexdigest()

    def _ast_to_structure_string(self, node: ast.AST) -> str:
        """Convert AST to a structure string ignoring names and values."""
        parts: list[str] = []
        parts.append(type(node).__name__)

        for child in ast.iter_child_nodes(node):
            # Skip nodes that carry only naming/value info
            if isinstance(child, (ast.Constant, ast.Name, ast.alias)):
                parts.append(type(child).__name__)
            else:
                parts.append(self._ast_to_structure_string(child))

        return f"({' '.join(parts)})"

    def _extract_token_sequence(self, tree: ast.AST) -> list[str]:
        """Extract normalized token sequence from AST (type-based, not name-based)."""
        tokens: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                tokens.append("NAME")
            elif isinstance(node, ast.Constant):
                tokens.append(f"CONST:{type(node.value).__name__}")
            elif isinstance(node, ast.Call):
                tokens.append("CALL")
            elif isinstance(node, ast.Attribute):
                tokens.append("ATTR")
            elif isinstance(node, ast.BinOp):
                tokens.append(f"BINOP:{type(node.op).__name__}")
            elif isinstance(node, ast.Compare):
                tokens.append("CMP")
            elif isinstance(node, ast.BoolOp):
                tokens.append(f"BOOLOP:{type(node.op).__name__}")
            elif isinstance(node, ast.UnaryOp):
                tokens.append(f"UNOP:{type(node.op).__name__}")
        return tokens

    def _generate_embedding(self, source: str) -> CodeEmbedding:
        """Generate a dense semantic embedding for the source code."""
        vector = self.embedding_provider.embed(source)
        return CodeEmbedding(
            vector=vector,
            dim=len(vector),
            model_id=self.embedding_provider.model_id,
        )
