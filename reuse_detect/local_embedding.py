"""Local embedding provider that works without external API tokens.

Uses TF-IDF style token hashing to produce a dense vector from code,
enabling similarity search without any network calls.
"""

import hashlib
import math
import re
from collections import Counter


class LocalEmbeddingProvider:
    """Embedding provider that works entirely offline using token hashing.

    Produces deterministic embeddings based on normalized code tokens.
    No API keys or network access required.
    """

    def __init__(self, dimension: int = 256):
        self._dimension = dimension
        self._model_id = "local-token-hash"

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> list[float]:
        """Generate a deterministic embedding vector from code text.

        Uses token frequency hashing projected into a fixed-dimension space.
        """
        tokens = self._tokenize(text)
        return self._tokens_to_vector(tokens)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        return [self.embed(t) for t in texts]

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize source code into meaningful tokens.

        Strips comments, normalizes identifiers to patterns,
        and extracts structural tokens.
        """
        # Remove comments
        text = re.sub(r"#.*$", "", text, flags=re.MULTILINE)
        # Remove string literals (replace with placeholder)
        text = re.sub(r'""".*?"""', "STR", text, flags=re.DOTALL)
        text = re.sub(r"'''.*?'''", "STR", text, flags=re.DOTALL)
        text = re.sub(r'"[^"]*"', "STR", text)
        text = re.sub(r"'[^']*'", "STR", text)

        # Extract tokens: keywords, operators, identifiers (normalized)
        tokens: list[str] = []

        # Python keywords and builtins to keep as-is
        keywords = {
            "def", "class", "return", "if", "else", "elif", "for", "while",
            "try", "except", "finally", "with", "as", "import", "from",
            "raise", "yield", "async", "await", "lambda", "pass", "break",
            "continue", "and", "or", "not", "in", "is", "None", "True",
            "False", "self", "cls",
        }

        # Split into words and operators
        parts = re.findall(r"[a-zA-Z_]\w*|[+\-*/=<>!&|^~%]+|[(){}\[\],.:;@]", text)

        for part in parts:
            if part in keywords:
                tokens.append(f"KW:{part}")
            elif re.match(r"^[A-Z][A-Z_0-9]+$", part):
                # CONSTANT_CASE -> normalize
                tokens.append("CONST_NAME")
            elif re.match(r"^[A-Z]", part):
                # ClassName -> normalize
                tokens.append("CLASS_NAME")
            elif re.match(r"^__.*__$", part):
                # Dunder methods
                tokens.append(f"DUNDER:{part}")
            elif re.match(r"^_", part):
                # Private name
                tokens.append("PRIVATE_NAME")
            elif re.match(r"^[a-z]", part):
                # Regular identifier -> normalize
                tokens.append("IDENT")
            else:
                # Operators and punctuation
                tokens.append(f"OP:{part}")

        return tokens

    def _tokens_to_vector(self, tokens: list[str]) -> list[float]:
        """Convert token list to a fixed-dimension vector using feature hashing."""
        vector = [0.0] * self._dimension

        if not tokens:
            return vector

        # Count token frequencies
        counts = Counter(tokens)
        total = len(tokens)

        # Hash each token type into the vector dimensions
        for token, count in counts.items():
            # Use multiple hash functions for better distribution
            for seed in range(3):
                h = int(
                    hashlib.md5(f"{seed}:{token}".encode()).hexdigest(), 16
                )
                idx = h % self._dimension
                sign = 1.0 if (h // self._dimension) % 2 == 0 else -1.0
                # TF component (log-scaled)
                tf = math.log1p(count / total)
                vector[idx] += sign * tf

        # L2 normalize
        norm = math.sqrt(sum(x * x for x in vector))
        if norm > 0:
            vector = [x / norm for x in vector]

        return vector
