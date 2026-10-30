"""Embedding adapters.

The hashing embedder is deterministic and has no extra dependencies. It does
not encode meaning. Use it for tests, offline smoke checks, and the default
CLI path. Semantic retrieval needs the sentence-transformers or
OpenAI-compatible adapter.
"""

from __future__ import annotations

import hashlib
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from persian_rag.errors import ConfigError, PersianRagError
from persian_rag.remote import embed_texts

EmbedderKind = Literal["hashing", "sentence-transformers", "openai"]
_DEFAULT_ST_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_DEFAULT_OPENAI_MODEL = "text-embedding-3-small"
_DEFAULT_BASE_URL = "https://api.openai.com/v1"


class Embedder(Protocol):
    """Turns text into dense vectors."""

    @property
    def dimension(self) -> int:
        """Length of every vector this embedder returns."""

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed passages that will be stored."""

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query."""


@dataclass(frozen=True, slots=True)
class EmbedderSpec:
    """Serialized embedder settings stored next to an index.

    API keys are never part of the spec.
    """

    kind: EmbedderKind
    dimension: int
    model: str | None = None
    base_url: str | None = None
    ngram_min: int = 2
    ngram_max: int = 4
    seed: int = 13

    def __post_init__(self) -> None:
        if self.dimension < 1:
            raise ConfigError("dimension must be >= 1")
        if self.ngram_min < 1 or self.ngram_max < self.ngram_min:
            raise ConfigError("ngram range is invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "dimension": self.dimension,
            "model": self.model,
            "base_url": self.base_url,
            "ngram_min": self.ngram_min,
            "ngram_max": self.ngram_max,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> EmbedderSpec:
        kind = _embedder_kind(data.get("kind"))
        dimension = _positive_int(data.get("dimension"), "Embedder dimension")
        model = data.get("model")
        base_url = data.get("base_url")
        return cls(
            kind=kind,
            dimension=dimension,
            model=model if isinstance(model, str) else None,
            base_url=base_url if isinstance(base_url, str) else None,
            ngram_min=_positive_int(data.get("ngram_min", 2), "ngram_min"),
            ngram_max=_positive_int(data.get("ngram_max", 4), "ngram_max"),
            seed=_non_negative_int(data.get("seed", 13), "seed"),
        )


class HashingEmbedder:
    """Bag-of-character-n-gram hashing embedder.

    Vectors are L2-normalized and deterministic for a given dimension, seed,
    and n-gram range. Similar strings share buckets; the geometry is not
    semantic.
    """

    def __init__(
        self,
        dimension: int = 256,
        *,
        ngram_range: tuple[int, int] = (2, 4),
        seed: int = 13,
    ) -> None:
        if dimension < 1:
            raise ValueError("dimension must be >= 1")
        ngram_min, ngram_max = ngram_range
        if ngram_min < 1 or ngram_max < ngram_min:
            raise ValueError("ngram_range must satisfy 1 <= min <= max")
        self._dimension = dimension
        self._ngram_min = ngram_min
        self._ngram_max = ngram_max
        self._seed = seed

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def ngram_range(self) -> tuple[int, int]:
        return self._ngram_min, self._ngram_max

    @property
    def seed(self) -> int:
        return self._seed

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    def to_spec(self) -> EmbedderSpec:
        return EmbedderSpec(
            kind="hashing",
            dimension=self._dimension,
            ngram_min=self._ngram_min,
            ngram_max=self._ngram_max,
            seed=self._seed,
        )

    def _vector(self, text: str) -> list[float]:
        values = [0.0] * self._dimension
        if not text:
            return values
        padded = f" {text} "
        for width in range(self._ngram_min, self._ngram_max + 1):
            if len(padded) < width:
                continue
            for offset in range(len(padded) - width + 1):
                gram = padded[offset : offset + width]
                index, sign = self._bucket(gram)
                values[index] += sign
        return _l2_normalize(values)

    def _bucket(self, gram: str) -> tuple[int, float]:
        digest = hashlib.blake2s(
            f"{self._seed}:{gram}".encode(),
            digest_size=8,
        ).digest()
        value = int.from_bytes(digest, "little", signed=False)
        sign = 1.0 if value & 0x100 else -1.0
        return value % self._dimension, sign


class SentenceTransformerEmbedder:
    """Adapter for ``sentence-transformers`` multilingual models.

    Install the extra: ``pip install 'persian-rag[sentence-transformers]'``.
    The default model is multilingual and can embed Persian text. It is a
    default, not a claim about which model fits a particular corpus.
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_ST_MODEL,
        *,
        device: str | None = None,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Install it with: pip install 'persian-rag[sentence-transformers]'"
            ) from exc

        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self._model = SentenceTransformer(model_name, device=device)
        dimension = self._model.get_sentence_embedding_dimension()
        if not isinstance(dimension, int) or dimension < 1:
            raise PersianRagError(f"Could not read the embedding size of {model_name}")
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        encoded = self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [[float(item) for item in row] for row in encoded]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def to_spec(self) -> EmbedderSpec:
        return EmbedderSpec(
            kind="sentence-transformers",
            dimension=self._dimension,
            model=self.model_name,
        )


class OpenAICompatibleEmbedder:
    """Client for an OpenAI-compatible ``/v1/embeddings`` API.

    Works with the official API and with servers that implement the same
    request shape. Uses the Python standard library, so there is no extra
    dependency. The API key is read from ``api_key`` or ``OPENAI_API_KEY``.
    """

    def __init__(
        self,
        model: str = _DEFAULT_OPENAI_MODEL,
        *,
        api_key: str | None = None,
        base_url: str = _DEFAULT_BASE_URL,
        timeout: float = 60.0,
        dimensions: int | None = None,
        batch_size: int = 64,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        resolved_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY", "")
        if not resolved_key:
            raise ConfigError("OpenAI API key is missing. Set OPENAI_API_KEY or pass api_key.")
        self.model = model
        self.api_key = resolved_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.batch_size = batch_size
        self._dimensions = dimensions

    @property
    def dimension(self) -> int:
        if self._dimensions is None:
            raise ConfigError(
                "Embedding dimension is not known yet. Pass dimensions= or embed a text first."
            )
        return self._dimensions

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = embed_texts(
            list(texts),
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            dimensions=self._dimensions,
            batch_size=self.batch_size,
        )
        if vectors:
            width = len(vectors[0])
            if self._dimensions is None:
                self._dimensions = width
            elif width != self._dimensions:
                raise PersianRagError(
                    f"Endpoint returned dimension {width}, expected {self._dimensions}"
                )
        return vectors

    def embed_query(self, text: str) -> list[float]:
        vectors = self.embed_documents([text])
        if not vectors:
            raise PersianRagError("Embedding endpoint returned no vector")
        return vectors[0]

    def to_spec(self) -> EmbedderSpec:
        return EmbedderSpec(
            kind="openai",
            dimension=self.dimension,
            model=self.model,
            base_url=self.base_url,
        )


def create_embedder(
    spec: EmbedderSpec,
    *,
    api_key: str | None = None,
) -> HashingEmbedder | SentenceTransformerEmbedder | OpenAICompatibleEmbedder:
    """Build an embedder from a saved :class:`EmbedderSpec`."""

    if spec.kind == "hashing":
        return HashingEmbedder(
            dimension=spec.dimension,
            ngram_range=(spec.ngram_min, spec.ngram_max),
            seed=spec.seed,
        )
    if spec.kind == "sentence-transformers":
        return SentenceTransformerEmbedder(model_name=spec.model or _DEFAULT_ST_MODEL)
    if spec.kind == "openai":
        return OpenAICompatibleEmbedder(
            model=spec.model or _DEFAULT_OPENAI_MODEL,
            api_key=api_key,
            base_url=spec.base_url or _DEFAULT_BASE_URL,
            dimensions=spec.dimension,
        )
    raise ConfigError(f"Unknown embedder kind: {spec.kind}")


def _embedder_kind(value: object) -> EmbedderKind:
    if value == "hashing":
        return "hashing"
    if value == "sentence-transformers":
        return "sentence-transformers"
    if value == "openai":
        return "openai"
    raise ConfigError(f"Unknown embedder kind: {value!r}")


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigError(f"{label} is invalid")
    return value


def _non_negative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ConfigError(f"{label} is invalid")
    return value


def _l2_normalize(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0.0:
        return values
    return [value / norm for value in values]
