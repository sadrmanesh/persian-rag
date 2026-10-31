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
import re
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from persian_rag.errors import ConfigError, PersianRagError
from persian_rag.remote import embed_texts

EmbedderKind = Literal["hashing", "sentence-transformers", "openai"]
PrefixMode = Literal["auto", "e5", "none"]

DEFAULT_ST_MODEL = "intfloat/multilingual-e5-small"
"""Default sentence-transformers model since 0.4.0."""

LEGACY_ST_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
"""Default sentence-transformers model before 0.4.0."""

E5_BASE_MODEL = "intfloat/multilingual-e5-base"
"""More accurate, larger option: 278M parameters and 768 dimensions."""

ST_MODEL_ALIASES: dict[str, str] = {
    "e5-small": DEFAULT_ST_MODEL,
    "e5-base": E5_BASE_MODEL,
}
"""Short names accepted wherever a sentence-transformers model name is."""

E5_QUERY_PREFIX = "query: "
E5_PASSAGE_PREFIX = "passage: "
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
    query_prefix: str | None = None
    passage_prefix: str | None = None
    """Text put before queries and passages (sentence-transformers only).

    ``None`` means the index was written before 0.4.0, when no prefixes were
    used. An empty string means no prefix.
    """

    def __post_init__(self) -> None:
        if self.dimension < 1:
            raise ConfigError("dimension must be >= 1")
        if self.ngram_min < 1 or self.ngram_max < self.ngram_min:
            raise ConfigError("ngram range is invalid")

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "kind": self.kind,
            "dimension": self.dimension,
            "model": self.model,
            "base_url": self.base_url,
            "ngram_min": self.ngram_min,
            "ngram_max": self.ngram_max,
            "seed": self.seed,
        }
        if self.query_prefix is not None or self.passage_prefix is not None:
            data["query_prefix"] = self.query_prefix or ""
            data["passage_prefix"] = self.passage_prefix or ""
        return data

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> EmbedderSpec:
        kind = _embedder_kind(data.get("kind"))
        dimension = _positive_int(data.get("dimension"), "Embedder dimension")
        model = data.get("model")
        base_url = data.get("base_url")
        query_prefix = _optional_prefix(data.get("query_prefix"), "query_prefix")
        passage_prefix = _optional_prefix(data.get("passage_prefix"), "passage_prefix")
        model_name = model if isinstance(model, str) and model else None
        if kind == "sentence-transformers" and model_name is None and query_prefix is None:
            # Written before 0.4.0 without a model name: that meant the old default.
            model_name = LEGACY_ST_MODEL
        return cls(
            kind=kind,
            dimension=dimension,
            model=model_name,
            base_url=base_url if isinstance(base_url, str) else None,
            ngram_min=_positive_int(data.get("ngram_min", 2), "ngram_min"),
            ngram_max=_positive_int(data.get("ngram_max", 4), "ngram_max"),
            seed=_non_negative_int(data.get("seed", 13), "seed"),
            query_prefix=query_prefix,
            passage_prefix=passage_prefix,
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
    """Adapter for ``sentence-transformers`` models.

    Install the extra: ``pip install 'persian-rag[sentence-transformers]'``.
    The default model, ``intfloat/multilingual-e5-small``, is multilingual and
    was the best small model in the project's Persian measurements (see the
    benchmarks page). It is a default, not a claim about every corpus.
    ``intfloat/multilingual-e5-base`` (alias ``"e5-base"``) is more accurate,
    especially on noisy queries, at about 2.4 times the size, twice the vector
    size, and about 2.4 times slower encoding on CPU. ``"e5-small"`` and
    ``"e5-base"`` are accepted as short names; the full name is stored.

    Prefixes: E5 models expect ``"query: "`` before questions and
    ``"passage: "`` before stored text. ``prefix="auto"`` (the default) uses
    the prompts a model defines in its sentence-transformers config, then the
    E5 prefixes for E5 and persian-rag-embed models, and no prefix for anything
    else. ``prefix="e5"`` forces the E5 prefixes and ``prefix="none"`` turns
    them off. ``query_prefix`` and ``passage_prefix`` override all of that.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_ST_MODEL,
        *,
        device: str | None = None,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        prefix: PrefixMode = "auto",
        query_prefix: str | None = None,
        passage_prefix: str | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if prefix not in ("auto", "e5", "none"):
            raise ValueError("prefix must be 'auto', 'e5', or 'none'")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Install it with: pip install 'persian-rag[sentence-transformers]'"
            ) from exc

        model_name = resolve_model_name(model_name)
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self._model = SentenceTransformer(model_name, device=device)
        # Newer sentence-transformers renamed the size getter.
        get_dimension = getattr(self._model, "get_embedding_dimension", None)
        if get_dimension is None:
            get_dimension = self._model.get_sentence_embedding_dimension
        dimension = get_dimension()
        if not isinstance(dimension, int) or dimension < 1:
            raise PersianRagError(f"Could not read the embedding size of {model_name}")
        self._dimension = dimension
        prompts = _model_prompts(self._model)
        resolved_query, resolved_passage = resolve_prefixes(model_name, prefix, prompts=prompts)
        self.query_prefix = resolved_query if query_prefix is None else query_prefix
        self.passage_prefix = resolved_passage if passage_prefix is None else passage_prefix
        # Use the model's own prompt API when the prefixes are exactly its
        # (non-empty) prompts. Plain encode is used otherwise.
        self._use_prompts = (
            bool(self.query_prefix or self.passage_prefix)
            and prompts.get("query") == self.query_prefix
            and prompts.get("document") == self.passage_prefix
            and hasattr(self._model, "encode_query")
            and hasattr(self._model, "encode_document")
        )

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        if self._use_prompts:
            return self._encode(self._model.encode_document, list(texts))
        return self._encode(self._model.encode, [self.passage_prefix + text for text in texts])

    def embed_query(self, text: str) -> list[float]:
        if self._use_prompts:
            return self._encode(self._model.encode_query, [text])[0]
        return self._encode(self._model.encode, [self.query_prefix + text])[0]

    def to_spec(self) -> EmbedderSpec:
        return EmbedderSpec(
            kind="sentence-transformers",
            dimension=self._dimension,
            model=self.model_name,
            query_prefix=self.query_prefix,
            passage_prefix=self.passage_prefix,
        )

    def _encode(self, method: Any, texts: list[str]) -> list[list[float]]:
        encoded = method(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [[float(item) for item in row] for row in encoded]


def resolve_model_name(model_name: str) -> str:
    """Expand a short name (``"e5-small"``, ``"e5-base"``) to the full model name.

    Any other name is returned unchanged.
    """

    return ST_MODEL_ALIASES.get(model_name.strip().lower(), model_name)


_E5_NAME_RE = re.compile(r"(^|[/_-])(m?e5)([-_]|$)", re.IGNORECASE)


def is_e5_model(model_name: str) -> bool:
    """Return ``True`` for E5 style models that expect query and passage prefixes.

    Matches names such as ``intfloat/multilingual-e5-small``, ``e5-base-v2``
    and the ``persian-rag-embed`` models. Instruction tuned E5 models
    (``-instruct``) use a different query format and are excluded.
    """

    name = model_name.lower()
    if "instruct" in name:
        return False
    return bool(_E5_NAME_RE.search(name)) or "persian-rag-embed" in name


def resolve_prefixes(
    model_name: str,
    mode: PrefixMode = "auto",
    *,
    prompts: dict[str, str] | None = None,
) -> tuple[str, str]:
    """Return ``(query_prefix, passage_prefix)`` for a model and prefix mode."""

    if mode == "none":
        return "", ""
    if mode == "e5":
        return E5_QUERY_PREFIX, E5_PASSAGE_PREFIX
    if mode != "auto":
        raise ValueError("prefix must be 'auto', 'e5', or 'none'")
    query = (prompts or {}).get("query")
    document = (prompts or {}).get("document")
    if query is not None and document is not None and (query or document):
        # Recent sentence-transformers versions set empty "query"/"document"
        # prompts on every model, so only non-empty prompts count.
        return query, document
    if is_e5_model(model_name):
        return E5_QUERY_PREFIX, E5_PASSAGE_PREFIX
    return "", ""


def _model_prompts(model: object) -> dict[str, str]:
    raw = getattr(model, "prompts", None)
    if not isinstance(raw, dict):
        return {}
    return {str(key): value for key, value in raw.items() if isinstance(value, str)}


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
        model_name = resolve_model_name(spec.model or DEFAULT_ST_MODEL)
        if spec.query_prefix is None and spec.passage_prefix is None:
            # Index written before 0.4.0: its vectors were made without prefixes.
            if is_e5_model(model_name):
                warnings.warn(
                    f"This index was built with {model_name} before persian-rag 0.4.0, "
                    "without the E5 query/passage prefixes. It keeps working as built, "
                    "but rebuilding it with 'persian-rag ingest' gives better retrieval.",
                    UserWarning,
                    stacklevel=2,
                )
            return SentenceTransformerEmbedder(
                model_name=model_name, query_prefix="", passage_prefix=""
            )
        return SentenceTransformerEmbedder(
            model_name=model_name,
            query_prefix=spec.query_prefix or "",
            passage_prefix=spec.passage_prefix or "",
        )
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


def _optional_prefix(value: object, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ConfigError(f"{label} must be a string")
    return value


def _l2_normalize(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0.0:
        return values
    return [value / norm for value in values]
