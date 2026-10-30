"""Save and load a JSON index.

The file stores normalized chunks and their vectors so a later query does
not re-embed the corpus. Query embedding still needs the same embedder.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from persian_rag.chunk import Chunk, ChunkingConfig
from persian_rag.embeddings import Embedder, EmbedderSpec, create_embedder
from persian_rag.errors import IndexFormatError, IndexNotFoundError
from persian_rag.normalize import NormalizeConfig
from persian_rag.retrieval import HybridRetriever
from persian_rag.version import INDEX_VERSION

INDEX_FILENAME = "index.json"


@dataclass(frozen=True, slots=True)
class SavedIndex:
    """An index read from disk."""

    version: str
    embedder: EmbedderSpec
    normalize: NormalizeConfig
    chunking: ChunkingConfig
    chunks: list[Chunk]
    embeddings: list[list[float]]


def resolve_index_path(path: str | Path) -> Path:
    """Return the JSON file for a directory or an explicit ``.json`` path."""

    candidate = Path(path)
    if candidate.suffix == ".json":
        return candidate
    return candidate / INDEX_FILENAME


def save_index(
    path: str | Path,
    retriever: HybridRetriever,
    embedder_spec: EmbedderSpec,
    chunking: ChunkingConfig,
) -> Path:
    """Write ``retriever`` to ``path`` and return the file that was written."""

    chunks = retriever.chunks
    vectors = retriever.embeddings()
    if len(chunks) != len(vectors):
        raise IndexFormatError("Chunks and embeddings are out of sync")
    if vectors and len(vectors[0]) != embedder_spec.dimension:
        raise IndexFormatError(
            "Embedder dimension does not match the stored vectors: "
            f"{embedder_spec.dimension} != {len(vectors[0])}"
        )

    records: list[dict[str, Any]] = []
    for chunk, vector in zip(chunks, vectors, strict=True):
        payload = chunk.to_dict()
        payload["metadata"] = _json_object(chunk.metadata, f"metadata of {chunk.id}")
        payload["embedding"] = [float(value) for value in vector]
        records.append(payload)

    document = {
        "version": INDEX_VERSION,
        "embedder": embedder_spec.to_dict(),
        "normalize": retriever.normalize_config.to_dict(),
        "chunking": chunking.to_dict(),
        "chunks": records,
    }
    target = resolve_index_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target


def load_index(path: str | Path) -> SavedIndex:
    """Read an index written by :func:`save_index`."""

    target = resolve_index_path(path)
    if not target.is_file():
        raise IndexNotFoundError(f"No index at {target}")
    try:
        parsed = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise IndexFormatError(f"{target} is not valid JSON") from exc
    if not isinstance(parsed, dict):
        raise IndexFormatError("Index root must be a JSON object")
    root = cast(dict[str, Any], parsed)

    version = root.get("version")
    if version != INDEX_VERSION:
        raise IndexFormatError(f"Unsupported index version: {version!r}")

    embedder_raw = root.get("embedder")
    normalize_raw = root.get("normalize")
    chunking_raw = root.get("chunking")
    chunk_raw = root.get("chunks")
    if not isinstance(embedder_raw, dict):
        raise IndexFormatError("Index is missing embedder settings")
    if not isinstance(normalize_raw, dict):
        raise IndexFormatError("Index is missing normalization settings")
    if not isinstance(chunking_raw, dict):
        raise IndexFormatError("Index is missing chunking settings")
    if not isinstance(chunk_raw, list):
        raise IndexFormatError("Index chunks must be a list")

    spec = EmbedderSpec.from_dict(cast(dict[str, Any], embedder_raw))
    chunks: list[Chunk] = []
    embeddings: list[list[float]] = []
    for item in chunk_raw:
        if not isinstance(item, dict):
            raise IndexFormatError("Each chunk must be a JSON object")
        record = cast(dict[str, Any], item)
        embedding = record.get("embedding")
        if not isinstance(embedding, list) or not embedding:
            raise IndexFormatError(f"Chunk {record.get('id')!r} is missing an embedding")
        vector = [_as_float(value) for value in embedding]
        if len(vector) != spec.dimension:
            raise IndexFormatError(
                f"Chunk {record.get('id')!r} has dimension {len(vector)}, expected {spec.dimension}"
            )
        try:
            chunk = Chunk.from_dict(record)
        except (KeyError, TypeError, ValueError) as exc:
            raise IndexFormatError(f"Invalid chunk: {exc}") from exc
        chunks.append(chunk)
        embeddings.append(vector)

    try:
        normalize_config = NormalizeConfig.from_dict(cast(dict[str, Any], normalize_raw))
        chunking = ChunkingConfig.from_dict(cast(dict[str, Any], chunking_raw))
    except (TypeError, ValueError) as exc:
        raise IndexFormatError(f"Invalid index configuration: {exc}") from exc

    return SavedIndex(
        version=str(version),
        embedder=spec,
        normalize=normalize_config,
        chunking=chunking,
        chunks=chunks,
        embeddings=embeddings,
    )


def retriever_from_saved(saved: SavedIndex, embedder: Embedder | None = None) -> HybridRetriever:
    """Rebuild a retriever from a saved index.

    When ``embedder`` is omitted, one is created from the saved spec. That
    loads a sentence-transformers model or requires an API key for the
    OpenAI-compatible adapter.
    """

    model = embedder if embedder is not None else create_embedder(saved.embedder)
    if model.dimension != saved.embedder.dimension:
        raise IndexFormatError(
            "Embedder dimension does not match the index: "
            f"{model.dimension} != {saved.embedder.dimension}"
        )
    retriever = HybridRetriever(model, normalize_config=saved.normalize)
    retriever.add_chunks(saved.chunks, embeddings=saved.embeddings)
    return retriever


def _as_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise IndexFormatError("Embedding values must be numeric")
    return float(value)


def _json_object(value: dict[str, Any], label: str) -> dict[str, Any]:
    try:
        encoded = json.dumps(value, ensure_ascii=False)
        parsed = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise IndexFormatError(f"{label} is not JSON serializable") from exc
    if not isinstance(parsed, dict):
        raise IndexFormatError(f"{label} must be a JSON object")
    return cast(dict[str, Any], parsed)
