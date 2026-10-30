"""Incremental reindexing with content hashes.

Re-running ingestion on a large folder should not re-embed every file. An
:class:`IndexManifest` records, per source document, a SHA-256 hash of its
text, metadata, and the normalization and chunking settings, plus the ids of
its chunks. :class:`IncrementalIndexer` compares a fresh set of documents to
the manifest and then:

* adds new documents;
* replaces changed documents (old chunks are deleted, new ones added);
* deletes documents that disappeared (unless ``delete_missing=False``);
* leaves unchanged documents alone.

Chunks of a changed document whose text did not change keep their vectors,
so an edit to one paragraph re-embeds only the affected chunks.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from persian_rag.chunk import Chunk, ChunkingConfig, chunk_text
from persian_rag.documents import SourceDocument
from persian_rag.normalize import NormalizeConfig, normalize
from persian_rag.retrieval import HybridRetriever


def content_hash(
    text: str,
    metadata: Mapping[str, Any] | None = None,
    *,
    normalize_config: NormalizeConfig | None = None,
    chunking: ChunkingConfig | None = None,
) -> str:
    """Return a stable SHA-256 hex digest for a document and its settings.

    The hash changes when the text, the metadata, or the normalization and
    chunking settings change, because each of those changes the chunks.
    """

    payload = {
        "text": text,
        "metadata": dict(metadata or {}),
        "normalize": (normalize_config or NormalizeConfig()).to_dict(),
        "chunking": (chunking or ChunkingConfig()).to_dict(),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class DocumentState:
    """What the index holds for one source document."""

    hash: str
    chunk_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"hash": self.hash, "chunks": list(self.chunk_ids)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DocumentState:
        digest = data.get("hash")
        chunks = data.get("chunks", [])
        if not isinstance(digest, str) or not isinstance(chunks, list):
            raise ValueError("document state needs a string hash and a list of chunks")
        return cls(hash=digest, chunk_ids=tuple(str(item) for item in chunks))


@dataclass(slots=True)
class IndexManifest:
    """Document hashes and chunk ids, saved next to the chunks."""

    documents: dict[str, DocumentState] = field(default_factory=dict)

    def __contains__(self, document_id: object) -> bool:
        return document_id in self.documents

    def __len__(self) -> int:
        return len(self.documents)

    def to_dict(self) -> dict[str, Any]:
        return {doc_id: state.to_dict() for doc_id, state in self.documents.items()}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> IndexManifest:
        manifest = cls()
        for doc_id, raw in (data or {}).items():
            if isinstance(raw, Mapping):
                manifest.documents[str(doc_id)] = DocumentState.from_dict(raw)
        return manifest

    @classmethod
    def from_retriever(cls, retriever: HybridRetriever) -> IndexManifest:
        """Build a manifest without hashes from an existing retriever.

        Every document is treated as changed on the next sync, which is the
        safe choice for indexes written before manifests existed.
        """

        manifest = cls()
        for doc_id in retriever.document_ids():
            manifest.documents[doc_id] = DocumentState(
                hash="", chunk_ids=tuple(retriever.chunk_ids_for_document(doc_id))
            )
        return manifest


@dataclass(frozen=True, slots=True)
class ReindexReport:
    """Document ids touched by a sync, and chunk counts."""

    added: tuple[str, ...] = ()
    updated: tuple[str, ...] = ()
    deleted: tuple[str, ...] = ()
    unchanged: tuple[str, ...] = ()
    chunks_added: int = 0
    chunks_deleted: int = 0
    chunks_reused: int = 0

    @property
    def changed(self) -> bool:
        return bool(self.added or self.updated or self.deleted)

    def summary(self) -> str:
        return (
            f"added {len(self.added)}, updated {len(self.updated)}, "
            f"deleted {len(self.deleted)}, unchanged {len(self.unchanged)} document(s); "
            f"+{self.chunks_added} / -{self.chunks_deleted} chunk(s), "
            f"{self.chunks_reused} vector(s) reused"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "added": list(self.added),
            "updated": list(self.updated),
            "deleted": list(self.deleted),
            "unchanged": list(self.unchanged),
            "chunks_added": self.chunks_added,
            "chunks_deleted": self.chunks_deleted,
            "chunks_reused": self.chunks_reused,
        }


class IncrementalIndexer:
    """Keep a :class:`~persian_rag.retrieval.HybridRetriever` in sync with sources.

    The retriever's vector store must support ``delete``. ``chunking``
    defaults to :class:`~persian_rag.chunk.ChunkingConfig`; normalization
    always follows ``retriever.normalize_config``.
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        *,
        chunking: ChunkingConfig | None = None,
        manifest: IndexManifest | None = None,
    ) -> None:
        self.retriever = retriever
        self.chunking = chunking or ChunkingConfig()
        self.manifest = (
            manifest if manifest is not None else IndexManifest.from_retriever(retriever)
        )

    def hash_document(self, document: SourceDocument) -> str:
        return content_hash(
            document.text,
            document.metadata,
            normalize_config=self.retriever.normalize_config,
            chunking=self.chunking,
        )

    def chunk_document(self, document: SourceDocument) -> list[Chunk]:
        normalized = normalize(document.text, self.retriever.normalize_config)
        return chunk_text(
            normalized,
            self.chunking,
            document_id=document.id,
            metadata=dict(document.metadata),
        )

    def sync(
        self,
        documents: Iterable[SourceDocument],
        *,
        delete_missing: bool = True,
    ) -> ReindexReport:
        """Make the index match ``documents``.

        With ``delete_missing=False`` documents absent from ``documents``
        stay in the index, which suits feeding a subset of files.
        """

        incoming: dict[str, SourceDocument] = {}
        for document in documents:
            if document.id in incoming:
                raise ValueError(f"duplicate document id: {document.id}")
            incoming[document.id] = document
        added: list[str] = []
        updated: list[str] = []
        unchanged: list[str] = []
        counts = {"added": 0, "deleted": 0, "reused": 0}
        for doc_id, document in incoming.items():
            digest = self.hash_document(document)
            state = self.manifest.documents.get(doc_id)
            if state is None and not self.retriever.chunk_ids_for_document(doc_id):
                self._write(document, digest, reuse=False, counts=counts)
                added.append(doc_id)
            elif state is not None and state.hash == digest and state.hash:
                unchanged.append(doc_id)
            else:
                self._write(document, digest, reuse=True, counts=counts)
                updated.append(doc_id)
        deleted: list[str] = []
        if delete_missing:
            for doc_id in [d for d in self.manifest.documents if d not in incoming]:
                counts["deleted"] += self._remove(doc_id)
                deleted.append(doc_id)
        return ReindexReport(
            added=tuple(added),
            updated=tuple(updated),
            deleted=tuple(deleted),
            unchanged=tuple(unchanged),
            chunks_added=counts["added"],
            chunks_deleted=counts["deleted"],
            chunks_reused=counts["reused"],
        )

    def upsert(self, documents: Iterable[SourceDocument]) -> ReindexReport:
        """Add or replace ``documents`` without deleting anything else."""

        return self.sync(documents, delete_missing=False)

    def delete(self, document_ids: Sequence[str]) -> ReindexReport:
        """Remove documents from the index and the manifest."""

        removed: list[str] = []
        chunks = 0
        for doc_id in dict.fromkeys(document_ids):
            if doc_id in self.manifest or self.retriever.chunk_ids_for_document(doc_id):
                chunks += self._remove(doc_id)
                removed.append(doc_id)
        return ReindexReport(deleted=tuple(removed), chunks_deleted=chunks)

    def _remove(self, doc_id: str) -> int:
        state = self.manifest.documents.pop(doc_id, None)
        ids = list(state.chunk_ids) if state is not None else []
        ids.extend(self.retriever.chunk_ids_for_document(doc_id))
        return self.retriever.delete_chunks(ids)

    def _write(
        self,
        document: SourceDocument,
        digest: str,
        *,
        reuse: bool,
        counts: dict[str, int],
    ) -> None:
        chunks = self.chunk_document(document)
        old_vectors: dict[str, list[float]] = {}
        if reuse:
            existing = {chunk.id: chunk for chunk in self.retriever.chunks}
            stale = self.retriever.chunk_ids_for_document(document.id)
            state = self.manifest.documents.get(document.id)
            if state is not None:
                stale = list(dict.fromkeys([*state.chunk_ids, *stale]))
            vectors = dict(zip(self._order(), self.retriever.embeddings(), strict=True))
            for chunk_id in stale:
                chunk = existing.get(chunk_id)
                if chunk is not None and chunk_id in vectors:
                    old_vectors.setdefault(chunk.text, vectors[chunk_id])
            counts["deleted"] += self.retriever.delete_chunks(stale)
        if chunks:
            fresh = [chunk for chunk in chunks if chunk.text not in old_vectors]
            if fresh:
                new_vectors = self.retriever.embedder.embed_documents(
                    [chunk.text for chunk in fresh]
                )
                for chunk, vector in zip(fresh, new_vectors, strict=True):
                    old_vectors.setdefault(chunk.text, list(vector))
            counts["reused"] += len(chunks) - len(fresh)
            self.retriever.add_chunks(chunks, embeddings=[old_vectors[c.text] for c in chunks])
            counts["added"] += len(chunks)
        self.manifest.documents[document.id] = DocumentState(
            hash=digest, chunk_ids=tuple(chunk.id for chunk in chunks)
        )

    def _order(self) -> list[str]:
        return [chunk.id for chunk in self.retriever.chunks]
