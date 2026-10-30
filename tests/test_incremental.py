from __future__ import annotations

import json
from pathlib import Path

import pytest

from persian_rag.chunk import ChunkingConfig
from persian_rag.documents import SourceDocument
from persian_rag.embeddings import HashingEmbedder
from persian_rag.incremental import (
    IncrementalIndexer,
    IndexManifest,
    content_hash,
)
from persian_rag.normalize import NormalizeConfig
from persian_rag.persist import load_index, retriever_from_saved, save_index
from persian_rag.retrieval import HybridRetriever


class CountingEmbedder(HashingEmbedder):
    def __init__(self) -> None:
        super().__init__(dimension=64)
        self.embedded: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:  # type: ignore[override]
        self.embedded.extend(texts)
        return super().embed_documents(texts)


def _docs(**texts: str) -> list[SourceDocument]:
    return [SourceDocument(id=key, text=value, metadata={}) for key, value in texts.items()]


CHUNKING = ChunkingConfig(strategy="sentence", chunk_size=60, overlap=0, sentence_overlap=0)


def _indexer() -> tuple[IncrementalIndexer, CountingEmbedder]:
    embedder = CountingEmbedder()
    retriever = HybridRetriever(embedder)
    return IncrementalIndexer(retriever, chunking=CHUNKING, manifest=IndexManifest()), embedder


def test_hash_depends_on_text_metadata_and_settings() -> None:
    base = content_hash("متن", {"a": 1})
    assert base == content_hash("متن", {"a": 1})
    assert base != content_hash("متن دیگر", {"a": 1})
    assert base != content_hash("متن", {"a": 2})
    assert base != content_hash("متن", {"a": 1}, chunking=ChunkingConfig(chunk_size=500))
    assert base != content_hash("متن", {"a": 1}, normalize_config=NormalizeConfig(digits="ascii"))


def test_sync_adds_updates_deletes_and_skips() -> None:
    indexer, embedder = _indexer()
    first = indexer.sync(_docs(a="حافظ در شیراز بود.", b="فردوسی شاهنامه را سرود."))
    assert first.added == ("a", "b")
    assert first.chunks_added == 2
    embedder.embedded.clear()

    second = indexer.sync(_docs(a="حافظ در شیراز بود.", b="فردوسی شاهنامه را سرود."))
    assert second.unchanged == ("a", "b")
    assert not second.changed
    assert embedder.embedded == []

    third = indexer.sync(_docs(a="حافظ در شیراز زندگی کرد.", c="تهران پایتخت است."))
    assert third.updated == ("a",)
    assert third.added == ("c",)
    assert third.deleted == ("b",)
    assert sorted(indexer.retriever.document_ids()) == ["a", "c"]
    assert "b" not in indexer.manifest
    assert indexer.retriever.retrieve("شاهنامه", k=5, mode="sparse") == []
    assert "زندگی" in indexer.retriever.retrieve("زندگی", k=1, mode="sparse")[0].text


def test_unchanged_chunks_keep_their_vectors() -> None:
    indexer, embedder = _indexer()
    long_text = "جمله اول درباره حافظ است. جمله دوم درباره سعدی است. جمله سوم درباره مولوی است."
    indexer.sync(_docs(a=long_text))
    chunks_before = len(indexer.retriever)
    assert chunks_before > 1
    embedder.embedded.clear()
    report = indexer.sync(_docs(a=long_text.replace("مولوی", "نظامی")))
    assert report.updated == ("a",)
    assert report.chunks_reused == chunks_before - 1
    assert len(embedder.embedded) == 1


def test_keep_missing_and_delete() -> None:
    indexer, _ = _indexer()
    indexer.sync(_docs(a="یک.", b="دو."))
    report = indexer.upsert(_docs(c="سه."))
    assert report.added == ("c",)
    assert sorted(indexer.retriever.document_ids()) == ["a", "b", "c"]
    removed = indexer.delete(["a", "missing"])
    assert removed.deleted == ("a",)
    assert sorted(indexer.manifest.documents) == ["b", "c"]


def test_duplicate_ids_are_rejected() -> None:
    indexer, _ = _indexer()
    with pytest.raises(ValueError, match="duplicate"):
        indexer.sync(_docs(a="x") + _docs(a="y"))


def test_manifest_round_trips_through_saved_index(tmp_path: Path) -> None:
    indexer, embedder = _indexer()
    indexer.sync(_docs(a="حافظ در شیراز بود.", b="فردوسی شاهنامه را سرود."))
    target = save_index(
        tmp_path, indexer.retriever, embedder.to_spec(), CHUNKING, manifest=indexer.manifest
    )
    raw = json.loads(target.read_text(encoding="utf-8"))
    assert set(raw["documents"]) == {"a", "b"}
    saved = load_index(tmp_path)
    manifest = saved.manifest()
    assert manifest.documents["a"].chunk_ids == ("a:0",)
    restored = retriever_from_saved(saved, HashingEmbedder(dimension=64))
    again = IncrementalIndexer(restored, chunking=saved.chunking, manifest=manifest)
    report = again.sync(_docs(a="حافظ در شیراز بود.", b="فردوسی شاهنامه را سرود."))
    assert report.unchanged == ("a", "b")


def test_index_without_manifest_treats_documents_as_changed() -> None:
    retriever = HybridRetriever(HashingEmbedder(dimension=64))
    seed = IncrementalIndexer(retriever, chunking=CHUNKING, manifest=IndexManifest())
    seed.sync(_docs(a="یک."))
    legacy = IncrementalIndexer(retriever, chunking=CHUNKING)
    assert legacy.manifest.documents["a"].hash == ""
    report = legacy.sync(_docs(a="یک."))
    assert report.updated == ("a",)
    assert len(retriever) == 1
