from __future__ import annotations

import pytest

from persian_rag.embeddings import HashingEmbedder
from persian_rag.pipeline import build_retriever
from persian_rag.retrieval import HybridRetriever

CORPUS = {
    "hafez": "حافظ شیرازی در شهر شیراز زاده شد و دیوان غزلیات او مشهور است.",
    "ferdowsi": "فردوسی شاهنامه را در توس سرود و داستان رستم را روایت کرد.",
    "football": "فوتبال ورزشی گروهی است که با دو تیم یازده نفره بازی می شود.",
    "tehran": "تهران پایتخت ایران است و کوه دماوند در نزدیکی آن قرار دارد.",
}


@pytest.fixture
def corpus_retriever() -> HybridRetriever:
    return build_retriever(
        list(CORPUS.values()),
        HashingEmbedder(dimension=128),
        document_ids=list(CORPUS),
    )
