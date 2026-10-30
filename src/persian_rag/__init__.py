"""Persian-first retrieval-augmented generation.

Normalize Farsi text, split it into chunks, and retrieve passages with a
hybrid of BM25 and dense vectors. Generation is delegated to any callable.
"""

from persian_rag.bm25 import BM25Index
from persian_rag.chunk import Chunk, ChunkingConfig, chunk_text
from persian_rag.documents import SourceDocument, load_documents, strip_markdown
from persian_rag.embeddings import (
    Embedder,
    EmbedderSpec,
    HashingEmbedder,
    OpenAICompatibleEmbedder,
    SentenceTransformerEmbedder,
    create_embedder,
)
from persian_rag.errors import (
    ConfigError,
    IndexFormatError,
    IndexNotFoundError,
    PersianRagError,
)
from persian_rag.normalize import NormalizeConfig, normalize
from persian_rag.persist import SavedIndex, load_index, retriever_from_saved, save_index
from persian_rag.pipeline import (
    DEFAULT_PROMPT_EN,
    DEFAULT_PROMPT_FA,
    RAGAnswer,
    RAGPipeline,
    build_retriever,
)
from persian_rag.remote import make_openai_llm
from persian_rag.retrieval import HybridRetriever, RetrievalResult, reciprocal_rank_fusion
from persian_rag.store import FaissVectorStore, InMemoryVectorStore, VectorHit, VectorStore
from persian_rag.tokenize import split_sentence_spans, split_sentences, tokenize
from persian_rag.version import __version__

__all__ = [
    "DEFAULT_PROMPT_EN",
    "DEFAULT_PROMPT_FA",
    "BM25Index",
    "Chunk",
    "ChunkingConfig",
    "ConfigError",
    "Embedder",
    "EmbedderSpec",
    "FaissVectorStore",
    "HashingEmbedder",
    "HybridRetriever",
    "InMemoryVectorStore",
    "IndexFormatError",
    "IndexNotFoundError",
    "NormalizeConfig",
    "OpenAICompatibleEmbedder",
    "PersianRagError",
    "RAGAnswer",
    "RAGPipeline",
    "RetrievalResult",
    "SavedIndex",
    "SentenceTransformerEmbedder",
    "SourceDocument",
    "VectorHit",
    "VectorStore",
    "__version__",
    "build_retriever",
    "chunk_text",
    "create_embedder",
    "load_documents",
    "load_index",
    "make_openai_llm",
    "normalize",
    "reciprocal_rank_fusion",
    "retriever_from_saved",
    "save_index",
    "split_sentence_spans",
    "split_sentences",
    "strip_markdown",
    "tokenize",
]
