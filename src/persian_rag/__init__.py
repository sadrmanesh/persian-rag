"""Persian-first retrieval-augmented generation.

Normalize Farsi text, split it into chunks, and retrieve passages with a
hybrid of BM25 and dense vectors, optionally rerank them, and evaluate the
results. Generation is delegated to any callable.

Server-backed vector stores live in :mod:`persian_rag.vectorstores`, PDF and
DOCX loaders in :mod:`persian_rag.loaders`, and the HTTP API in
:mod:`persian_rag.server`; their dependencies are optional extras.
"""

from persian_rag.bm25 import BM25Index
from persian_rag.cache import (
    CachedEmbedder,
    InMemoryEmbeddingCache,
    LRUCache,
    SQLiteEmbeddingCache,
)
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
    MissingDependencyError,
    PersianRagError,
)
from persian_rag.evaluation import (
    EvalExample,
    LLMFaithfulnessJudge,
    RetrievalReport,
    TokenOverlapFaithfulness,
    evaluate_answers,
    evaluate_retrieval,
    load_eval_examples,
)
from persian_rag.loaders import clean_extracted_text, load_docx, load_pdf
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
from persian_rag.rerank import CrossEncoderReranker, LLMReranker, Reranker
from persian_rag.retrieval import HybridRetriever, RetrievalResult, reciprocal_rank_fusion
from persian_rag.store import FaissVectorStore, InMemoryVectorStore, VectorHit, VectorStore
from persian_rag.tokenize import split_sentence_spans, split_sentences, tokenize
from persian_rag.version import __version__

__all__ = [
    "DEFAULT_PROMPT_EN",
    "DEFAULT_PROMPT_FA",
    "BM25Index",
    "CachedEmbedder",
    "Chunk",
    "ChunkingConfig",
    "ConfigError",
    "CrossEncoderReranker",
    "Embedder",
    "EmbedderSpec",
    "EvalExample",
    "FaissVectorStore",
    "HashingEmbedder",
    "HybridRetriever",
    "InMemoryEmbeddingCache",
    "InMemoryVectorStore",
    "IndexFormatError",
    "IndexNotFoundError",
    "LLMFaithfulnessJudge",
    "LLMReranker",
    "LRUCache",
    "MissingDependencyError",
    "NormalizeConfig",
    "OpenAICompatibleEmbedder",
    "PersianRagError",
    "RAGAnswer",
    "RAGPipeline",
    "Reranker",
    "RetrievalReport",
    "RetrievalResult",
    "SQLiteEmbeddingCache",
    "SavedIndex",
    "SentenceTransformerEmbedder",
    "SourceDocument",
    "TokenOverlapFaithfulness",
    "VectorHit",
    "VectorStore",
    "__version__",
    "build_retriever",
    "chunk_text",
    "clean_extracted_text",
    "create_embedder",
    "evaluate_answers",
    "evaluate_retrieval",
    "load_documents",
    "load_docx",
    "load_eval_examples",
    "load_index",
    "load_pdf",
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
