"""Persian-first retrieval-augmented generation.

Normalize Farsi text, split it into chunks, and retrieve passages with a
hybrid of BM25 and dense vectors, optionally rerank them, and evaluate the
results. Generation is delegated to any callable.

Server-backed vector stores (Qdrant, pgvector, Elasticsearch, OpenSearch)
live in :mod:`persian_rag.vectorstores`, PDF and DOCX loaders in
:mod:`persian_rag.loaders`, HTML and web loaders in :mod:`persian_rag.web`,
and the HTTP API in :mod:`persian_rag.server`.
"""

from persian_rag.bm25 import BM25Index
from persian_rag.cache import (
    CachedEmbedder,
    InMemoryEmbeddingCache,
    LRUCache,
    SQLiteEmbeddingCache,
)
from persian_rag.chat import ChatAnswer, ChatSession, ConversationMemory
from persian_rag.chunk import Chunk, ChunkingConfig, chunk_text
from persian_rag.citations import (
    CITATION_PROMPT_EN,
    CITATION_PROMPT_FA,
    Citation,
    CitationReport,
    extract_citations,
)
from persian_rag.documents import (
    SourceDocument,
    load_documents,
    load_markdown,
    load_markdown_text,
    strip_markdown,
)
from persian_rag.embeddings import (
    DEFAULT_ST_MODEL,
    E5_BASE_MODEL,
    LEGACY_ST_MODEL,
    ST_MODEL_ALIASES,
    Embedder,
    EmbedderSpec,
    HashingEmbedder,
    OpenAICompatibleEmbedder,
    SentenceTransformerEmbedder,
    create_embedder,
    is_e5_model,
    resolve_model_name,
    resolve_prefixes,
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
from persian_rag.incremental import (
    IncrementalIndexer,
    IndexManifest,
    ReindexReport,
    content_hash,
)
from persian_rag.llm import OllamaLLM, OpenAICompatibleLLM, create_llm
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
from persian_rag.retrieval import (
    HybridRetriever,
    RetrievalResult,
    SparseBackend,
    reciprocal_rank_fusion,
)
from persian_rag.rewrite import QueryRewriter, RewriteConfig, RewritingRetriever
from persian_rag.store import FaissVectorStore, InMemoryVectorStore, VectorHit, VectorStore
from persian_rag.tokenize import split_sentence_spans, split_sentences, tokenize
from persian_rag.version import __version__
from persian_rag.web import extract_main_content, html_to_text, load_html, load_url, load_urls

__all__ = [
    "CITATION_PROMPT_EN",
    "CITATION_PROMPT_FA",
    "DEFAULT_PROMPT_EN",
    "DEFAULT_PROMPT_FA",
    "DEFAULT_ST_MODEL",
    "E5_BASE_MODEL",
    "LEGACY_ST_MODEL",
    "ST_MODEL_ALIASES",
    "BM25Index",
    "CachedEmbedder",
    "ChatAnswer",
    "ChatSession",
    "Chunk",
    "ChunkingConfig",
    "Citation",
    "CitationReport",
    "ConfigError",
    "ConversationMemory",
    "CrossEncoderReranker",
    "Embedder",
    "EmbedderSpec",
    "EvalExample",
    "FaissVectorStore",
    "HashingEmbedder",
    "HybridRetriever",
    "InMemoryEmbeddingCache",
    "InMemoryVectorStore",
    "IncrementalIndexer",
    "IndexFormatError",
    "IndexManifest",
    "IndexNotFoundError",
    "LLMFaithfulnessJudge",
    "LLMReranker",
    "LRUCache",
    "MissingDependencyError",
    "NormalizeConfig",
    "OllamaLLM",
    "OpenAICompatibleEmbedder",
    "OpenAICompatibleLLM",
    "PersianRagError",
    "QueryRewriter",
    "RAGAnswer",
    "RAGPipeline",
    "ReindexReport",
    "Reranker",
    "RetrievalReport",
    "RetrievalResult",
    "RewriteConfig",
    "RewritingRetriever",
    "SQLiteEmbeddingCache",
    "SavedIndex",
    "SentenceTransformerEmbedder",
    "SourceDocument",
    "SparseBackend",
    "TokenOverlapFaithfulness",
    "VectorHit",
    "VectorStore",
    "__version__",
    "build_retriever",
    "chunk_text",
    "clean_extracted_text",
    "content_hash",
    "create_embedder",
    "create_llm",
    "evaluate_answers",
    "evaluate_retrieval",
    "extract_citations",
    "extract_main_content",
    "html_to_text",
    "is_e5_model",
    "load_documents",
    "load_docx",
    "load_eval_examples",
    "load_html",
    "load_index",
    "load_markdown",
    "load_markdown_text",
    "load_pdf",
    "load_url",
    "load_urls",
    "make_openai_llm",
    "normalize",
    "reciprocal_rank_fusion",
    "resolve_model_name",
    "resolve_prefixes",
    "retriever_from_saved",
    "save_index",
    "split_sentence_spans",
    "split_sentences",
    "strip_markdown",
    "tokenize",
]
