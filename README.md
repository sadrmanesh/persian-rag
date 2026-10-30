# persian-rag

[![CI](https://github.com/sadrmanesh/persian-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/sadrmanesh/persian-rag/actions/workflows/ci.yml)

Persian-first retrieval-augmented generation for Python.

persian-rag is a small toolkit for building RAG over Farsi text. It normalizes Arabic and Persian characters, half-spaces, digits, and punctuation; splits and chunks text; embeds passages through a pluggable interface; and retrieves them with BM25, dense vectors, or both. Generation is a function you supply. Version 0.1.0 does not ship a language model and does not publish benchmark scores.

persian-rag یک جعبه‌ابزار پایتون برای بازیابی‌افزوده روی متن فارسی است. نویسه‌های عربی و فارسی، نیم‌فاصله، رقم‌ها و نشانه‌گذاری را یکدست می‌کند، متن را به تکه تقسیم می‌کند، و با BM25، بردار متراکم، یا ترکیب هر دو جست‌وجو می‌کند. تولید پاسخ به تابعی سپرده می‌شود که خودتان می‌دهید. نسخه ۰٫۱٫۰ مدل زبانی همراه ندارد و امتیازی برای کیفیت ادعا نمی‌کند.

## Features

- Persian text normalization: Arabic yeh/kaf and common presentation forms, ZWNJ, digits, optional diacritics, and punctuation.
- Sentence splitting and tokenization that keep half-spaces inside words such as `می‌روم`.
- Chunking: fixed size with overlap, sentence packing, and recursive separators.
- Embedder interface with adapters for `sentence-transformers` and any OpenAI-compatible embeddings API.
- An in-memory cosine store and an optional FAISS adapter.
- Hybrid retrieval with [reciprocal rank fusion](https://plg.uwaterloo.ca/gvcormac/cormacksigir09-rrf.pdf).
- A pipeline that accepts any `Callable[[str], str]` as the language model.
- A CLI that indexes a folder of `.txt` and `.md` files and queries the index.
- Optional extras for the heavy dependencies. The default install is pure Python.

## Installation

The project name is `persian-rag`. From a checkout of this repository:

```bash
pip install -e .
```

Optional pieces:

```bash
pip install -e ".[sentence-transformers]"
pip install -e ".[faiss]"
pip install -e ".[all]"
pip install -e ".[dev]"   # pytest, ruff, mypy
```

Requires Python 3.10 or newer.

The OpenAI-compatible adapter uses the standard library. Set `OPENAI_API_KEY`. `OPENAI_BASE_URL` overrides the default `https://api.openai.com/v1` for compatible servers.

## Quick start

`HashingEmbedder` is a deterministic character n-gram hash for tests and offline smoke checks. Semantic retrieval uses `SentenceTransformerEmbedder` or `OpenAICompatibleEmbedder`.

```python
from persian_rag import HashingEmbedder, RAGPipeline, build_retriever

texts = [
    "حافظ در شهر شیراز زاده شد و دیوان او مشهور است.",
    "فوتبال ورزشی گروهی با دو تیم است.",
]

retriever = build_retriever(texts, HashingEmbedder())


def answer(prompt: str) -> str:
    # Replace this with a call to your model.
    return "شیراز" if "شیراز" in prompt else "نمی‌دانم"


result = RAGPipeline(retriever, answer).query("حافظ در کدام شهر زاده شد؟")
print(result.answer)
for item in result.contexts:
    print(item.score, item.text)
```

The same flow is in [`examples/quickstart.py`](examples/quickstart.py).

Semantic embeddings:

```python
from persian_rag import SentenceTransformerEmbedder, build_retriever

embedder = SentenceTransformerEmbedder(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
retriever = build_retriever(texts, embedder)
```

That model is a multilingual default, not a recommendation for every corpus.

## Usage

### Normalization

```python
from persian_rag import NormalizeConfig, normalize

print(normalize("يك كتاب با اعداد ٤٥٦ و می رود..."))
# یک کتاب با اعداد ۴۵۶ و می‌رود...

plain = normalize("کِتاب", NormalizeConfig(remove_diacritics=True, digits="ascii"))
```

`normalize` is idempotent for a fixed config. Use the same `NormalizeConfig` for documents and queries. `build_retriever` does this for you.

| Option | Default | Effect |
| --- | --- | --- |
| `unify_arabic_chars` | `True` | `ي`/`ى` to `ی`, `ك` to `ک`, `أ`/`إ`/`ٱ` to `ا`, `ؤ` to `و`, `ة` to `ه`. `آ` and `ئ` stay. |
| `compatibility_forms` | `True` | NFKC, so Arabic presentation forms fold into base letters before the map above. |
| `zwnj` | `"normalize"` | Collapse repeated half-spaces. `"remove"` deletes them. `"space"` turns them into spaces. |
| `correct_half_spaces` | `True` | Join `می` / `نمی` / `بی` and `ها` / `های` / `هایی` / `تر` / `ترین` with ZWNJ when they are separated by a space or hyphen. |
| `digits` | `"persian"` | `"persian"`, `"ascii"`, or `"keep"`. Arabic-Indic digits are included. |
| `remove_diacritics` | `False` | Strip Arabic vowel marks and Qur'anic annotations when set. |
| `remove_tatweel` | `True` | Remove kashida (`ـ`). |
| `clean_punctuation` | `True` | Collapse repeats, drop decorative bullets, tidy spaces around punctuation. |
| `punctuation` | `"persian"` | Map `,;?%` to `،؛؟٪`, or the other way with `"ascii"`. `"keep"` leaves them. |

### Sentences, tokens, and chunks

```python
from persian_rag import ChunkingConfig, chunk_text, split_sentences, tokenize

tokenize("می‌روم به مدرسه.")
# ['می‌روم', 'به', 'مدرسه']

split_sentences("سلام. خوبی؟ نسخه 1.2 آماده است.")

chunks = chunk_text(
    text,
    ChunkingConfig(strategy="recursive", chunk_size=800, overlap=120),
    document_id="notes/hafez.txt",
)
```

`strategy` is `"fixed"`, `"sentence"`, or `"recursive"`.

- Fixed chunks are character windows. They snap to a space when one sits in the second half of the window, and they overlap by `overlap` characters.
- Sentence chunks pack sentences up to `chunk_size`. `sentence_overlap` repeats that many sentences. A sentence longer than `chunk_size` is split on characters.
- Recursive chunks split on paragraphs, lines, then Persian punctuation, then spaces, and only then on a hard character cut.

Chunk offsets refer to the string you passed in. `build_retriever` normalizes first, so stored offsets refer to the normalized text.

### Embeddings

```python
from persian_rag import OpenAICompatibleEmbedder

embedder = OpenAICompatibleEmbedder(
    model="text-embedding-3-small",
    dimensions=1536,
    base_url="https://api.openai.com/v1",
)
```

`Embedder` is a protocol with `dimension`, `embed_documents`, and `embed_query`. A custom class only needs those three.

### Vector stores

```python
from persian_rag import FaissVectorStore, InMemoryVectorStore

memory = InMemoryVectorStore()
faiss_store = FaissVectorStore(dimension=embedder.dimension, metric="cosine")
```

`InMemoryVectorStore` ranks by cosine similarity and has no extra dependency. `FaissVectorStore` needs the `faiss` extra. Its `cosine` metric normalizes vectors and uses inner product. Its `l2` metric returns the negative distance, so larger scores are closer in both cases. The FAISS adapter is add-only.

Pass `store=` to `build_retriever` or `HybridRetriever` to use either one.

### Hybrid retrieval

```python
from persian_rag import HybridRetriever

retriever = HybridRetriever(
    embedder,
    rrf_k=60,
    dense_weight=1.0,
    sparse_weight=1.0,
)
hits = retriever.retrieve("حافظ اهل کجا بود؟", k=5, mode="hybrid")
```

`mode` is `"hybrid"`, `"dense"`, or `"sparse"`. Hybrid scores are reciprocal rank fusion, not raw BM25 or cosine. Dense mode returns the store score. Sparse mode returns Okapi BM25 with the non-negative IDF `log(1 + (N - df + 0.5) / (df + 0.5))`.

`RetrievalResult.dense_rank` and `sparse_rank` are 1-based ranks inside the candidate pool (`k * candidate_multiplier`), or `None` when that list did not contain the chunk.

### RAG pipeline

```python
from persian_rag import DEFAULT_PROMPT_EN, RAGPipeline

pipeline = RAGPipeline(retriever, my_llm, prompt_template=DEFAULT_PROMPT_EN)
answer = pipeline.query("Where was Hafez born?", k=4)
```

The default prompt is Persian (`DEFAULT_PROMPT_FA`). Both templates contain `{context}` and `{question}`. Without an LLM, `query` still returns the contexts and the filled prompt, and `answer` is empty.

`make_openai_llm(model=..., api_key=...)` returns a callable for an OpenAI-compatible chat endpoint.

### Command line

```bash
persian-rag ingest ./docs --index .persian-rag
persian-rag query "حافظ اهل کجا بود؟" --index .persian-rag --json
```

`ingest` reads `.txt`, `.text`, `.md`, and `.markdown`, skips hidden files, and strips common Markdown. The default embedder is `hashing`. Choose another with `--embedder sentence-transformers` or `--embedder openai`.

```bash
persian-rag ingest ./docs --embedder sentence-transformers --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
OPENAI_API_KEY=... persian-rag query "پرسش" --llm openai --llm-model gpt-4o-mini
```

The index is a UTF-8 JSON file (`index.json` inside the directory you pass). It stores normalized chunks and vectors. It does not store API keys. Query-time embedding must use the same embedder settings.

Other flags worth knowing: `--strategy`, `--chunk-size`, `--overlap`, `--digits`, `--remove-diacritics`, `--zwnj`, `--mode`, `--prompt-lang fa|en`, `--show-prompt`. Run `persian-rag ingest --help` for the rest.

Files larger than 10 MiB are rejected.

## Development

```bash
pip install -e ".[dev]"
ruff check .
ruff format --check .
mypy src
pytest
```

Heavy sentence-transformer tests are skipped unless you set `PERSIAN_RAG_RUN_HEAVY=1` and install the extra. The FAISS test runs only when `faiss` and `numpy` are installed.

## Roadmap

0.1.0 is the retrieval core: normalize, chunk, embed, fuse, and call an LLM. Later versions can grow without changing that split.

Toward a pro edition:

- A reranker interface and a cross-encoder adapter.
- Optional morphological normalization for Persian clitics beyond the small affix list.
- Connectors for durable stores such as Qdrant, pgvector, and OpenSearch.
- Parsers for PDF and HTML, including documents where Persian reading order matters.
- Incremental indexing and deletion.
- An evaluation harness that can run on public Persian QA sets. Scores belong in that harness, not in the README.
- Citation-style answers and a check that the answer stays inside the retrieved text.
- A small HTTP service in front of `RAGPipeline`.

Nothing in that list is in this release.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Bugs and proposals can be filed in Persian or English.

## License

[MIT](LICENSE). Copyright (c) 2026 Naser Sadrmanesh.

## Author

Naser Sadrmanesh

- GitHub: [https://github.com/sadrmanesh](https://github.com/sadrmanesh)
- Site: [https://sadrmanesh.github.io](https://sadrmanesh.github.io)
- Email: [info.sadrmanesh@gmail.com](mailto:info.sadrmanesh@gmail.com)
