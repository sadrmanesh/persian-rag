# Benchmarks

New in 0.3.0. The repository ships a small Persian retrieval benchmark in `benchmarks/`:

* `data/corpus.jsonl`: 34 short documents on Persian literature, history, geography, culture, and everyday topics. Several documents are distractors that share words with other documents.
* `data/qa.jsonl`: 42 questions with the ids of the relevant documents. Twelve are tagged `spelling`: they use a different accepted spelling from the corpus (اطاق for اتاق, طهران for تهران, هیات for هیئت, باطری for باتری, بلیط for بلیت, مساله for مسئله, آئین for آیین, پائیز for پاییز, رییس for رئیس).
* `run_benchmark.py`: builds an index and scores every configuration.

```bash
python benchmarks/run_benchmark.py                     # offline, hashing embedder
python benchmarks/run_benchmark.py --json results.json
python benchmarks/run_benchmark.py --embedder sentence-transformers --model intfloat/multilingual-e5-small
```

The corpus and questions are also published on the Hugging Face Hub as [sadrmanesh/persian-rag-bench](https://huggingface.co/datasets/sadrmanesh/persian-rag-bench), and the [live demo](https://huggingface.co/spaces/sadrmanesh/persian-rag-demo) searches the same corpus:

```python
from datasets import load_dataset

corpus = load_dataset("sadrmanesh/persian-rag-bench", "corpus", split="test")
qa = load_dataset("sadrmanesh/persian-rag-bench", "qa", split="test")
```

## Results

persian-rag 0.3.0, Python 3.12, hashing embedder (512 dimensions, character 2 to 4 grams), chunks of 400 characters, `k=10`. Hit@k is the share of questions with a relevant document in the top k; MRR and nDCG use document matching. Latency is one `retrieve` call on an 8-core Linux machine, median and 95th percentile over five rounds.

| Configuration | Hit@1 | Hit@5 | MRR@10 | nDCG@10 | Hit@1 (spelling) | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| sparse | 0.881 | 0.881 | 0.881 | 0.879 | 0.583 | 0.05 | 0.07 |
| dense | 0.952 | 0.976 | 0.957 | 0.962 | 0.917 | 0.64 | 0.73 |
| hybrid | 0.952 | 0.976 | 0.964 | 0.967 | 0.833 | 0.68 | 0.75 |
| sparse + rewrite | 0.952 | 0.976 | 0.964 | 0.967 | 0.833 | 0.21 | 0.46 |
| dense + rewrite | 0.952 | 0.976 | 0.957 | 0.962 | 0.917 | 1.49 | 2.91 |
| hybrid + rewrite | 0.952 | 0.976 | 0.964 | 0.967 | 0.833 | 1.55 | 3.10 |

## Embedding models

New in 0.4.0. The default `sentence-transformers` model changed from `paraphrase-multilingual-MiniLM-L12-v2` to `intfloat/multilingual-e5-small`, with E5 `query: `/`passage: ` prefixes. Both models have 118M parameters and 384 dimensions. nDCG@10 on CPU, sentence-transformers 6.2.0, normalized cosine similarity, queries and passages normalized with `persian_rag.normalize()`, full-corpus ranking:

| Model | MIRACL fa dev | PQuAD validation | persian-rag-bench |
|---|---:|---:|---:|
| paraphrase-multilingual-MiniLM-L12-v2 (old default) | 0.4424 | 0.6437 | 0.8637 |
| multilingual-e5-small with prefixes (new default) | 0.7461 | 0.9242 | 0.9881 |
| multilingual-e5-base with prefixes (`e5-base`) | **0.7657** | **0.9247** | **1.0000** |

| Model | Set | MRR@10 | Recall@10 |
|---|---|---:|---:|
| MiniLM-L12-v2 | MIRACL fa dev | 0.4271 | 0.6356 |
| e5-small | MIRACL fa dev | 0.7138 | 0.9521 |
| MiniLM-L12-v2 | PQuAD validation | 0.5905 | 0.8113 |
| e5-small | PQuAD validation | 0.9042 | 0.9845 |
| e5-base | MIRACL fa dev | 0.7369 | 0.9590 |
| e5-base | PQuAD validation | 0.9043 | 0.9858 |

### e5-small or e5-base

`intfloat/multilingual-e5-base` is supported as the more accurate option (new in the next release). Use `SentenceTransformerEmbedder("e5-base")` or `--model e5-base`; the full name works too, and the E5 prefixes are added automatically.

| | e5-small (default) | e5-base |
|---|---:|---:|
| Parameters | 118M | 278M |
| Dimensions (index size) | 384 | 768 (2x) |
| Model download | about 470 MB | about 1.1 GB |
| CPU encoding speed, PQuAD paragraphs | 54 per second | 22 per second |
| MIRACL fa dev, normalized | 0.7461 | **0.7657** |
| MIRACL fa dev, raw | 0.7448 | **0.7653** |
| MIRACL fa dev, perturbed queries | 0.6645 | **0.7089** |
| PQuAD validation, normalized | 0.9242 | **0.9247** |
| PQuAD validation, perturbed queries | 0.8639 | **0.8906** |
| persian-rag-bench | 0.9881 | **1.0000** |
| persian-rag-bench, spelling Hit@1 | 0.9167 | **1.0000** |

Perturbed queries are written with Arabic yeh and kaf and with spaces instead of half spaces, and neither side is normalized; this is what text typed on an Arabic keyboard looks like. `persian_rag.normalize()` removes most of that difference, so the normalized rows are the ones to compare for a normal setup. The gain from e5-base is small on clean PQuAD questions and largest on MIRACL and on noisy queries. e5-base is a good choice when accuracy matters more than indexing time and memory; e5-small stays the default because it is much faster on CPU and close on clean text. Vectors from the two models are not compatible: switching models means rebuilding the index, and loading an e5-small index with e5-base is refused.

The e5-base rows were measured with the same evaluation script on a Colab T4 GPU. e5-small gives identical scores on CPU and GPU, and the e5-base MIRACL and persian-rag-bench scores were reproduced on CPU (the bench with `run_benchmark.py --model e5-base`). Speed was measured with sentence-transformers 6.2.0 on 8 Xeon cores, 512 PQuAD paragraphs with prefixes, batches of 32, at most 512 tokens. A single short query takes about 6 ms with e5-small and 13 ms with e5-base.

* **MIRACL fa dev**: 632 queries ranked against a pooled corpus of the 6,405 judged passages for those queries, not the full 2.2M-passage MIRACL corpus. The task is easier than the official one, so these numbers are not comparable with published MIRACL scores. They are useful for comparing models with each other.
* **PQuAD validation**: 5,995 answerable questions ranked against the 1,103 unique paragraphs of the split.
* **persian-rag-bench**: the 34 documents and 42 questions above, dense retrieval only.
* The prefixes matter: on persian-rag-bench e5-small without prefixes scores 0.9824 nDCG@10 and 0.8333 Hit@1 on the spelling subset, against 0.9881 and 0.9167 with them.

Reproduce the persian-rag-bench rows with `run_benchmark.py` (the `dense` row):

```bash
pip install "persian-rag[sentence-transformers]"
python benchmarks/run_benchmark.py --embedder sentence-transformers
python benchmarks/run_benchmark.py --embedder sentence-transformers \
  --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
python benchmarks/run_benchmark.py --embedder sentence-transformers --model e5-base
```

## Reading the numbers

* BM25 alone misses five of the twelve misspelled queries, because a different spelling is a different token. The [query rewriter](query-rewriting.md) recovers most of them: sparse Hit@1 on the spelling subset goes from 0.583 to 0.833, and overall Hit@1 from 0.881 to 0.952.
* The hashing embedder compares character n-grams, so it is naturally tolerant of one-letter spelling differences. That is why dense retrieval does well on the spelling subset even without rewriting. A semantic embedding model behaves differently; rerun the script with your model.
* Rewriting multiplies the retrieval calls by the number of variants, which shows in the latency columns. It stays in the low milliseconds for in-memory indexes.
* The set is small and the scores are high on purpose: it is a regression check for Persian-specific behaviour, not a leaderboard. Extend `qa.jsonl` with questions from your own domain for decisions that matter.

The benchmark runs in the test suite on every change, so the script and data stay valid.
