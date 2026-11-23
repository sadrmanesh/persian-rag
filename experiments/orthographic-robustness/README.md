# Orthographic robustness experiments

Retrieval experiments on Persian queries under controlled orthographic noise: Arabic-keyboard letters (`T_char`), ZWNJ deletion or spacing (`T_zwnj`), keyboard and homophone edits (`T_typo`), their composition (`T_comp`), and variants attested in the MIRACL-fa passage pool (`T_real`). Systems are BM25 with four tokenizers, persian-rag query rewriting, a character n-gram hashing embedder, dense encoders (mE5-small, mE5-base, BGE-M3, MiniLM-L12), and reciprocal rank fusion. The recorded summary tables and figures in this folder are the published run. Library code under `src/` is not part of this pipeline.

## Environment

Python 3.10 or newer. The recorded gold hashing cross-check used persian-rag 0.5.0 and Python 3.13.5. Dense encoding runs on CPU.

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[sentence-transformers]"
pip install -r experiments/orthographic-robustness/requirements.txt
```

`hazm` and `parsivar` are used only by the BM25 tokenizer variants. `sentence-transformers` and `torch` are used by the dense encoders. The shell scripts call `${PYTHON:-python3}`.

Persian labels in `diagrams.py` use Vazirmatn. Put the `Vazirmatn-*.ttf` files in `experiments/orthographic-robustness/fonts/`, or set `VAZIRMATN_DIR` to that directory. Figures 1, 2, 4, 5, and 6 do not need the font.

## Seed

`SEED = 42` in `code/common.py`. Each perturbation draws from `SHA-256(42 | query id | operator | edit rate)`, so query order does not matter. The MIRACL passage sample uses `random.Random(42)`. Bootstrap intervals and sign-flip tests use `numpy.random.default_rng(42)` with 10,000 resamples.

## Data

Do not commit `data/` or `cache_store/`. Create the data directory from `code/`:

```bash
cd experiments/orthographic-robustness
mkdir -p data
```

### Gold (persian-rag-bench)

34 passages and 42 queries, already in this repository (also on the Hugging Face Hub as `sadrmanesh/persian-rag-bench`).

```bash
cp ../../benchmarks/data/corpus.jsonl data/gold_corpus.jsonl
cp ../../benchmarks/data/qa.jsonl data/gold_qa.jsonl
```

### MIRACL-fa dev

632 development queries. `prepare_data.py` keeps every passage judged in the dev qrels and adds a seed-42 sample of the remaining Persian corpus up to 100,000 passages (pass a different total as the first argument). The five corpus shards must be named `fa-docs-0.jsonl.gz` through `fa-docs-4.jsonl.gz`. Upstream they are `docs-0.jsonl.gz` through `docs-4.jsonl.gz` (about 270 MB compressed).

```bash
BASE=https://huggingface.co/datasets/miracl/miracl/resolve/main/miracl-v1.0-fa
curl -L -o data/topics.miracl-v1.0-fa-dev.tsv \
  "$BASE/topics/topics.miracl-v1.0-fa-dev.tsv"
curl -L -o data/qrels.miracl-v1.0-fa-dev.tsv \
  "$BASE/qrels/qrels.miracl-v1.0-fa-dev.tsv"

CORPUS=https://huggingface.co/datasets/miracl/miracl-corpus/resolve/main/miracl-corpus-v1.0-fa
for i in 0 1 2 3 4; do
  curl -L -o "data/fa-docs-${i}.jsonl.gz" "$CORPUS/docs-${i}.jsonl.gz"
done
```

Topics and qrels are TSV. Each corpus line is JSON with `docid`, `title`, and `text`. Licence: Apache-2.0 (MIRACL).

### PQuAD validation

1,103 validation paragraphs and the answerable questions (5,995). The loader reads one parquet row per article, with a `paragraphs` list of `{context, qas}` objects, matching the official JSON.

```bash
curl -L -o /tmp/PQuAD_Validation.json \
  https://raw.githubusercontent.com/AUT-NLP/PQuAD/main/Dataset/Validation.json
python - <<'PY'
import json
import pandas as pd
raw = json.load(open("/tmp/PQuAD_Validation.json", encoding="utf-8"))
pd.DataFrame(raw["data"]).to_parquet("data/pquad_validation.parquet", index=False)
PY
```

Check the dataset card for the PQuAD licence before redistributing the file.

### Nested 30k index

`miracl30k` is a nested subset of the 100k pool. Retrieval and `encode_corpus.py ... sub30k` read `data/miracl_sub30k_idx.npy` (integer positions into `data/miracl_corpus.jsonl`). None of these scripts writes that file. Gold, PQuAD, and the 100k MIRACL runs do not need it. The `results/miracl30k_*` tables and `figures/*_miracl30k.*` files are from the recorded 30k pool; reproducing them requires the same index.

## Run order

Run from `experiments/orthographic-robustness/code` so imports resolve. Prepared collections, perturbed queries, and the attested-variant list land in `../data`. Embeddings and runs land in `../cache_store`. Metrics, statistics, and figures land in `../results` and `../figures`.

```bash
cd experiments/orthographic-robustness/code

python prepare_data.py
python mine_variants.py
python make_queries.py

python run_retrieval.py gold sparse
python run_retrieval.py pquad sparse
python run_retrieval.py miracl sparse
python run_retrieval.py miracl30k sparse

python encode_corpus.py gold e5_small
python encode_corpus.py gold e5_base
python encode_corpus.py gold minilm
python encode_corpus.py gold bge_m3
python encode_corpus.py pquad e5_small
python encode_corpus.py pquad e5_base
python encode_corpus.py pquad minilm
python run_retrieval.py gold dense e5_small
python run_retrieval.py gold dense e5_base
python run_retrieval.py gold dense minilm
python run_retrieval.py gold dense bge_m3
python run_retrieval.py pquad dense e5_small
python run_retrieval.py pquad dense e5_base
python run_retrieval.py pquad dense minilm
python run_retrieval.py miracl dense e5_small
python run_retrieval.py miracl dense minilm

python encode_corpus.py miracl e5_small sub30k
python encode_corpus.py miracl minilm sub30k
python encode_corpus.py miracl e5_base sub30k
python encode_corpus.py miracl bge_m3 sub30k
python run_retrieval.py miracl30k dense e5_small
python run_retrieval.py miracl30k dense minilm
python run_retrieval.py miracl30k dense e5_base
python run_retrieval.py miracl30k dense bge_m3

python run_retrieval.py gold fusion
python run_retrieval.py pquad fusion
python run_retrieval.py miracl fusion
python run_retrieval.py miracl30k fusion

python evaluate.py gold pquad miracl miracl30k
python stats.py gold ndcg10
python stats.py pquad ndcg10
python stats.py miracl ndcg10
python stats.py miracl30k ndcg10
python stats.py miracl30k recall100
python gold_stats.py
python figures.py gold pquad miracl miracl30k
python diagrams.py
python make_tables.py gold pquad miracl miracl30k
```

`make_tables.py` prints Markdown tables and does not write a file. `build_source.py` assembles a manuscript from `source_parts/` and `refs/`, which are not in this folder; skip it unless those inputs are present.

PQuAD dense runs keep the main conditions (edit rate 0.15) and the attested-variant conditions. The dose-response grid is stored for MIRACL and, for sparse systems, PQuAD.

### Shell scripts

Same directory. `PYTHON` selects the interpreter.

| Script | What it runs |
| --- | --- |
| `run_sparse_all.sh` | Sparse retrieval for `miracl30k`, then `miracl`. It does not run gold or PQuAD. |
| `run_encode_all.sh` | Passage encoding: gold and PQuAD for all four models; MIRACL-100k for mE5-small and MiniLM; MIRACL `bge_m3` on `sub30k`; MIRACL `e5_base` on the full 100k pool. |
| `run_dense_all.sh` | Waits until `cache_store/emb/miracl_sub30k_bge_m3/idx.npy` exists, stops other `run_encode_all.sh` / `encode_corpus.py miracl e5_base` processes, deletes `cache_store/emb/miracl_e5_base`, encodes mE5-base on the 30k subset, then dense retrieval, fusion, evaluation, statistics, and figures for the collections in the published tables. |
| `run_pquad_finish.sh` | PQuAD dense runs for mE5-base and MiniLM, then PQuAD fusion, evaluation, statistics, figures, and `build_source.py`. |

`run_dense_all.sh` is a batch driver for a machine that is already encoding. The Python commands above are the sequence that matches the published system set without stopping other processes.

## Outputs

| Path | Contents |
| --- | --- |
| `data/*_corpus.jsonl`, `data/*_queries.jsonl`, `data/*_queries_perturbed.jsonl` | Pools, clean queries, perturbed queries |
| `data/attested_variants.json` | Variant list from `mine_variants.py` |
| `cache_store/emb/` | Float16 passage-embedding chunks |
| `cache_store/qemb/` | Query-embedding caches |
| `cache_store/runs/<collection>/*.pkl` | Top-100 runs |
| `results/<collection>_per_query.csv.gz` | Per-query metrics (gitignored; not in the recorded snapshot) |
| `results/<collection>_summary.csv` | Mean metrics, deltas, retention |
| `results/<collection>_stats_<metric>.csv` | Paired tests and Holm adjustment |
| `results/gold_significance.json` | McNemar and MRR tests on the 42 gold queries |
| `results/attested_variants_stats.json` | Variant counts and natural query noise |
| `results/gold_library_benchmark_hashing.json` | Hashing-embedder run of `benchmarks/run_benchmark.py` on the gold set |
| `figures/fig*.png`, `figures/fig*.pdf` | 300 dpi PNG and vector PDF |
| `figures/captions.md` | Figure captions |
| `revision_source.md` | Only if `build_source.py` is run |
