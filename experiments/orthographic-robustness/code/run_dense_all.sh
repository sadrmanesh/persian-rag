#!/bin/bash
# Dense retrieval, fusion, evaluation, statistics and figures (CPU budgeted).
# MIRACL-100k: mE5-small, MiniLM. MIRACL-30k (nested): mE5-small, MiniLM, mE5-base, BGE-M3.
# PQuAD: mE5-small, MiniLM, mE5-base. Gold: all four.
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
E=../cache_store/emb
while [ ! -f $E/miracl_sub30k_bge_m3/idx.npy ]; do sleep 20; done
pkill -f run_encode_all.sh; pkill -f "encode_corpus.py miracl e5_base"; sleep 2
rm -rf $E/miracl_e5_base
$PY encode_corpus.py miracl e5_base sub30k
for m in e5_small bge_m3 e5_base minilm; do $PY run_retrieval.py gold dense $m; done
for m in e5_small bge_m3 e5_base minilm; do $PY run_retrieval.py miracl30k dense $m; done
for m in e5_small minilm; do $PY run_retrieval.py miracl dense $m; done
for m in e5_small e5_base minilm; do $PY run_retrieval.py pquad dense $m; done
for c in gold pquad miracl miracl30k; do $PY run_retrieval.py $c fusion; done
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for c in gold pquad miracl miracl30k; do $PY evaluate.py $c; $PY stats.py $c ndcg10; done
$PY stats.py miracl30k recall100
$PY gold_stats.py
$PY figures.py gold pquad miracl miracl30k
echo ALL_DENSE_DONE
