#!/bin/bash
# Finish PQuAD dense runs, then fusion, evaluation, statistics, figures and the source document.
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
for m in e5_base minilm; do $PY run_retrieval.py pquad dense $m; done
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
$PY run_retrieval.py pquad fusion
$PY evaluate.py pquad
$PY stats.py pquad ndcg10
$PY figures.py pquad
$PY build_source.py
echo PQUAD_DONE
