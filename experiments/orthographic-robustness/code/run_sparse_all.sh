#!/bin/bash
# Lexical systems (BM25 x4 tokenizers, rewrite, hashing) on all collections.
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for c in miracl30k miracl; do $PY run_retrieval.py $c sparse; done
