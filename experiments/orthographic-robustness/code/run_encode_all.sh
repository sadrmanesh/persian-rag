#!/bin/bash
# Sequential dense encoding of all collections (CPU).
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
for m in e5_small e5_base minilm bge_m3; do
  $PY encode_corpus.py gold $m
  $PY encode_corpus.py pquad $m
done
$PY encode_corpus.py miracl e5_small
$PY encode_corpus.py miracl minilm
$PY encode_corpus.py miracl bge_m3 sub30k
$PY encode_corpus.py miracl e5_base
