"""Encode passage collections with dense models (CPU, chunked, resumable).

Usage: python encode_corpus.py <collection> <model_key> [subset]
Author: Mohammad Naser Sadrmanesh
"""
import sys, time
import numpy as np, torch
from sentence_transformers import SentenceTransformer
from common import read_jsonl, DATA, CACHE

MODELS = {
    "e5_small": ("intfloat/multilingual-e5-small", "query: ", "passage: "),
    "e5_base": ("intfloat/multilingual-e5-base", "query: ", "passage: "),
    "minilm": ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "", ""),
    "bge_m3": ("BAAI/bge-m3", "", ""),
}
MAX_LEN = 256
CHUNK = 5000


def load_model(key):
    name, qp, pp = MODELS[key]
    torch.set_num_threads(8)
    m = SentenceTransformer(name, device="cpu")
    m.max_seq_length = MAX_LEN
    return m, qp, pp


def main(coll, key, subset=None):
    docs = read_jsonl(DATA / f"{coll}_corpus.jsonl")
    idx = np.arange(len(docs))
    if subset:
        idx = np.load(DATA / f"{coll}_{subset}_idx.npy")
    out = CACHE / "emb" / f"{coll}{'_'+subset if subset else ''}_{key}"
    out.mkdir(parents=True, exist_ok=True)
    m, qp, pp = load_model(key)
    for c0 in range(0, len(idx), CHUNK):
        f = out / f"chunk_{c0:07d}.npy"
        if f.exists():
            continue
        t = time.time()
        texts = [pp + docs[i]["text"] for i in idx[c0:c0 + CHUNK]]
        e = m.encode(texts, batch_size=32, normalize_embeddings=True, convert_to_numpy=True)
        np.save(f, e.astype(np.float16))
        print(f"{coll} {key} {c0+len(texts)}/{len(idx)} {len(texts)/(time.time()-t):.1f}/s", flush=True)
    np.save(out / "idx.npy", idx)
    print("done", coll, key, flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
