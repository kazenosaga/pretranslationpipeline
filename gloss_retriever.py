# gloss_retriever.py
# Lightweight embedding-based glossary retriever with optional CSV fallback.

import os, json, numpy as np
from typing import List, Tuple

__TERMS: list | None = None
__TARGETS: list | None = None
__EMB: np.ndarray | None = None
__META: dict | None = None
__EMBEDDER = None
__HAVE_INDEX = False

def init(index_dir: str = "./resources/gloss_index") -> bool:

    global __TERMS, __TARGETS, __EMB, __META, __EMBEDDER, __HAVE_INDEX
    terms_p   = os.path.join(index_dir, "gloss_terms.json")
    targets_p = os.path.join(index_dir, "gloss_targets.json")
    emb_p     = os.path.join(index_dir, "gloss_emb.npy")
    meta_p    = os.path.join(index_dir, "metadata.json")

    if not (os.path.exists(terms_p) and os.path.exists(targets_p)
            and os.path.exists(emb_p) and os.path.exists(meta_p)):
        __HAVE_INDEX = False
        return False

    with open(terms_p, "r", encoding="utf-8") as f:
        __TERMS = json.load(f)
    with open(targets_p, "r", encoding="utf-8") as f:
        __TARGETS = json.load(f)
    __EMB = np.load(emb_p).astype("float32")
    with open(meta_p, "r", encoding="utf-8") as f:
        __META = json.load(f)

    from sentence_transformers import SentenceTransformer
    __EMBEDDER = SentenceTransformer(__META["model"])
    __HAVE_INDEX = True
    return True

def have_index() -> bool:
    return bool(__HAVE_INDEX)

def retrieve(text: str, top_k: int = 8, min_sim: float = 0.25) -> List[Tuple[str, str]]:

    if not __HAVE_INDEX or __EMBEDDER is None or __EMB is None:
        return []
    norm = bool(__META.get("normalize", True))
    q = __EMBEDDER.encode([text], normalize_embeddings=norm)
    q = np.asarray(q, dtype="float32")  # (1, D)
    sims = (q @ __EMB.T).ravel()
    idxs = np.argsort(-sims)[:top_k]
    out = []
    for i in idxs:
        if float(sims[i]) >= min_sim:
            out.append((__TERMS[i], __TARGETS[i]))
    return out
