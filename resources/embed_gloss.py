#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
from datetime import datetime
from dateutil.tz import tzlocal

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main():
    ap = argparse.ArgumentParser(
        description="Build an embedding index for a translation glossary (source,target CSV)."
    )
    ap.add_argument("--csv", default="gloss.csv", help="Path to glossary CSV (default: gloss.csv)")
    ap.add_argument("--out-dir", default="gloss_index", help="Output directory (default: gloss_index)")
    ap.add_argument(
        "--model",
        default="intfloat/multilingual-e5-small",
        help="Sentence-embeddings model name (default: intfloat/multilingual-e5-small)",
    )
    ap.add_argument(
        "--normalize",
        action="store_true",
        default=True,
        help="L2-normalize embeddings for cosine similarity (default: on)",
    )
    ap.add_argument("--no-normalize", dest="normalize", action="store_false")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load CSV (robust to UTF-8 BOM, auto-detect delimiter)
    with open(args.csv, "rb") as f:
        csv_bytes = f.read()
    csv_sha = sha256_bytes(csv_bytes)

    df = pd.read_csv(args.csv, encoding="utf-8-sig")
    # Basic validation
    expected_cols = {"source", "target"}
    if not expected_cols.issubset({c.strip().lower() for c in df.columns}):
        raise SystemExit(
            f"CSV must have columns: source,target (found: {list(df.columns)})"
        )

    # Normalize column names just in case
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.fillna("")

    # Drop empty sources; de-duplicate by source (keep first)
    df = df[df["source"].str.strip() != ""].copy()
    df = df.drop_duplicates(subset=["source"], keep="first").reset_index(drop=True)

    if df.empty:
        raise SystemExit("No valid rows: 'source' column is empty after cleaning.")

    terms = df["source"].astype(str).tolist()
    targets = df["target"].astype(str).tolist()

    print(f"Loaded {len(terms)} glossary entries from {args.csv}")

    # Load embedder
    print(f"Loading embedder: {args.model}")
    embedder = SentenceTransformer(args.model)

    # Encode (batching handled by SentenceTransformer)
    print("Encoding terms...")
    emb = embedder.encode(
        terms,
        normalize_embeddings=args.normalize,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype("float32")

    # Save artifacts
    terms_path = os.path.join(args.out_dir, "gloss_terms.json")
    targets_path = os.path.join(args.out_dir, "gloss_targets.json")
    emb_path = os.path.join(args.out_dir, "gloss_emb.npy")
    meta_path = os.path.join(args.out_dir, "metadata.json")

    with open(terms_path, "w", encoding="utf-8") as f:
        json.dump(terms, f, ensure_ascii=False, indent=2)
    with open(targets_path, "w", encoding="utf-8") as f:
        json.dump(targets, f, ensure_ascii=False, indent=2)
    np.save(emb_path, emb)

    metadata = {
        "model": args.model,
        "normalize": args.normalize,
        "csv_path": os.path.abspath(args.csv),
        "csv_sha256": csv_sha,
        "count": int(len(terms)),
        "dim": int(emb.shape[1]),
        "created_at": datetime.now(tzlocal()).isoformat(),
        "notes": "Embeddings align 1:1 with gloss_terms.json and gloss_targets.json",
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Saved:\n  {terms_path}\n  {targets_path}\n  {emb_path}\n  {meta_path}")
    print("Done.")


if __name__ == "__main__":
    main()