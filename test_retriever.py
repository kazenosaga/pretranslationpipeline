import json, numpy as np
from sentence_transformers import SentenceTransformer

IDX_DIR = "./resources/gloss_index"
terms = json.load(open(f"{IDX_DIR}/gloss_terms.json", encoding="utf-8"))
targets = json.load(open(f"{IDX_DIR}/gloss_targets.json", encoding="utf-8"))
meta = json.load(open(f"{IDX_DIR}/metadata.json", encoding="utf-8"))
emb = np.load(f"{IDX_DIR}/gloss_emb.npy")

embedder = SentenceTransformer(meta["model"])

def retrieve(text, top_k=8, min_sim=0.25):
    q = embedder.encode([text], normalize_embeddings=meta.get("normalize", True)).astype("float32")
    sims = (q @ emb.T).ravel()
    idxs = np.argsort(-sims)[:top_k]
    return [(terms[i], targets[i], float(sims[i])) for i in idxs if sims[i] >= min_sim]

tests = [
    "В курсе используется PostgreSQL версии 16.",
    "Преподавателю необходимо изучить документ.",
    "Рекомендательные блокировки в СУБД.",
    "Полная очистка помогает предотвратить разрастание таблиц.",
    "Для эффективной реализации необходимо выполнить денормализацию: добавить избыточное полев таблицу книг (обновляемое триггером), на которое можно наложить ограничение проверки. (К идее денормализации можно было прийти и другим путем: вычисление количества книг суммированием заведомо неэффективно при большом количестве операций."
]

for t in tests:
    print("\nTEXT:", t)
    for src, tgt, score in retrieve(t, top_k=8, min_sim=0.20):
        print(f"  - {src} => {tgt}   (sim={score:.3f})")
