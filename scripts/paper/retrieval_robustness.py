"""Retrieval robustness checks on the held-out patients (no training).

Reports standard same-patient retrieval (Table 1), retrieval restricted to same-histology
candidates, a BM25 keyword baseline, top-1 histology concordance, and patient-cluster
bootstrap CIs. Runs on CPU. Run from the repo root:
  EVAL=data/training/mnrl/eval.csv CKPT=models/mnrl/final/model.safetensors OUT=results/retrieval \
      python scripts/paper/retrieval_robustness.py
"""
import os, sys, json, time, numpy as np, pandas as pd, torch, re
sys.modules['flash_attn'] = None  # CPU run: Jina falls back to its pure-PyTorch path
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
import transformers
_orig = transformers.AutoModel.from_pretrained
def _no_flash(*a, **k):  # CPU run: standard attention instead of flash/triton kernels (same weights)
    k.setdefault("use_flash_attn", False); return _orig(*a, **k)
transformers.AutoModel.from_pretrained = _no_flash
import src.models.alignment as _al
_al.AutoModel.from_pretrained = _no_flash
from src.models.alignment import _load_jina_model, _embed_texts, _compute_retrieval_metrics
torch.set_num_threads(int(os.environ.get("NT", "48")))
OUT = os.environ.get("OUT", "results/retrieval"); os.makedirs(OUT, exist_ok=True)
LIMIT = int(os.environ.get("LIMIT", "0"))
df = pd.read_csv(os.environ.get("EVAL", "data/training/mnrl/eval.csv"))
rad = df[["anchor", "mrn", "cancer_type"]].drop_duplicates(subset="anchor").reset_index(drop=True)
pat = df[["positive", "mrn", "cancer_type"]].drop_duplicates(subset="positive").reset_index(drop=True)
if LIMIT: rad, pat = rad.head(LIMIT), pat.head(LIMIT)
print("queries", len(rad), "gallery", len(pat), flush=True)
models = {"jina_base": None, "gem_mnrl_full_data": os.environ.get("CKPT", "models/mnrl/final/model.safetensors")}
emb = {}
for name, ckpt in models.items():
    f = f"{OUT}/{name}_{'test' if LIMIT else 'full'}.npz"
    if os.path.exists(f):
        z = np.load(f); emb[name] = (z["rad"], z["pat"]); continue
    t = time.time(); m = _load_jina_model(checkpoint_path=ckpt, device="cpu")
    r = _embed_texts(m, rad["anchor"].tolist(), batch_size=16, max_length=2048, device="cpu")
    p = _embed_texts(m, pat["positive"].tolist(), batch_size=16, max_length=2048, device="cpu")
    np.savez(f, rad=r, pat=p); emb[name] = (r, p); print(name, "embedded in %.0fs" % (time.time() - t), flush=True)
    del m
qm, gm = rad["mrn"].values, pat["mrn"].values
qt, gt = rad["cancer_type"].values, pat["cancer_type"].values

def bm25_sim(queries, docs, k1=1.5, b=0.75):
    from sklearn.feature_extraction.text import CountVectorizer
    v = CountVectorizer(lowercase=True, token_pattern=r"(?u)\b[a-z][a-z0-9]+\b", stop_words="english")
    D = v.fit_transform(docs).tocsc().astype(float); Q = (v.transform(queries) > 0).astype(float)
    N = D.shape[0]; df_ = np.asarray((D > 0).sum(0)).ravel(); idf = np.log((N - df_ + 0.5) / (df_ + 0.5) + 1)
    dl = np.asarray(D.sum(1)).ravel(); avg = dl.mean()
    D = D.tocoo(); denom = D.data + k1 * (1 - b + b * dl[D.row] / avg)
    import scipy.sparse as sp
    W = sp.csr_matrix((D.data * (k1 + 1) / denom * idf[D.col], (D.row, D.col)), shape=D.shape)
    return (Q @ W.T).toarray()

def restricted(sim):  # gallery limited to same histology as the query
    s = sim.copy(); s[qt[:, None] != gt[None, :]] = -np.inf; return s

def r1_hits(sim):
    return (gm[np.argmax(sim, 1)] == qm).astype(float)

def cluster_ci(hits_a, hits_b=None, n=2000, seed=7):
    rng = np.random.default_rng(seed); pts = np.unique(qm); idx = {p: np.where(qm == p)[0] for p in pts}
    est = []
    for _ in range(n):
        s = np.concatenate([idx[p] for p in rng.choice(pts, len(pts))])
        est.append(hits_a[s].mean() - (hits_b[s].mean() if hits_b is not None else 0))
    return [float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5))]

sims = {}
for name, (r, p) in emb.items():
    r = r / np.linalg.norm(r, axis=1, keepdims=True); p = p / np.linalg.norm(p, axis=1, keepdims=True); sims[name] = r @ p.T
sims["bm25"] = bm25_sim(rad["anchor"].tolist(), pat["positive"].tolist())
res = {}
for name, s in sims.items():
    m = _compute_retrieval_metrics(s, qm, gm)
    mr = _compute_retrieval_metrics(restricted(s), qm, gm)
    h = r1_hits(s); hr = r1_hits(restricted(s))
    res[name] = {"standard": m, "same_histology": mr, "R@1_ci": cluster_ci(h), "R@1_same_hist_ci": cluster_ci(hr),
                 "top1_histology_concordance": float((gt[np.argmax(s, 1)] == qt).mean())}
    print(name, "R@1 %.3f %s | same-hist R@1 %.3f | MRR %.3f | top1 histology concordance %.3f" % (
        m["R@1"], np.round(res[name]["R@1_ci"], 3), mr["R@1"], m["MRR"], res[name]["top1_histology_concordance"]), flush=True)
h_g, h_j = r1_hits(sims["gem_mnrl_full_data"]), r1_hits(sims["jina_base"])
res["diff_gem_minus_jina_R@1_ci"] = cluster_ci(h_g, h_j)
res["diff_gem_minus_jina_sameHist_R@1_ci"] = cluster_ci(r1_hits(restricted(sims["gem_mnrl_full_data"])), r1_hits(restricted(sims["jina_base"])))
res["diff_gem_minus_bm25_R@1_ci"] = cluster_ci(h_g, r1_hits(sims["bm25"]))
# chance level for same-histology restriction: mean over queries of (#same-patient same-hist gallery)/(#same-hist gallery)
cand = np.array([(gt == t).sum() for t in qt]); rel = np.array([((gt == t) & (gm == m)).sum() for t, m in zip(qt, qm)])
res["same_hist_chance_R@1"] = float(np.mean(rel / np.maximum(cand, 1)))
print({k: v for k, v in res.items() if k.startswith(("diff", "same_hist_chance"))}, flush=True)
json.dump(res, open(f"{OUT}/metrics{'_test' if LIMIT else ''}.json", "w"), indent=1, default=float)
