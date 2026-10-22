"""Patient-cluster bootstrap 95% CIs for Table 2 (paired differences vs untuned JINAv3).

Uses the out-of-fold predictions saved by `python -m src.models.evaluation` (`*_oof.npz`).
Run from the repo root, e.g.:
  python scripts/paper/table2_ci.py --results results --tumor data/eval/tumor_type_eval.csv \
      --mgmt data/eval/mgmt_eval.csv --survival data/eval/os_eval.csv
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.stats import cluster_bootstrap_ci

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="results")
ap.add_argument("--tumor", default="data/eval/tumor_type_eval.csv")
ap.add_argument("--mgmt", default="data/eval/mgmt_eval.csv")
ap.add_argument("--survival", default="data/eval/os_eval.csv")
ap.add_argument("--baseline", default="jina_base_text_match")
ap.add_argument("--models", nargs="+", default=["mnrl_full_data", "nv_embed", "gatortron_large"])
a = ap.parse_args()

# Rows in the same order as evaluation.py keeps them (valid labels; 3-class tumor type)
tt = pd.read_csv(a.tumor); tt = tt[tt.class_label.notna() & (tt.class_label != 3)]
mg = pd.read_csv(a.mgmt); mg = mg[mg.mgmt.notna()]
sv = pd.read_csv(a.survival); sv = sv[sv.one_year_survival.notna()]
pid = lambda d: d[[c for c in ("MRN", "mrn") if c in d.columns][0]].astype(str).values
out = {}
for task, df in [("tumor_type", tt), ("mgmt", mg), ("survival", sv)]:
    load = lambda m: np.load(Path(a.results) / f"{task}_{m}_oof.npz")
    base = load(a.baseline); y = base["labels"].astype(int); g = pid(df)
    assert len(y) == len(g), f"{task}: {len(y)} predictions vs {len(g)} rows"
    for m in a.models:
        z = load(m)
        for metric in ("accuracy", "auroc"):
            r = cluster_bootstrap_ci(y, z["proba"], g, metric, prob_b=base["proba"])
            out[f"{task}|{m}-{a.baseline}|{metric}"] = r
            print(f"{task:10s} {m:22s} {metric:8s} diff {100*r['estimate']:+.1f} [{100*r['ci_low']:+.1f}, {100*r['ci_high']:+.1f}]")
json.dump(out, open(Path(a.results) / "table2_ci.json", "w"), indent=1)
