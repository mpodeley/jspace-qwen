"""Reconciliation with Christ et al. 2510.26543. Their 'add-N generalizes' works
because add-N operators form a LINEAR family (a number line): v(add-k) ~ k*d, so
the operator directions are collinear and you can interpolate an unseen N. Our
+/x/- operators are distinct functions -> not collinear -> don't generalize.

Measure, for each domain, how much of the operator-direction set lies on a single
line: the fraction of variance the top singular vector of the mean-centred
operator directions explains (1.0 = perfectly collinear)."""
import os, sys, json
from pathlib import Path
os.environ.setdefault("HF_HOME", "/var/home/matias/Projects/jspace-qwen/models")
sys.path.insert(0, "/var/home/matias/Projects/jspace-qwen/scripts")
import torch
import op_core
from _common import band_layers, evenly_spaced_layers, load_model, resolve_tag

MODEL_KEY = sys.argv[1] if len(sys.argv) > 1 else "1.7b"
model = load_model(MODEL_KEY)
ws = band_layers(evenly_spaced_layers(model.n_layers), model.n_layers)["workspace"]


def collinearity(domain):
    dom = op_core.load_domain(domain)
    v = op_core.op_dirs(model, ws, dom)
    keys = dom.op_keys
    # stack operator directions (concat over ws), mean-centre, SVD
    M = torch.stack([torch.cat([v[k][l] for l in ws]) for k in keys])  # [n_op, d]
    M = M - M.mean(0, keepdim=True)
    s = torch.linalg.svdvals(M.float())
    top1 = float(s[0] ** 2 / (s.pow(2).sum() + 1e-9))
    # consecutive-difference cosines (only meaningful if operators have an order)
    diffs = [torch.cat([v[keys[i + 1]][l] - v[keys[i]][l] for l in ws]) for i in range(len(keys) - 1)]
    coss = []
    for i in range(len(diffs) - 1):
        a, b = diffs[i], diffs[i + 1]
        coss.append(float((a @ b) / (a.norm() * b.norm() + 1e-9)))
    return keys, top1, coss


results = {}
for dom in ("arith_addN", "arithmetic", "relations"):
    keys, top1, coss = collinearity(dom)
    results[dom] = {"operators": keys, "top1_singular_variance": round(top1, 4),
                    "consecutive_difference_cosines": [round(c, 4) for c in coss]}
    print(f"\n{dom}  ({len(keys)} operators: {keys})")
    print(f"  top-1 singular variance of operator set: {top1:.3f}  (1.0 = collinear / on one line)")
    if coss:
        print(f"  consecutive-difference cosines: {[round(c,2) for c in coss]}")

out = (Path(__file__).resolve().parent.parent / "results" / "geometry"
       / f"{resolve_tag(MODEL_KEY)}_collinearity.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(results, indent=2))
print(f"\nsaved {out}")
