"""OPMU multi-utility Pareto evaluation (paper Sec. 7, Table "pareto").

    python pareto_eval.py --rule linear|chebyshev --pred oracle|multi|direct [--kinds named,corner,dirichlet]

For every weight vector w in configs/scenarios.json:
  u_w(p,a) = w . z(p,a), with z standardised by configs/utility_basis.json
  single-arm points = (cost, mean utility) -> upper hull h(b); routed points come from sweeping the decision
  rule, merged with single-arm points -> upper hull f(b)
  metrics: gain = mean_b[f(b) - h(b)] over 8 budgets (same as the harness)
           HV   = hypervolume in the normalised (utility, -cost) plane (single-arm hull HV as reference)
           CPT90 = cost to reach 90% of the best single arm's utility / that arm's cost
           dominate = f >= h at all 8 budgets
Decision rules: linear = argmax(u - lambda c); chebyshev = argmin_a max{wu (1 - u_hat), (1 - wu) c_hat}
Predictors: oracle = true u (upper bound); multi = OPMU, multi-output ridge on the 9 dimensions, scalarised
afterwards; direct = one ridge per w on u_w (per-scenario router)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness"))
import protocol as P   # noqa: E402

N_B = 8
LAM = np.concatenate([[0], np.logspace(-3, 2, 160)])
WU = np.linspace(0.01, 0.99, 99)


def load(data_dir=ROOT / "data" / "gen", configs=ROOT / "configs"):
    basis = json.loads((Path(configs) / "utility_basis.json").read_text())
    sc = json.loads((Path(configs) / "scenarios.json").read_text())
    data = P.load(data_dir)
    lab = pd.read_csv(Path(data_dir) / "labels.csv")
    dims = basis["dims"]
    arms = list(data.arms[data.arms.heldout == 0].arm)
    src = lab[lab.seed == 0]
    piv = {d: src.pivot_table(index="prompt_id", columns="arm", values=d, aggfunc="mean") for d in dims}
    ok = piv[dims[0]].reindex(columns=arms).dropna().index
    pid = pd.Index(data.prompts.prompt_id)
    keep = pid.get_indexer(ok)
    split = data.prompts.split.to_numpy()[keep]
    Z = np.stack([((piv[d].loc[ok, arms].to_numpy() - basis["mu"][d]) / max(basis["sd"][d], 1e-9))
                  for d in dims])                                    # (D, P, A)
    c = data.arms.set_index("arm").usd_per_clip.loc[arms].to_numpy()
    X = P.default_X(data.feats, slice(None))[keep]
    return basis, sc, dims, arms, Z, c, X, split, ok


def hv(cost, util, cmax, ref=0.0):
    """Hypervolume in the normalised plane: step integral of the improvement over ref, sorted by cost."""
    o = np.argsort(cost)
    cs, us = cost[o] / cmax, util[o]
    best, area, prev = ref, 0.0, 0.0
    for x, u in zip(cs, us):
        area += max(best - ref, 0) * (x - prev)
        best = max(best, u)
        prev = x
    return area + max(best - ref, 0) * (1.0 - prev)


def route_points(U, Y, c, rule):
    """Routed policy points (mean cost, mean true utility). U is predicted utility, Y true utility."""
    rows = np.arange(len(Y))
    pts = []
    if rule == "linear":
        scale = np.ptp(Y.mean(0)) / max(np.ptp(c), 1e-9)
        for lam in LAM * scale:
            ch = (U - lam * c).argmax(1)
            pts.append((c[ch].mean(), Y[rows, ch].mean()))
    else:                                     # chebyshev: minimise weighted max distance to ideal (utility 1, cost 0)
        lo, hi = U.min(), U.max()
        un = (U - lo) / max(hi - lo, 1e-9)
        cn = c / max(c.max(), 1e-9)
        for wu in WU:
            ch = np.minimum.reduce([np.maximum(wu * (1 - un), (1 - wu) * cn[None, :])], 0).argmin(1)
            pts.append((c[ch].mean(), Y[rows, ch].mean()))
    return np.array(pts)


def evaluate(w, dims, Z, c, X, split, rule, pred):
    wv = np.array([w[d] for d in dims])
    Y = np.tensordot(wv, Z, axes=1)                                  # (P, A) true utility
    tr, ev = split == "train", split == "dev"
    if pred == "oracle":
        U = Y[ev]
    elif pred == "direct":
        U = P.ridge(X[tr], Y[tr], X[ev])
    else:                                                            # multi: predict each dimension, then scalarise
        Dp = np.stack([P.ridge(X[tr], Z[k][tr], X[ev]) for k in range(len(dims))])
        U = np.tensordot(wv, Dp, axes=1)
    Yev = Y[ev]
    q = Yev.mean(0)
    h = P.hull_fn(c, q)
    pts = route_points(U, Yev, c, rule)
    f = P.hull_fn(np.r_[pts[:, 0], c], np.r_[pts[:, 1], q])
    lo2, hi2 = np.sort(c)[1], c.max()
    bs = np.linspace(lo2, hi2, N_B + 2)[1:-1]
    g = np.array([f(b) - h(b) for b in bs])
    lo, hi = q.min(), q.max()
    nq = (q - lo) / max(hi - lo, 1e-9)
    npu = (pts[:, 1] - lo) / max(hi - lo, 1e-9)
    hv_r = hv(np.r_[pts[:, 0], c], np.r_[npu, nq], c.max())
    hv_h = hv(c, nq, c.max())
    jb = q.argmax()
    need = 0.9 * q[jb]
    cand = pts[pts[:, 1] >= need, 0]
    cpt = float(cand.min() / c[jb]) if len(cand) else np.nan
    cand_h = c[q >= need]
    cpt_h = float(cand_h.min() / c[jb]) if len(cand_h) else np.nan
    return dict(gain=float(g.mean()), gain_min=float(g.min()), dominate=bool((g > 0).all()),
                hv=float(hv_r), hv_hull=float(hv_h), cpt90=cpt, cpt90_hull=cpt_h)


def run(rule, pred, kinds=("named", "corner", "dirichlet"), data_dir=ROOT / "data" / "gen", configs=ROOT / "configs"):
    basis, sc, dims, arms, Z, c, X, split, ok = load(data_dir, configs)
    grid = [g for g in sc["grid"] if g["kind"] in kinds]
    rec = [dict(kind=g["kind"], name=g["name"], **evaluate(g["w"], dims, Z, c, X, split, rule, pred)) for g in grid]
    info = dict(n_arms=len(arms), n_prompts=len(ok), n_train=int((split == "train").sum()),
                n_dev=int((split == "dev").sum()), n_dims=len(dims), n_w=len(grid))
    return pd.DataFrame(rec), info


def summarise(s):
    return dict(gain=s.gain.mean(), gain_min=s.gain.min(), dominance=s.dominate.mean(),
                hv=s.hv.mean(), hv_hull=s.hv_hull.mean(), dhv=s.hv.mean() - s.hv_hull.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", default="linear", choices=["linear", "chebyshev"])
    ap.add_argument("--pred", default="oracle", choices=["oracle", "multi", "direct"])
    ap.add_argument("--kinds", default="named,corner,dirichlet")
    ap.add_argument("--data", default=str(ROOT / "data" / "gen"))
    ap.add_argument("--out", default=str(ROOT / "results" / "pareto"))
    a = ap.parse_args()
    kinds = a.kinds.split(",")
    R, info = run(a.rule, a.pred, kinds, a.data)
    print(f"arms {info['n_arms']}  prompts {info['n_prompts']} (train {info['n_train']} / dev {info['n_dev']})"
          f"  dims {info['n_dims']}  weights {info['n_w']}  rule={a.rule} pred={a.pred}")
    out = Path(a.out) / f"eval_{a.rule}_{a.pred}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    R.to_csv(out, index=False)
    print("\nnamed scenarios:")
    print(R[R.kind == "named"][["name", "gain", "gain_min", "dominate", "hv", "hv_hull", "cpt90", "cpt90_hull"]]
          .round(4).to_string(index=False))
    for k in kinds:
        s = R[R.kind == k]
        if len(s):
            m = summarise(s)
            print(f"\n{k} (n={len(s)}): mean gain {m['gain']:+.4f}  min {m['gain_min']:+.4f}  "
                  f"dominance {m['dominance']:.0%}  HV {m['hv']:.4f} vs hull {m['hv_hull']:.4f}  dHV {m['dhv']:+.4f}")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
