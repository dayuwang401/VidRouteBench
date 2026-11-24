"""Run one cold-start evaluation and append the result to results.tsv.

  python run.py --method mean                  # dev evaluation (train -> dev)
  python run.py --method mean --cv 5           # 5-fold CV over train U dev (more stable, 5x slower)
  python run.py --method mean --data ../data/rb_human --cv 5   # RewardBench human labels (12 closed models)
  python run.py --final ...                    # maintainers only: needs the decrypted test split

score = recovery averaged over K values and over arms with enough room. Full-label routing gain and mean
probe cost are reported as well.
"""
import argparse
import importlib
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

import protocol as P

HERE = Path(__file__).resolve().parent


def folds_of(data, cv, final):
    sp = data.prompts.split.to_numpy()
    if final:
        return [(np.where(np.isin(sp, ["train", "dev"]))[0], np.where(sp == "test")[0])]
    pool = np.where(np.isin(sp, ["train", "dev"]))[0]
    if cv:
        return [(pool[a], pool[b]) for a, b in KFold(cv, shuffle=True, random_state=0).split(pool)]
    return [(np.where(sp == "train")[0], np.where(sp == "dev")[0])]


def make_ctx(data, tr, ev, known, star):
    kn = data.arms.index[data.arms.arm.isin(known)].to_numpy()
    return P.Ctx(
        train=tr, eval=ev,
        feats_train={k: v[tr] for k, v in data.feats.items()},
        feats_eval={k: v[ev] for k, v in data.feats.items()},
        text_train=data.prompts.text.iloc[tr].tolist(), text_eval=data.prompts.text.iloc[ev].tolist(),
        known_arms=data.arms.iloc[kn].reset_index(drop=True), star_arm=data.arms.iloc[star],
        Q_known_train=data.Q[np.ix_(tr, kn)], D_known_train=data.D[np.ix_(tr, kn)] if data.dims else None,
        dims=data.dims, prompts_train=data.prompts.iloc[tr].reset_index(drop=True),
        prompts_eval=data.prompts.iloc[ev].reset_index(drop=True))


def evaluate(method, data, folds, ks, reps, seed=0):
    arms = data.arms[data.arms.heldout == 0].index.to_numpy()
    c_all = data.arms.usd_per_clip.to_numpy()
    rec = []
    for fi, (tr0, ev0) in enumerate(folds):
        # only prompts labelled for every participating arm
        ok = ~np.isnan(data.Q[:, arms]).any(1)
        tr, ev = tr0[ok[tr0]], ev0[ok[ev0]]
        Y, c = data.Q[np.ix_(ev, arms)], c_all[arms]
        budgets = P.budgets_for(c)
        for si, star in enumerate(arms):
            known = [data.arms.arm[a] for a in arms if a != star]
            ctx = make_ctx(data, tr, ev, known, star)
            Pk = method.fit_known(ctx)                                  # (n_ev, n_known)
            ystar_tr = data.Q[tr, star]
            dstar_tr = data.D[tr, star] if data.dims else None
            X = P.default_X(data.feats, slice(None))

            def score(pred):
                Pm = np.zeros((len(ev), len(arms)))
                Pm[:, [i for i in range(len(arms)) if i != si]] = Pk
                Pm[:, si] = -np.inf if pred is None else pred
                return P.frontier_gain(Pm, Y, c, budgets)

            g_full = score(P.ridge(X[tr], ystar_tr[:, None], X[ev])[:, 0])
            g_excl = score(None)
            for K in ks:
                for r in range(reps):
                    rng = np.random.default_rng([seed, fi, si, K, r])
                    probe = np.asarray(method.select_probes(ctx, K, rng))
                    assert len(probe) <= K and len(set(probe.tolist())) == len(probe), "too many or duplicate probes"
                    pred = method.predict_star(ctx, probe, ystar_tr[probe],
                                               None if dstar_tr is None else dstar_tr[probe])
                    g = score(np.asarray(pred, float))
                    rec.append(dict(fold=fi, star=data.arms.arm[star], K=K, rep=r, gain=g.mean(),
                                    full=g_full.mean(), excl=g_excl.mean(),
                                    probe_cost=len(probe) * c_all[star]))
    R = pd.DataFrame(rec)
    base = R.groupby("star")[["full", "excl"]].mean()
    base["room"] = base.full - base.excl
    useful = base.index[base.room > P.ROOM_MIN]
    S = R[R.star.isin(useful)].groupby("K").agg(gain=("gain", "mean"), cost=("probe_cost", "mean"))
    full_m, excl_m = base.loc[useful, "full"].mean(), base.loc[useful, "excl"].mean()
    S["recovery"] = (S.gain - excl_m) / (full_m - excl_m)
    return dict(score=float(S.recovery.mean()), by_k=S, base=base, useful=list(useful),
                full_gain=float(base.full.mean()), raw=R)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, help="methods/<name>.py")
    ap.add_argument("--data", default=str(HERE.parent / "data" / "gen"))
    ap.add_argument("--cv", type=int, default=0)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--ks", default=",".join(map(str, P.KS)))
    ap.add_argument("-m", "--message", default="")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--no-log", action="store_true")
    a = ap.parse_args()

    t0 = time.time()
    data = P.load(a.data)
    if a.final:
        assert (data.prompts.split == "test").any(), "no test prompts: --final needs the decrypted test split merged in"
    method = importlib.import_module(f"methods.{a.method}").Method()
    res = evaluate(method, data, folds_of(data, a.cv, a.final), [int(k) for k in a.ks.split(",")], a.reps)
    dt = time.time() - t0
    print(res["base"].round(4).to_string())
    print(f"\narms with room > {P.ROOM_MIN}: {res['useful']}")
    print(res["by_k"].round(4).to_string())
    print(f"\nscore(recovery)={res['score']:.4f}  full_gain={res['full_gain']:+.4f}  {dt:.0f}s")
    if not a.no_log:
        try:
            commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                                    cwd=HERE).stdout.strip()
        except Exception:
            commit = ""
        f = HERE.parent / ("results_final.tsv" if a.final else "results.tsv")
        new = not f.exists()
        with open(f, "a") as fh:
            if new:
                fh.write("time\tcommit\tdata\tmethod\tcv\tscore\tfull_gain\tsec\tmessage\n")
            fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{commit}\t{Path(a.data).name}\t{a.method}\t{a.cv}\t"
                     f"{res['score']:.4f}\t{res['full_gain']:.4f}\t{dt:.0f}\t{a.message}\n")


if __name__ == "__main__":
    main()
