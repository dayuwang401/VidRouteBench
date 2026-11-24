"""Reproduce every table of arXiv:2611.28099 v1 from the released data.

    python reproduce/reproduce_v1.py            # ~3 min on CPU; writes results/v1_reproduction.md

The v1 tables were computed while generation was still running, so they come from three states of the same
append-only label file. data/gen/ is the latest of the three; reproduce/v1/ holds row manifests for the two
earlier ones:
  gate    (Table 2)        8 arms, 899 prompts, rows in v1/gate_rows.csv.gz, arm prices in v1/gate_arms.csv
  pareto  (Table 5)        16 arms, rows in v1/pareto_rows.csv.gz
  current (Tables 1, 3, 4) data/gen/ as shipped
Every reproduced number is rounded to the precision printed in the paper and compared with it.
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "harness"), str(ROOT / "scripts")]
import protocol as P              # noqa: E402
import run as H                   # noqa: E402
import gate_check                 # noqa: E402
import pareto_eval                # noqa: E402
from methods import mean, profile, ridge_k, bow, dimprofile   # noqa: E402

DATA = ROOT / "data" / "gen"
V1 = Path(__file__).resolve().parent / "v1"
BUILD = ROOT / "build" / "v1"
KEY = ["prompt_id", "arm", "seed"]


def snapshot(name, rows, arms=None, restrict_prompts=False):
    out = BUILD / name
    (out / "features").mkdir(parents=True, exist_ok=True)
    keep = pd.read_csv(rows)
    lab = pd.read_csv(DATA / "labels.csv").merge(keep, on=KEY, how="inner")
    assert len(lab) == len(keep), f"{name}: {len(keep) - len(lab)} manifest rows missing from data/gen/labels.csv"
    prompts = pd.read_csv(DATA / "prompts.csv")
    sel = prompts.prompt_id.isin(lab.prompt_id).to_numpy() if restrict_prompts else np.ones(len(prompts), bool)
    prompts[sel].to_csv(out / "prompts.csv", index=False)
    for f in (DATA / "features").glob("*.npy"):
        np.save(out / "features" / f.name, np.load(f)[sel])
    lab.to_csv(out / "labels.csv", index=False)
    shutil.copy(arms or DATA / "arms.csv", out / "arms.csv")
    return out


def cold(method, data, cv):
    return H.evaluate(method.Method(), data, H.folds_of(data, cv, False), P.KS, 3)


def main():
    rows = []

    def check(table, name, got, paper, nd, pct=False):
        v = 100 * got if pct else got
        if f"{v:.{nd}f}" == f"{paper:.{nd}f}":
            status = "ok"
        elif abs(v - paper) <= 10 ** -nd + 1e-12:     # paper rounded a 4-decimal printout a second time
            status = f"ok (rounding, {v:.{nd + 2}f})"
        else:
            status = "MISMATCH"
        rows.append((table, name, f"{paper:.{nd}f}", f"{v:.{nd}f}", status))

    # ---- Table 1: arms (mean Q on seed 0, current data) ----
    lab = pd.read_csv(DATA / "labels.csv")
    mq = lab[lab.seed == 0].groupby("arm").target.mean()
    for arm, q in {"ltx097d_lr8": -1.444, "ltx097d_s8": -1.782, "ltx097d_s4": -2.074, "wan22_5b_s20": -1.355,
                   "wan22_5b_s10": -1.814, "cog2b_s25": -1.485, "wan21_13b_s25": -0.985, "wan22_5b_s40": -1.120,
                   "wan21_13b_s12": -1.650, "cog5b_s25": -1.403, "cog5b_s15": -2.063, "wan21_13b_s50": -0.624,
                   "cog2b_s50": -0.726, "cog5b_s50": -0.501, "kandinsky5_s50": -0.202,
                   "wan22_a14b_s40": -0.122}.items():
        check("T1 arms", f"mean Q {arm}", mq[arm], q, 3)

    # ---- Table 2: gate (gate snapshot) ----
    g = gate_check.compute(snapshot("gate", V1 / "gate_rows.csv.gz", V1 / "gate_arms.csv", restrict_prompts=True))
    for k, paper, nd, pct in [("sbm", -0.513, 3, 0), ("oracle", 0.129, 3, 0), ("oracle_gap", 0.642, 3, 0),
                              ("gap_within", 0.426, 3, 0), ("gap_cross", 0.199, 3, 0), ("gap_real", 47, 0, 1),
                              ("seed_sd", 0.429, 3, 0), ("var_prompt", 55.3, 1, 1), ("var_arm", 11.3, 1, 1),
                              ("var_inter", 33.4, 1, 1), ("F", 4.47, 2, 0), ("var_inter_comp", 0.585, 3, 0),
                              ("var_noise_comp", 0.337, 3, 0), ("inter_share", 63, 0, 1), ("r2_mean", 0.205, 3, 0),
                              ("full_gain", 0.188, 3, 0), ("shuf_gain", 0.010, 3, 0), ("shuf_gain_sd", 0.012, 3, 0)]:
        check("T2 gate", k, g[k], paper, nd, bool(pct))
    check("T2 gate", "F p-value (x1e-63)", g["F_p"] * 1e63, 1.6, 1)
    check("T2 gate", "R2 min", min(g["r2_by_arm"].values()), 0.091, 3)
    check("T2 gate", "R2 max", max(g["r2_by_arm"].values()), 0.306, 3)

    # ---- Tables 3 and 4: router baselines and cold start (current data) ----
    data = P.load(DATA)
    res = {(m.__name__.split(".")[-1], cv): cold(m, data, cv)
           for m in (mean, profile, ridge_k, bow, dimprofile) for cv in (5, 0)}
    check("T3 router", "embedding ridge, 5-fold", res["profile", 5]["full_gain"], 0.210, 3)
    check("T3 router", "TF-IDF ridge, 5-fold", res["bow", 5]["full_gain"], 0.175, 3)
    check("T3 router", "embedding ridge, train->dev", res["profile", 0]["full_gain"], 0.168, 3)
    check("T3 router", "TF-IDF ridge, train->dev", res["bow", 0]["full_gain"], 0.009, 3)
    for m, cv5, dev in [("mean", -3.32, -0.66), ("ridge_k", -2.44, -0.52), ("profile", 0.04, 0.21),
                        ("dimprofile", -0.23, 0.25)]:
        check("T4 cold start", f"{m} recovery, 5-fold", res[m, 5]["score"], cv5, 2)
        check("T4 cold start", f"{m} recovery, dev", res[m, 0]["score"], dev, 2)
    for K, c in zip(P.KS, (1.18, 2.36, 5.90)):
        check("T4 cold start", f"probe cost K={K} ($, dev)", res["mean", 0]["by_k"].loc[K, "cost"], c, 2)
    rb = P.load(ROOT / "data" / "rb_human")
    rbm, rbp = cold(mean, rb, 5), cold(profile, rb, 5)
    check("T4 cold start", "RewardBench mean recovery", rbm["score"], 0.786, 3)
    check("T4 cold start", "RewardBench profile recovery", rbp["score"], 0.815, 3)
    check("T4 cold start", "RewardBench full-label gain", rbm["full_gain"], 0.030, 3)

    # ---- Table 5: OPMU over 200 Dirichlet directions (pareto snapshot) ----
    snap = snapshot("pareto", V1 / "pareto_rows.csv.gz")
    for label, rule, pred, gain, gmin, dom, dhv in [
            ("oracle linear", "linear", "oracle", 0.382, 0.225, 100, 0.680),
            ("oracle Chebyshev", "chebyshev", "oracle", 0.373, 0.217, 100, 0.643),
            ("OPMU linear", "linear", "multi", 0.050, 0.000, 94, 0.110),
            ("per-scenario linear", "linear", "direct", 0.045, 0.000, 90, 0.103),
            ("OPMU Chebyshev", "chebyshev", "multi", 0.043, 0.000, 85, 0.092)]:
        R, _ = pareto_eval.run(rule, pred, ("dirichlet",), snap)
        s = pareto_eval.summarise(R)
        check("T5 pareto", f"{label} gain", s["gain"], gain, 3)
        check("T5 pareto", f"{label} min gain", s["gain_min"], gmin, 3)
        check("T5 pareto", f"{label} dominance (%)", s["dominance"], dom, 0, pct=True)
        check("T5 pareto", f"{label} dHV", s["dhv"], dhv, 3)

    T = pd.DataFrame(rows, columns=["table", "quantity", "paper", "reproduced", "status"])
    bad = (T.status == "MISMATCH").sum()
    out = ROOT / "results" / "v1_reproduction.md"
    out.parent.mkdir(exist_ok=True)
    lines = ["| table | quantity | paper | reproduced | status |", "|---|---|---|---|---|"]
    lines += [f"| {' | '.join(r)} |" for r in rows]
    out.write_text("# arXiv:2611.28099 v1 reproduction\n\n" + "\n".join(lines) + "\n")
    print(T.to_string(index=False))
    print(f"\n{len(T) - bad}/{len(T)} numbers match the paper -> {out.relative_to(ROOT)}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
