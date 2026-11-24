"""Gate: does per-prompt specialisation exist at all? (paper Sec. 4, Table "gate")

1) oracle gap, and how much of it survives selecting on seed 0 and scoring on seed 1;
2) prompt x arm interaction: variance decomposition, F test with seed replication, and whether the interaction
   is predictable from prompt features (out-of-fold R^2);
3) full-label routing gain versus a shuffled-label control (harness protocol).

    python gate_check.py [--data DIR]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness"))
import protocol as P   # noqa: E402


def wide(lab, seed):
    d = lab[lab.seed == seed]
    return d.pivot_table(index="prompt_id", columns="arm", values="target", aggfunc="mean")


def compute(data_dir):
    data = P.load(data_dir)
    lab = pd.read_csv(Path(data_dir) / "labels.csv")
    arms = list(data.arms[data.arms.heldout == 0].arm)
    Q0 = wide(lab, 0)[arms].dropna()
    out = dict(n_arms=len(arms), n_prompts=len(Q0))

    # ---------- 1. oracle gap and seed noise ----------
    out["sbm"] = sbm = Q0.mean(axis=0).max()
    out["oracle"] = orc = Q0.max(axis=1).mean()
    out["oracle_gap"] = orc - sbm
    Q1 = wide(lab, 1)
    common = [a for a in arms if a in Q1.columns and Q1[a].notna().any()]
    ids = Q0.index.intersection(Q1.dropna(subset=common).index)
    if len(ids) > 30 and len(common) >= 3:
        A, B = Q0.loc[ids, common], Q1.loc[ids, common]
        # pick the arm on one seed, score on the other; pure noise would give <= 0
        pick_a = A.to_numpy().argmax(1)
        pick_b = B.to_numpy().argmax(1)
        r = np.arange(len(ids))
        cross = 0.5 * ((B.to_numpy()[r, pick_a] - B.to_numpy().mean(0).max())
                       + (A.to_numpy()[r, pick_b] - A.to_numpy().mean(0).max()))
        within = 0.5 * ((A.max(axis=1) - A.mean(axis=0).max()).mean() + (B.max(axis=1) - B.mean(axis=0).max()).mean())
        sd = np.nanmean([lab[(lab.arm == a) & (lab.prompt_id.isin(ids))].groupby("prompt_id").target.std().mean()
                         for a in common])
        out.update(seed_subset=(len(ids), len(common)), gap_within=within, gap_cross=cross.mean(),
                   gap_real=cross.mean() / within, seed_sd=sd)

    # ---------- 2. interaction ----------
    M = Q0.to_numpy()
    gm = M.mean()
    E = M - M.mean(1, keepdims=True) - M.mean(0, keepdims=True) + gm
    v_tot = M.var()
    out.update(var_prompt=M.mean(1).var() / v_tot, var_arm=M.mean(0).var() / v_tot, var_inter=(E ** 2).mean() / v_tot)
    # two-way ANOVA with replication (seeds 0/1): F = interaction MS / within-cell (seed) MS.
    # Note: permuting prompts within each arm is NOT a valid null here, because it also destroys the prompt
    # main effect and inflates the null interaction variance.
    have = lab[lab.seed == 1].arm.unique()
    cnt = lab[lab.arm.isin(have)].groupby(["prompt_id", "arm"]).seed.nunique().unstack().dropna()
    ids2 = cnt.index[(cnt == 2).all(axis=1)]
    if len(ids2) > 30:
        aa = sorted(have)
        d2 = lab[lab.arm.isin(aa) & lab.prompt_id.isin(ids2)]
        Y2 = np.zeros((len(ids2), len(aa), 2))
        pix2 = {q: i for i, q in enumerate(ids2)}
        for r in d2.itertuples():
            Y2[pix2[r.prompt_id], aa.index(r.arm), int(r.seed)] = r.target
        g2, cell = Y2.mean(), Y2.mean(2)
        n_p, n_a = len(ids2), len(aa)
        ss_i = 2 * ((cell - cell.mean(1, keepdims=True) - cell.mean(0, keepdims=True) + g2) ** 2).sum()
        ss_e = ((Y2 - cell[:, :, None]) ** 2).sum()
        df_i, df_e = (n_p - 1) * (n_a - 1), n_p * n_a
        ms_i, ms_e = ss_i / df_i, ss_e / df_e
        F = ms_i / ms_e
        v_i = max(0.0, (ms_i - ms_e) / 2)
        out.update(anova_subset=(n_p, n_a), F=F, F_df=(df_i, df_e), F_p=stats.f.sf(F, df_i, df_e),
                   var_inter_comp=v_i, var_noise_comp=ms_e, inter_share=v_i / (v_i + ms_e))
    # is the interaction predictable from prompt features? (out-of-fold R^2 per arm)
    X = P.default_X(data.feats, slice(None))
    pid = pd.Index(data.prompts.prompt_id)
    X = X[pid.get_indexer(Q0.index)]
    r2 = []
    for j in range(M.shape[1]):
        y = E[:, j]
        pred = np.zeros_like(y)
        for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
            pred[te] = P.ridge(X[tr], y[tr][:, None], X[te])[:, 0]
        r2.append(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    out["r2_by_arm"] = dict(zip(Q0.columns, r2))
    out["r2_mean"] = float(np.mean(r2))

    # ---------- 3. full-label routing gain vs shuffled control ----------
    ai = data.arms.index[data.arms.arm.isin(arms)].to_numpy()
    c = data.arms.usd_per_clip.to_numpy()[ai]
    sp = data.prompts.split.to_numpy()
    ok = ~np.isnan(data.Q[:, ai]).any(1)
    tr = np.where((sp == "train") & ok)[0]
    ev = np.where((sp == "dev") & ok)[0]
    Xa = P.default_X(data.feats, slice(None))
    Y = data.Q[np.ix_(ev, ai)]
    b = P.budgets_for(c)
    g_true = P.frontier_gain(P.ridge(Xa[tr], data.Q[np.ix_(tr, ai)], Xa[ev]), Y, c, b)
    sh = []
    for s in range(5):
        r2_ = np.random.default_rng(s).permutation(len(tr))
        g = P.frontier_gain(P.ridge(Xa[tr], data.Q[np.ix_(tr, ai)][r2_], Xa[ev]), Y, c, b)
        sh.append(g.mean())
    out.update(full_gain=g_true.mean(), full_gain_by_budget=g_true, shuf_gain=np.mean(sh), shuf_gain_sd=np.std(sh))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "gen"))
    o = compute(ap.parse_args().data)
    print(f"arms {o['n_arms']}, prompts complete on seed 0: {o['n_prompts']}")
    print("\n1) oracle gap (seed 0)")
    print(f"   single best arm (SBM) {o['sbm']:.4f}   per-prompt oracle {o['oracle']:.4f}   gap {o['oracle_gap']:+.4f}")
    if "gap_within" in o:
        print(f"   two-seed subset: {o['seed_subset'][0]} prompts x {o['seed_subset'][1]} arms")
        print(f"   same-seed gap {o['gap_within']:+.4f}   cross-seed gap {o['gap_cross']:+.4f}"
              f"  -> real fraction {o['gap_real']:.0%}")
        print(f"   seed s.d. of target (same prompt, same arm) {o['seed_sd']:.4f}")
    else:
        print("   not enough two-seed data, noise estimate skipped")
    print("\n2) variance decomposition (share of total)")
    print(f"   prompt {o['var_prompt']:.1%}   arm {o['var_arm']:.1%}   interaction+residual {o['var_inter']:.1%}")
    if "F" in o:
        n_p, n_a = o["anova_subset"]
        print(f"   replicated subset {n_p} prompts x {n_a} arms x 2 seeds: interaction F{o['F_df']} = {o['F']:.3f}"
              f"  p = {o['F_p']:.3g}")
        print(f"   variance components: interaction {o['var_inter_comp']:.4f}  seed noise {o['var_noise_comp']:.4f}"
              f"  -> interaction share {o['inter_share']:.0%}")
    print("   out-of-fold R^2 of the interaction per arm (>0: prompt text predicts the arm's relative strength):")
    for a, v in sorted(o["r2_by_arm"].items(), key=lambda t: -t[1]):
        print(f"      {a:16s} {v:+.3f}")
    print(f"   mean {o['r2_mean']:+.3f}")
    print("\n3) full-label routing gain (harness protocol, dev split)")
    print(f"   real features {o['full_gain']:+.4f}   shuffled labels {o['shuf_gain']:+.4f} +- {o['shuf_gain_sd']:.4f}")
    print(f"   per budget: {np.round(o['full_gain_by_budget'], 4)}")


if __name__ == "__main__":
    main()
