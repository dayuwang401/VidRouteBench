"""Route over (model, number of samples): reproduces Table 6 of the paper (v2) on CPU in a few minutes.

Every prompt may be sent to one arm k times (k = 1..4); a 2B reward model (VideoReward Overall) keeps the best
sample and the frozen target scores it. Cost = k * (arm price + $0.0006 judge). The budget is shared by a group
of B prompts.

    python examples/route_over_samples.py                 # full table, groups of 5/10/20/40/all
    python examples/route_over_samples.py --groups 10 --reps 50

Data: data/samples/labels_multiseed.csv.gz (all scored clips, all seeds), data/samples/noise_panel.csv
(400 prompts with aligned seeds), prompt features data/gen/features/emb_qwen3.npy + data/samples/emb_siglip_text.npy.
Evaluation uses panel dev prompts and seeds 2/3/4/100, all different from the training seed 0.
"""
import argparse, itertools, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "harness"))
from protocol import ridge  # noqa: E402

EV_SEEDS, TR_SEEDS, KMAX, JUDGE = [2, 3, 4, 100], [0, 2, 3, 4], 4, 0.0006
EK = {1: 0.0, 2: 0.5642, 3: 0.8463, 4: 1.0294}                 # E[max of k standard normals]


def bok(T, V, k):
    """mean over all k-subsets of seeds of: pick by V, score by T."""
    acc = [np.take_along_axis(T[list(cb)], V[list(cb)].argmax(0)[None], 0)[0]
           for cb in itertools.combinations(range(T.shape[0]), k)]
    return np.mean(acc, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", default="5,10,20,40,all")
    ap.add_argument("--reps", type=int, default=200)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    arms_df = pd.read_csv(ROOT / "data/gen/arms.csv"); arms_df = arms_df[arms_df.heldout == 0]
    arms = list(arms_df.arm); c = arms_df.set_index("arm").usd_per_clip.loc[arms].to_numpy()
    prompts = pd.read_csv(ROOT / "data/gen/prompts.csv"); pid = list(prompts.prompt_id)
    pr = prompts.set_index("prompt_id"); row = {p: i for i, p in enumerate(pid)}
    X = np.hstack([np.load(ROOT / "data/gen/features/emb_qwen3.npy"), np.load(ROOT / "data/samples/emb_siglip_text.npy")])
    lab = pd.read_csv(ROOT / "data/samples/labels_multiseed.csv.gz"); lab = lab[lab.arm.isin(arms)]
    panel = set(pd.read_csv(ROOT / "data/samples/noise_panel.csv").prompt_id)
    piv = lambda sub, col, idx: sub.pivot_table(index="prompt_id", columns="arm", values=col, aggfunc="mean").reindex(index=idx, columns=arms).to_numpy()

    # evaluation: panel dev prompts with all 16 arms on seeds 2/3/4/100
    dev = [p for p in pid if pr.loc[p, "split"] == "dev" and p in panel]
    Te = np.stack([piv(lab[lab.seed == s], "target", dev) for s in EV_SEEDS])
    Ve = np.stack([piv(lab[lab.seed == s], "vr_overall", dev) for s in EV_SEEDS])
    ok = ~np.isnan(Te).any((0, 2)) & ~np.isnan(Ve).any((0, 2))
    dev = [p for p, o in zip(dev, ok) if o]; Te, Ve = Te[:, ok], Ve[:, ok]; di = np.array([row[p] for p in dev]); n = len(dev)
    ACT = [(j, k) for k in range(1, KMAX + 1) for j in range(len(arms))]
    Q = np.stack([bok(Te[:, :, j], Ve[:, :, j], k) for j, k in ACT], 1)
    C = np.array([k * (c[j] + JUDGE) for j, k in ACT]); K1 = np.array([k == 1 for _, k in ACT])

    # training (train prompts only)
    trp = [p for p in pid if pr.loc[p, "split"] == "train"]; ti = np.array([row[p] for p in trp])
    Ytr = piv(lab[lab.prompt_id.isin(trp)], "target", trp)
    mu = np.stack([ridge(X[ti][~np.isnan(Ytr[:, j])], Ytr[~np.isnan(Ytr[:, j]), j:j + 1], X[di])[:, 0] for j in range(len(arms))], 1)
    tpp = [p for p in trp if p in panel]
    Tt = np.stack([piv(lab[lab.seed == s], "target", tpp) for s in TR_SEEDS]); Vt = np.stack([piv(lab[lab.seed == s], "vr_overall", tpp) for s in TR_SEEDS])
    okt = ~np.isnan(Tt).any((0, 2)) & ~np.isnan(Vt).any((0, 2)); Tt, Vt = Tt[:, okt], Vt[:, okt]
    tpi = np.array([row[p] for p, o in zip(tpp, okt) if o]); base_t = Tt.mean(0)
    gk = {(j, k): float((bok(Tt[:, :, j], Vt[:, :, j], k) - base_t[:, j]).mean()) if k > 1 else 0.0 for j, k in ACT}
    sd_t = Tt.std(0, ddof=1); sig = sd_t.mean(0)
    eff = np.array([gk[(j, 2)] / max(sig[j] * EK[2], 1e-9) for j in range(len(arms))])
    sd_hat = np.clip(np.stack([ridge(X[tpi], sd_t[:, j:j + 1], X[di])[:, 0] for j in range(len(arms))], 1), 0, None)
    MU = mu[:, [j for j, _ in ACT]]
    M2 = np.stack([mu[:, j] if k == 1 else mu[:, j] + ridge(X[tpi], (bok(Tt[:, :, j], Vt[:, :, j], k) - base_t[:, j])[:, None], X[di])[:, 0]
                   for j, k in ACT], 1)
    POL = {"R1 router, one sample": np.where(K1[None], MU, -np.inf),
           "R2-M1 router + per-arm g_k": MU + np.array([gk[x] for x in ACT])[None],
           "R2-M3 router + predicted spread": MU + sd_hat[:, [j for j, _ in ACT]] * np.array([eff[j] * EK[k] for j, k in ACT])[None],
           "R2-M2 router, best-of-k targets": M2,
           "oracle (true values)": Q}

    budgets = np.linspace(np.sort(C[K1])[1], C[K1].max(), 10)[1:-1]
    LAM = np.concatenate([[0], np.logspace(-4, 3, 160)])
    qbar = np.array([base_t[:, j].mean() + gk[(j, k)] for j, k in ACT])

    def route(U, tot):
        best = None
        for lam in LAM:
            ch = (U - lam * C[None]).argmax(1)
            if C[ch].sum() <= tot + 1e-12:
                v = U[np.arange(len(ch)), ch].sum()
                if best is None or v > best[0]: best = (v, ch)
        return best[1]

    def agnostic(mask, B, tot):        # best integer mix of at most two actions, chosen from training means
        idx = np.where(mask)[0]; best = (-np.inf, None)
        for i in idx:
            for j in idx[idx >= i]:
                for ki in range(B + 1):
                    kj = B - ki
                    if ki * C[i] + kj * C[j] <= tot + 1e-12 and ki * qbar[i] + kj * qbar[j] > best[0]:
                        best = (ki * qbar[i] + kj * qbar[j], (i, ki, j, kj))
        return best[1]

    print(f"{n} dev prompts x {len(arms)} arms x seeds {EV_SEEDS}; {len(ACT)} actions (k<=4)\n")
    for G in a.groups.split(","):
        B = n if G == "all" else int(G); reps = 1 if B == n else a.reps
        res = {nm: [] for nm in ["B0 prompt-agnostic, one sample", "B1 prompt-agnostic resampling"] + list(POL)}
        for b in budgets:
            tot = B * b; acc = {nm: [] for nm in res}
            for _ in range(reps):
                g = np.arange(n) if B == n else rng.choice(n, B, replace=False)
                for nm, m in [("B0 prompt-agnostic, one sample", K1), ("B1 prompt-agnostic resampling", np.ones(len(ACT), bool))]:
                    i, ki, j, kj = agnostic(m, B, tot); acc[nm].append((ki * Q[g, i].mean() + kj * Q[g, j].mean()) / B)
                for nm, U in POL.items():
                    acc[nm].append(Q[g, route(U[g], tot)].mean())
            for nm in res: res[nm].append(np.array(acc[nm]))
        b0 = np.mean([x.mean() for x in res["B0 prompt-agnostic, one sample"]])
        b1 = res["B1 prompt-agnostic resampling"]
        print(f"group size {G}")
        for nm, v in res.items():
            gain = np.mean([x.mean() for x in v]) - b0
            lose = np.mean([np.mean(x < y) for x, y in zip(v, b1)])
            win = 1 / (1 + np.exp(-gain))
            print(f"  {nm:34s} gain vs B0 {gain:+.3f}   win rate {win:.1%}   groups losing to B1 {lose:.0%}")
        print()


if __name__ == "__main__":
    main()
