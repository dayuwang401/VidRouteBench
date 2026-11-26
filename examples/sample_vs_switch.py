"""Sampling a weak model k times vs. calling a strong model once, at matched cost (Section 8, Figure 2 of v2).

    python examples/sample_vs_switch.py            # per-arm best-of-k table + matched-cost comparison
    python examples/sample_vs_switch.py --prompt p00042   # show one prompt: every arm, every seed

best-of-k: draw k of the 4 aligned seeds (0/2/3/4), keep the one VideoReward Overall prefers, score with the frozen
target; averaged over all k-subsets and over the 400-prompt noise panel (prompts where all 16 arms have the 4 seeds).
"""
import argparse, itertools
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEEDS, JUDGE = [0, 2, 3, 4], 0.0006


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--prompt", default=None); ap.add_argument("--strong", default="wan22_a14b_s40")
    a = ap.parse_args()
    arms_df = pd.read_csv(ROOT / "data/gen/arms.csv"); arms_df = arms_df[arms_df.heldout == 0]
    arms = list(arms_df.arm); cost = arms_df.set_index("arm").usd_per_clip
    lab = pd.read_csv(ROOT / "data/samples/labels_multiseed.csv.gz")
    if a.prompt:
        txt = pd.read_csv(ROOT / "data/gen/prompts.csv").set_index("prompt_id").text[a.prompt]
        print(f"{a.prompt}: {txt}\n")
        t = lab[(lab.prompt_id == a.prompt) & lab.seed.isin(SEEDS)].pivot_table(index="arm", columns="seed", values="target")
        v = lab[(lab.prompt_id == a.prompt) & lab.seed.isin(SEEDS)].pivot_table(index="arm", columns="seed", values="vr_overall")
        t["$/clip"] = cost.reindex(t.index); t["VR picks seed"] = v.idxmax(axis=1)
        print(t.sort_values("$/clip").round(3).to_string()); return
    panel = pd.read_csv(ROOT / "data/samples/noise_panel.csv").prompt_id
    cube = lambda col: np.stack([lab[lab.seed == s].pivot_table(index="prompt_id", columns="arm", values=col).reindex(index=panel, columns=arms).to_numpy() for s in SEEDS])
    T, V = cube("target"), cube("vr_overall"); ok = ~np.isnan(T).any((0, 2)) & ~np.isnan(V).any((0, 2)); T, V = T[:, ok], V[:, ok]
    print(f"{ok.sum()} panel prompts x {len(arms)} arms x seeds {SEEDS}")
    rows = []
    for j, arm in enumerate(arms):
        for k in range(1, 5):
            acc = [np.take_along_axis(T[list(cb), :, j], V[list(cb), :, j].argmax(0)[None], 0)[0] for cb in itertools.combinations(range(4), k)]
            rows.append(dict(arm=arm, k=k, cost=k * (cost[arm] + JUDGE), quality=float(np.mean(acc))))
    R = pd.DataFrame(rows)
    W = R.pivot(index="arm", columns="k", values="quality"); W["$/clip"] = cost.reindex(W.index); W["k=1->4"] = W[4] - W[1]
    print("\nbest-of-k quality per arm (frozen target; VideoReward selects):")
    print(W.sort_values("$/clip").round(3).to_string())
    gap = T.mean(0).max(1).mean() - T.mean(0).mean(0).max()
    print(f"\nseed s.d. within (prompt, arm): {T.std(0).mean():.3f}   perfect per-prompt model choice gains: {gap:.3f}")
    s = R[(R.arm == a.strong) & (R.k == 1)].iloc[0]
    print(f"\nmatched cost vs one sample of {a.strong} (${s.cost:.3f}, quality {s.quality:+.3f}):")
    cand = R[(R.cost <= s.cost * 1.001) & (R.arm != a.strong)].sort_values("quality", ascending=False).head(5)
    for r in cand.itertuples():
        print(f"  {r.arm:16s} x{r.k}   ${r.cost:.3f}   quality {r.quality:+.3f}   ({r.quality - s.quality:+.3f}, "
              f"predicted human win rate {1 / (1 + np.exp(-(r.quality - s.quality))):.1%})")


if __name__ == "__main__":
    main()
