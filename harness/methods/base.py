"""Method interface. To add a method, copy this file to methods/<name>.py and override the three functions.

fit_known(ctx)                      -> (n_eval, n_known) predicted quality of the known arms on eval prompts
select_probes(ctx, k, rng)          -> indices of at most k train prompts (positions in ctx.train, 0..n_train-1)
predict_star(ctx, probe, y, d)      -> (n_eval,) predicted quality of the new arm on eval prompts
    y: target of the new arm on the probes; d: (k, n_dim) per-dimension scores (None if unavailable)

Rules: use only what is in ctx plus the probe labels passed in; never read data/*/labels.csv or any test file;
do not cache state across folds or across new arms. select_probes may use ctx.Q_known_train (known-arm labels).
"""
import numpy as np

from protocol import default_X, ridge


class Method:
    def fit_known(self, ctx):
        return ridge(default_X(ctx.feats_train, slice(None)), ctx.Q_known_train,
                     default_X(ctx.feats_eval, slice(None)))

    def select_probes(self, ctx, k, rng):
        return rng.choice(len(ctx.train), k, replace=False)

    def predict_star(self, ctx, probe, y, d):
        return np.full(len(ctx.eval), y.mean())
