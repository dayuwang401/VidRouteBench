"""Bag-of-words baseline: prompts are represented by TF-IDF instead of a pretrained embedding; otherwise
identical to profile.

Answers "does routing need a semantic prompt representation?": TF-IDF sees word overlap only.
Features are computed from ctx.text_* inside the method and never written to data/*/features/, so other
methods' inputs are untouched.
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

from methods.base import Method as Base
from protocol import ridge


class Method(Base):
    def _X(self, ctx):
        v = TfidfVectorizer(sublinear_tf=True, min_df=3, max_features=4096,
                            ngram_range=(1, 2), stop_words="english")
        Xtr = v.fit_transform(ctx.text_train).toarray()
        Xev = v.transform(ctx.text_eval).toarray()
        s = Xtr.std(0)
        keep = s > 0
        return Xtr[:, keep], Xev[:, keep]

    def fit_known(self, ctx):
        Xtr, Xev = self._X(ctx)
        self._Pk = ridge(Xtr, ctx.Q_known_train, Xev)
        return self._Pk

    def predict_star(self, ctx, probe, y, d):
        F = ctx.Q_known_train[probe]
        mu_f, mu_y = F.mean(0), y.mean()
        w = Ridge(alpha=1.0).fit(F - mu_f, y - mu_y).coef_
        return mu_y + (self._Pk - mu_f) @ w
