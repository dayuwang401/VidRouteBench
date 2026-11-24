"""M1 baseline (profile): on the probes, regress the new arm's quality on the known arms' quality,
then apply that map to the known arms' predictions on eval prompts."""
import numpy as np
from sklearn.linear_model import Ridge

from methods.base import Method as Base


class Method(Base):
    def fit_known(self, ctx):
        self._Pk = super().fit_known(ctx)
        return self._Pk

    def predict_star(self, ctx, probe, y, d):
        F = ctx.Q_known_train[probe]
        mu_f, mu_y = F.mean(0), y.mean()
        w = Ridge(alpha=1.0).fit(F - mu_f, y - mu_y).coef_
        return mu_y + (self._Pk - mu_f) @ w
