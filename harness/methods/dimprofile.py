"""dim-profile (reported in the paper as a rejected variant): regress the new arm's probe scores on the known
arms' scores *per quality dimension*, then map the dimension predictions back to the scalar target.

Motivation: the frozen scalar target collapses several quality dimensions into one number, while arms differ
in which dimension they are strong at. The probe run already yields d in R^{K x D}; profiling each dimension
separately uses that information. Known-arm dimension predictions are fitted from ctx.D_known_train with the
same prompt features as the scalar channel.
"""
import numpy as np
from sklearn.linear_model import Ridge

from methods.base import Method as Base
from protocol import default_X, ridge


class Method(Base):
    ALPHA = 1.0

    def fit_known(self, ctx):
        Xtr = default_X(ctx.feats_train, slice(None))
        Xev = default_X(ctx.feats_eval, slice(None))
        self._Pk = ridge(Xtr, ctx.Q_known_train, Xev)             # (n_ev, n_known) scalar channel
        D = ctx.D_known_train                                      # (n_tr, n_known, n_dim)
        self._Dk = None
        if D is not None and D.size:
            nk, nd = D.shape[1], D.shape[2]
            ok = ~np.isnan(D).any((0, 1))
            self._ok = ok
            flat = D[:, :, ok].reshape(len(D), -1)
            self._Dk = ridge(Xtr, flat, Xev).reshape(len(Xev), nk, ok.sum())
        return self._Pk

    def predict_star(self, ctx, probe, y, d):
        mu_y = y.mean()
        if self._Dk is None or d is None:
            F = ctx.Q_known_train[probe]
            w = Ridge(alpha=self.ALPHA).fit(F - F.mean(0), y - mu_y).coef_
            return mu_y + (self._Pk - F.mean(0)) @ w
        # per dimension: regress the new arm's probe scores on the known arms' scores in that dimension
        dk = ctx.D_known_train[probe][:, :, self._ok]              # (K, n_known, n_dim)
        ds = d[:, self._ok]                                        # (K, n_dim)
        pred = np.zeros((len(self._Pk), ds.shape[1]))
        for j in range(ds.shape[1]):
            F, yj = dk[:, :, j], ds[:, j]
            mu_f, mu_j = F.mean(0), yj.mean()
            w = Ridge(alpha=self.ALPHA).fit(F - mu_f, yj - mu_j).coef_
            pred[:, j] = mu_j + (self._Dk[:, :, j] - mu_f) @ w
        # dimensions -> scalar: least-squares map fitted on the probes only (labels.csv is never read)
        A = np.c_[ds - ds.mean(0), np.ones(len(ds))]
        coef, *_ = np.linalg.lstsq(A, y - mu_y, rcond=None)
        return mu_y + np.c_[pred - ds.mean(0), np.ones(len(pred))] @ coef
