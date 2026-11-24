"""M2 baseline (ridge_K): fit a prompt -> quality ridge for the new arm on its K probes alone,
without borrowing anything from the known arms."""
from methods.base import Method as Base
from protocol import default_X, ridge


class Method(Base):
    def predict_star(self, ctx, probe, y, d):
        Xtr = default_X(ctx.feats_train, slice(None))[probe]
        return ridge(Xtr, y[:, None], default_X(ctx.feats_eval, slice(None)))[:, 0]
