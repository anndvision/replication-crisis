import numpy as np
import statsmodels.api as sm
from econml import grf

from .core import Test


class GRF(Test):
    def __init__(self, w_hat=0.5, num_estimators=1000, verbose=False) -> None:
        self.num_estimators = num_estimators
        super().__init__(w_hat, verbose)

    def run(self, x, z, y):
        if x.ndim == 1:
            x = x.copy()
            x = x.reshape(-1, 1)
        if z.ndim == 1:
            z = z.copy()
            z = z.ravel()

        n, d = x.shape
        n0 = z.sum()
        n1 = (1 - z).sum()

        max_features = int(min(np.ceil(np.sqrt(d) + 20), d))
        max_samples = 0.5

        m0_model = grf.RegressionForest(
            n_estimators=self.num_estimators,
            max_samples=max_samples,
            max_features=max_features,
            inference=False,
        )
        _ = m0_model.fit(x, y, sample_weight=(1 - z))

        m1_model = grf.RegressionForest(
            n_estimators=self.num_estimators,
            max_samples=max_samples,
            max_features=max_features,
            inference=False,
        )
        _ = m1_model.fit(x, y, sample_weight=z)

        m0 = m0_model.oob_predict(x).ravel()
        m1 = m1_model.oob_predict(x).ravel()

        s = (m1 - m0).mean()
        s += (y[z == 1] - m1[z == 1]).mean()
        s -= (y[z == 0] - m1[z == 0]).mean()
        s /= 2
        y_pred = z * m1 + (1 - z) * m0
        r2 = 1 - (np.square(y - y_pred).sum() / np.square(y - y.mean()).sum())
        adj = (n - 1) / (n - d - 1)
        r2_adj = 1 - (1 - r2) * adj
        var = np.square(y - (n0 / n) * m1 - (n1 / n) * m0)
        var[z == 1] /= n1 * (n1 - 1)
        var[z == 0] /= n0 * (n0 - 1)

        return {
            "ate": s,
            "r2": r2,
            "r2-adj": r2_adj,
            "ate_var": var.sum(),
        }


class CausalForest(Test):
    def __init__(self, w_hat=0.5, num_estimators=1000, verbose=False) -> None:
        self.num_estimators = num_estimators
        super().__init__(w_hat, verbose)

    def run(self, x, z, y):
        if x.ndim == 1:
            x = x.copy()
            x = x.reshape(-1, 1)
        if z.ndim == 1:
            z = z.copy()
            z = z.ravel()

        n, d = x.shape
        n0 = z.sum()
        n1 = (1 - z).sum()

        max_features = int(min(np.ceil(np.sqrt(d) + 20), d))
        max_samples = 0.5

        m0_model = grf.RegressionForest(
            n_estimators=self.num_estimators,
            max_samples=max_samples,
            max_features=max_features,
            inference=False,
        )
        _ = m0_model.fit(x, y, sample_weight=(1 - z))

        s_model = grf.CausalForest(
            n_estimators=self.num_estimators,
            max_samples=max_samples,
            max_features=max_features,
            inference=False,
        )
        _ = s_model.fit(X=x, T=z.reshape(-1, 1), y=y)

        m0 = m0_model.oob_predict(x).ravel()
        s = s_model.oob_predict(x).ravel()
        m1 = s + m0

        y_pred = z * s + m0
        r2 = 1 - (np.square(y - y_pred).sum() / np.square(y - y.mean()).sum())
        adj = (n - 1) / (n - d - 1)
        r2_adj = 1 - (1 - r2) * adj
        var = np.square(y - (n0 / n) * m1 - (n1 / n) * m0)
        var[z == 1] /= n1 * (n1 - 1)
        var[z == 0] /= n0 * (n0 - 1)

        return {
            "ate": s.mean(),
            "r2": r2,
            "r2-adj": r2_adj,
            "ate_var": var.sum(),
        }


class GRFInteract(Test):
    def __init__(self, w_hat=0.5, num_estimators=1000, verbose=False) -> None:
        self.num_estimators = num_estimators
        super().__init__(w_hat, verbose)

    def run(self, x, z, y):
        if x.ndim == 1:
            x = x.copy()
            x = x.reshape(-1, 1)
        if z.ndim == 1:
            z = z.copy()
            z = z.reshape(-1, 1)

        X = np.hstack([z, x])
        n, d = X.shape
        n0 = z.sum()
        n1 = (1 - z).sum()

        max_features = int(min(np.ceil(np.sqrt(d) + 20), d))
        max_samples = 0.5

        m_model = grf.RegressionForest(
            n_estimators=self.num_estimators,
            max_samples=max_samples,
            max_features=max_features,
            inference=False,
        )
        _ = m_model.fit(X, y)

        X0 = np.hstack([np.zeros_like(z), x])
        X1 = np.hstack([np.ones_like(z), x])

        # y_pred = m_model.oob_predict(X).ravel()

        m0 = m_model.oob_predict(X0).ravel()
        m1 = m_model.oob_predict(X1).ravel()
        s = (m1 - m0).mean()

        z = z.ravel()
        y_pred = z * m1 + (1 - z) * m0

        r2 = 1 - (np.square(y - y_pred).sum() / np.square(y - y.mean()).sum())
        adj = (n - 1) / (n - d - 1)
        r2_adj = 1 - (1 - r2) * adj
        var = np.square(y - (n0 / n) * m1 - (n1 / n) * m0)
        var[z == 1] /= n1 * (n1 - 1)
        var[z == 0] /= n0 * (n0 - 1)

        return {
            "ate": s,
            "r2": r2,
            "r2-adj": r2_adj,
            "ate_var": var.sum(),
        }


class GRFAdjusted(Test):
    def __init__(self, w_hat=0.5, num_estimators=1000, verbose=False) -> None:
        self.num_estimators = num_estimators
        super().__init__(w_hat, verbose)

    def run(self, x, z, y):
        if x.ndim == 1:
            x = x.copy()
            x = x.reshape(-1, 1)
        if z.ndim == 1:
            z = z.copy()
            z = z.reshape(-1, 1)

        d = x.shape[-1]

        max_features = int(min(np.ceil(np.sqrt(d) + 20), d))
        max_samples = 0.5

        m_model = grf.RegressionForest(
            n_estimators=self.num_estimators,
            max_samples=max_samples,
            max_features=max_features,
            inference=False,
        )
        _ = m_model.fit(x, y)

        y_pred = m_model.oob_predict(x).ravel()

        results = sm.OLS(y - y_pred, sm.add_constant(z)).fit()

        if self.verbose:
            print(results.summary())
        # return results.params[1]
        variances = np.diag(results.cov_params())

        return {
            "ate": results.params[1],
            "r2": results.rsquared,
            "r2-adj": results.rsquared_adj,
            "ate_var": variances[1],
        }
