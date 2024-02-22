import numpy as np
import statsmodels.api as sm
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

from .core import Test


class GP(Test):
    def __init__(self, w_hat=None, verbose=False) -> None:
        super().__init__(w_hat, verbose)

    def run(self, x, z, y):
        if x.ndim == 1:
            x = x.copy()
            x = x.reshape(-1, 1)
        if z.ndim == 1:
            z = z.copy()
            z = z.reshape(-1, 1)
        n, d = x.shape

        z = 2 * z - 1
        X = np.hstack([z, x])

        kernel = 1.0 * Matern(
            length_scale=np.asarray([1.0] * X.shape[-1]),
            length_scale_bounds=(1e-2, 1e5),
        ) + WhiteKernel(
            noise_level=1,
            noise_level_bounds=(1e-3, 1e3),
        )
        m = GaussianProcessRegressor(
            kernel=kernel,
            alpha=0.0,
            normalize_y=True,
        )
        m.fit(X, y)

        X0 = np.hstack([-np.ones_like(z), x])
        m0 = m.predict(X0).ravel()
        X1 = np.hstack([np.ones_like(z), x])
        m1 = m.predict(X1).ravel()
        s = (m1 - m0).mean()

        y_pred = m.predict(X).ravel()
        r2 = 1 - (np.square(y - y_pred).sum() / np.square(y - y.mean()).sum())
        adj = (n - 1) / (n - d - 1)
        r2_adj = 1 - (1 - r2) * adj

        return {
            "ate": s,
            "r2": r2,
            "r2-adj": r2_adj,
            "ate_var": 0,
        }


class GPSingle(Test):
    def __init__(self, w_hat=None, verbose=False) -> None:
        super().__init__(w_hat, verbose)

    def run(self, x, z, y):
        if x.ndim == 1:
            x = x.copy()
            x = x.reshape(-1, 1)
        if z.ndim == 1:
            z = z.copy()
            z = z.reshape(-1, 1)

        kernel = 1.0 * Matern(
            length_scale=np.asarray([1.0] * x.shape[-1]),
            length_scale_bounds=(1e-2, 1e5),
        ) + WhiteKernel(
            noise_level=1,
            noise_level_bounds=(1e-3, 1e3),
        )
        m = GaussianProcessRegressor(
            kernel=kernel,
            alpha=0.0,
            normalize_y=True,
        )
        m.fit(x, y)

        y_pred = m.predict(x).ravel()

        results = sm.OLS(y - y_pred, sm.add_constant(z)).fit()

        if self.verbose:
            print(results.summary())
        variances = np.diag(results.cov_params())

        return {
            "ate": results.params[1],
            "r2": results.rsquared,
            "r2-adj": results.rsquared_adj,
            "ate_var": variances[1],
        }


class GPClassifier(Test):
    def __init__(self, w_hat=None, verbose=False) -> None:
        self.m = GaussianProcessClassifier(warm_start=True)
        super().__init__(w_hat, verbose)

    def run(self, x, z, y):
        if x.ndim == 1:
            x = x.copy()
            x = x.reshape(-1, 1)
        if z.ndim == 1:
            z = z.copy()
            z = z.reshape(-1, 1)
        n, d = x.shape

        z = 2 * z - 1
        X = np.hstack([z, x])

        if self.m.kernel is None:
            self.m.kernel = 1.0 * Matern(
                length_scale=np.asarray([1.0] * X.shape[-1]),
                length_scale_bounds=(1e-2, 1e5),
            )
        self.m.fit(X, y)

        X0 = np.hstack([-np.ones_like(z), x])
        m0 = self.m.predict_proba(X0)[:, 1]
        X1 = np.hstack([np.ones_like(z), x])
        m1 = self.m.predict_proba(X1)[:, 1]
        s = (m1 - m0).mean()

        y_pred = self.m.predict_proba(X)[:, 1]
        r2 = 1 - (np.square(y - y_pred).sum() / np.square(y - y.mean()).sum())
        adj = (n - 1) / (n - d - 1)
        r2_adj = 1 - (1 - r2) * adj

        return {
            "ate": s,
            "r2": r2,
            "r2-adj": r2_adj,
            "ate_var": 0,
        }


class GPClassifierSingle(Test):
    def __init__(self, w_hat=None, verbose=False) -> None:
        super().__init__(w_hat, verbose)

    def run(self, x, z, y):
        if x.ndim == 1:
            x = x.copy()
            x = x.reshape(-1, 1)
        if z.ndim == 1:
            z = z.copy()
            z = z.reshape(-1, 1)

        kernel = 1.0 * Matern(
            length_scale=np.asarray([1.0] * x.shape[-1]),
            length_scale_bounds=(1e-2, 1e5),
        )
        m = GaussianProcessClassifier(kernel=kernel)
        m.fit(x, y)

        y_pred = m.predict_proba(x)[:, 1]

        results = sm.OLS(y - y_pred, sm.add_constant(z)).fit()

        if self.verbose:
            print(results.summary())
        variances = np.diag(results.cov_params())

        return {
            "ate": results.params[1],
            "r2": results.rsquared,
            "r2-adj": results.rsquared_adj,
            "ate_var": variances[1],
        }
