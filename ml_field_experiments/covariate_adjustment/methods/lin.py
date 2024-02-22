import numpy as np
import statsmodels.api as sm

from .core import Test


class LinRegression(Test):
    def __init__(self, w_hat=None, verbose=False) -> None:
        super().__init__(w_hat, verbose)

    def run(self, x, z, y):
        if x.ndim == 1:
            x = x.copy()
            x = x.reshape(-1, 1)
        if z.ndim == 1:
            z = z.copy()
            z = z.reshape(-1, 1)

        X = sm.add_constant(np.hstack([z, x, z * (x - x.mean(0))]))

        m = sm.OLS(y, X)

        results = m.fit()
        if self.verbose:
            print(results.summary())
        # return results.params[1]
        variances = np.diag(results.cov_params())

        X0 = np.hstack(
            [np.ones_like(z), np.zeros_like(z), x, np.zeros_like(z) * (x - x.mean(0))]
        )
        X1 = np.hstack(
            [np.ones_like(z), np.ones_like(z), x, np.ones_like(z) * (x - x.mean(0))]
        )
        mu0 = results.predict(X0).mean()
        mu1 = results.predict(X1).mean()

        return {
            "ate": results.params[1],
            "r2": results.rsquared,
            "r2-adj": results.rsquared_adj,
            "ate_var": variances[1],
            "mu0": mu0,
            "mu1": mu1,
        }


class LinLassoRegression(Test):
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

        X = sm.add_constant(np.hstack([z, x, z * (x - x.mean(0))]))

        m = sm.OLS(y, X)

        results = m.fit_regularized(
            method="sqrt_lasso",
            L1_wt=1.0,
            refit=True,
        )

        y_pred = results.predict()

        r2 = 1 - (np.square(y - y_pred).sum() / np.square(y - y.mean()).sum())
        adj = (n - 1) / (n - d - 1)
        r2_adj = 1 - (1 - r2) * adj

        return {
            "ate": results.params[1],
            "r2": r2,
            "r2-adj": r2_adj,
            "ate_var": np.nan,
        }
