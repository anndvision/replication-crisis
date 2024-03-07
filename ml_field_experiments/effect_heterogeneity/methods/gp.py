import numpy as np
import statsmodels.api as sm
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.model_selection import KFold
from scipy import stats

from .core import Test


class GPTest(Test):
    def __init__(
        self,
        w_hat=0.5,
        verbose=False,
        num_splits=3,
    ) -> None:
        self.num_splits = num_splits
        super().__init__(verbose=verbose, w_hat=w_hat)

    def run(self, x, z, y, tau=None):
        if x.ndim == 1:
            x = x.copy()
            x = x.reshape(-1, 1)
        if z.ndim == 1:
            z = z.copy()
            z = z.reshape(-1, 1)

        if self.num_splits > 1:

            b = np.zeros_like(z).ravel()
            s = np.zeros_like(z).ravel()

            for train_index, test_index in KFold(
                n_splits=self.num_splits, shuffle=False
            ).split(x):
                X = np.hstack([2 * z[train_index] - 1, x[train_index]])

                kernel_s = 1.0 * Matern(
                    length_scale=np.asarray([1.0] * X.shape[-1]),
                    length_scale_bounds=(1e-2, 1e5),
                ) + WhiteKernel(
                    noise_level=1,
                    noise_level_bounds=(1e-3, 1e3),
                )
                s_model = GaussianProcessRegressor(
                    kernel=kernel_s,
                    alpha=0.0,
                    normalize_y=True,
                )
                s_model.fit(X, y[train_index])

                X0 = np.hstack([-np.ones_like(z[test_index]), x[test_index]])
                m0 = s_model.predict(X0).ravel()
                X1 = np.hstack([np.ones_like(z[test_index]), x[test_index]])
                m1 = s_model.predict(X1).ravel()

                b[test_index] = m0
                s[test_index] = m1 - m0
        else:
            X = np.hstack([2 * z - 1, x])

            kernel_s = 1.0 * Matern(
                length_scale=np.asarray([1.0] * X.shape[-1]),
                length_scale_bounds=(1e-2, 1e5),
            ) + WhiteKernel(
                noise_level=1,
                noise_level_bounds=(1e-3, 1e3),
            )
            s_model = GaussianProcessRegressor(
                kernel=kernel_s,
                alpha=0.0,
                normalize_y=True,
            )
            s_model.fit(X, y)

            X0 = np.hstack([-np.ones_like(z), x])
            m0 = s_model.predict(X0).ravel()
            X1 = np.hstack([np.ones_like(z), x])
            m1 = s_model.predict(X1).ravel()

            b = m0
            s = m1 - m0

        z = z.ravel()

        X = np.stack(
            [
                b,
                self.w_hat * s,
                (z - self.w_hat),
                (z - self.w_hat) * (s.ravel() - s.mean()),
            ]
        ).T

        m = sm.OLS(y, sm.add_constant(X))
        results = m.fit().get_robustcov_results()

        if self.verbose:
            print(results.summary())
            print(s.var())

        p_val_ate = results.pvalues[-2]
        p_val_het = 1 - stats.t.cdf(results.tvalues[-1], results.df_resid)

        if tau is not None:
            mse = np.square(tau.ravel() - s).mean()
        else:
            mse = np.nan

        return {
            "ate": results.params[-2],
            "p-val-ate": p_val_ate,
            "p-val-het": p_val_het,
            "mse": mse,
            "CATE": s,
        }


class GPClassifierTest(Test):
    def __init__(
        self,
        w_hat=0.5,
        verbose=False,
        num_splits=3,
    ) -> None:
        self.num_splits = num_splits
        super().__init__(verbose=verbose, w_hat=w_hat)

    def run(self, x, z, y, tau=None):
        if x.ndim == 1:
            x = x.copy()
            x = x.reshape(-1, 1)
        if z.ndim == 1:
            z = z.copy()
            z = z.reshape(-1, 1)

        if self.num_splits > 1:

            b = np.zeros_like(z).ravel()
            s = np.zeros_like(z).ravel()

            kernel_s = 1.0 * Matern(
                length_scale=np.asarray([1.0] * (x.shape[-1] + 1)),
                length_scale_bounds=(1e-2, 1e5),
            )
            s_model = GaussianProcessClassifier(
                kernel=kernel_s,
                n_jobs=-1,
                warm_start=True,
            )

            for train_index, test_index in KFold(
                n_splits=self.num_splits, shuffle=False
            ).split(x):
                X = np.hstack([2 * z[train_index] - 1, x[train_index]])
                s_model.fit(X, y[train_index])

                X0 = np.hstack([-np.ones_like(z[test_index]), x[test_index]])
                m0 = s_model.predict_proba(X0)[:, 1]
                X1 = np.hstack([np.ones_like(z[test_index]), x[test_index]])
                m1 = s_model.predict_proba(X1)[:, 1]

                b[test_index] = m0
                s[test_index] = m1 - m0

        z = z.ravel()

        X = np.stack(
            [
                b,
                self.w_hat * s,
                (z - self.w_hat),
                (z - self.w_hat) * (s.ravel() - s.mean()),
            ]
        ).T

        m = sm.OLS(y, sm.add_constant(X))
        results = m.fit().get_robustcov_results()

        if self.verbose:
            print(results.summary())
            print(s.var())

        p_val_ate = results.pvalues[-2]
        p_val_het = 1 - stats.t.cdf(results.tvalues[-1], results.df_resid)

        if tau is not None:
            mse = np.square(tau.ravel() - s).mean()
        else:
            mse = np.nan

        return {
            "ate": results.params[-2],
            "p-val-ate": p_val_ate,
            "p-val-het": p_val_het,
            "mse": mse,
            "CATE": s,
        }
