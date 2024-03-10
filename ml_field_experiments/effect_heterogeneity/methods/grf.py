import numpy as np
import statsmodels.api as sm
from econml.grf import RegressionForest
from scipy import stats
from sklearn.model_selection import train_test_split

from .core import Test


class GRFTest(Test):
    def __init__(
        self,
        w_hat=0.5,
        verbose=False,
        num_estimators=1000,
    ) -> None:
        self.num_estimators = num_estimators
        super().__init__(verbose=verbose, w_hat=w_hat)

    def run(self, x, z, y, tau=None):
        if x.ndim == 1:
            x = x.copy()
            x = x.reshape(-1, 1)
        if z.ndim == 1:
            z = z.copy()
            z = z.reshape(-1, 1)

        max_samples = 0.5
        max_features = int(min(np.ceil(np.sqrt(x.shape[-1]) + 20), x.shape[-1]))

        b_model = RegressionForest(
            n_estimators=max(50, self.num_estimators // 4),
            max_samples=max_samples,
            inference=False,
        )
        s_model = RegressionForest(
            n_estimators=self.num_estimators,
            max_samples=max_samples,
            max_features=max_features,
            inference=False,
        )

        _ = b_model.fit(x, y)
        _ = s_model.fit(np.hstack([z, x]), y)

        b = b_model.oob_predict(x).ravel()
        m0 = s_model.oob_predict(np.hstack([np.zeros_like(z), x])).ravel()
        m1 = s_model.oob_predict(np.hstack([np.ones_like(z), x])).ravel()

        z = z.ravel()

        s = m1 - m0
        s += (y[z == 1] - m1[z == 1]).mean()
        s -= (y[z == 0] - m0[z == 0]).mean()

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
        if self.verbose == True:
            print(results.summary())

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
