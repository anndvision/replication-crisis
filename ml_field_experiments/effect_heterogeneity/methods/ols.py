import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn import preprocessing
from scipy import stats

from .core import Test


class Lasso(object):

    def __init__(
        self,
        alpha=1.0,
        fit_intercept=True,
    ) -> None:
        self.m = None
        self.alpha = alpha
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        X = X.astype("float64")
        y = y.astype("float64")
        if self.fit_intercept:
            X = sm.add_constant(X)
        self.m = sm.OLS(
            y,
            X,
        ).fit_regularized(
            method="sqrt_lasso",
            L1_wt=self.alpha,
            refit=True,
        )

    def predict(self, X):
        if self.fit_intercept:
            X = np.hstack([np.ones_like(X[:, :1]), X])
        return self.m.predict(X)


class OLSTest(Test):
    def __init__(
        self,
        w_hat=0.5,
        verbose=False,
        num_splits=10,
        alpha=0.0,
        stratify=False,
    ) -> None:
        self.num_splits = num_splits
        self.alpha = alpha
        self.stratify = stratify
        super().__init__(verbose=verbose, w_hat=w_hat)

    def run(self, x, z, y, tau=None):
        if x.ndim == 1:
            x = x.copy()
            x = x.reshape(-1, 1)
        if z.ndim == 2:
            z = z.copy()
            z = z.ravel()

        b = np.zeros_like(z).ravel()
        s = np.zeros_like(z).ravel()

        if self.stratify:
            splits = StratifiedKFold(n_splits=self.num_splits, shuffle=False).split(
                x, y
            )
        else:
            splits = KFold(n_splits=self.num_splits, shuffle=False).split(x)

        for train_index, test_index in splits:
            b_model = (
                LinearRegression() if self.alpha == 0.0 else Lasso(alpha=self.alpha)
            )
            b_model.fit(x[train_index], y[train_index])
            b[test_index] = b_model.predict(x[test_index])

            m0_model = (
                LinearRegression() if self.alpha == 0.0 else Lasso(alpha=self.alpha)
            )
            m0_model.fit(
                x[train_index][z[train_index] == 0], y[train_index][z[train_index] == 0]
            )
            m0 = m0_model.predict(x[test_index])

            m1_model = (
                LinearRegression() if self.alpha == 0.0 else Lasso(alpha=self.alpha)
            )
            m1_model.fit(
                x[train_index][z[train_index] == 1], y[train_index][z[train_index] == 1]
            )
            m1 = m1_model.predict(x[test_index])

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
        p_val_het = (
            1 - stats.t.cdf(results.tvalues[-1], results.df_resid)
            if s.var() > 1e-5
            else 1.0
        )

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


class OLSLassoTest(OLSTest):

    def __init__(
        self,
        w_hat=0.5,
        verbose=False,
        num_splits=10,
        alpha=1.0,
        stratify=False,
    ) -> None:
        super().__init__(
            verbose=verbose,
            w_hat=w_hat,
            num_splits=num_splits,
            alpha=alpha,
            stratify=stratify,
        )


class OLSInteractTest(Test):

    def __init__(
        self,
        w_hat=0.5,
        verbose=False,
        num_splits=10,
        alpha=0.0,
        stratify=False,
    ) -> None:
        self.num_splits = num_splits
        self.alpha = alpha
        self.stratify = stratify
        super().__init__(verbose=verbose, w_hat=w_hat)

    def run(self, x, z, y, tau=None):
        if x.ndim == 1:
            x = x.copy()
            x = x.reshape(-1, 1)
        if z.ndim == 1:
            z = z.copy()
            z = z.reshape(-1, 1)

        b = np.zeros_like(z).ravel()
        s = np.zeros_like(z).ravel()

        if self.stratify:
            splits = StratifiedKFold(n_splits=self.num_splits, shuffle=False).split(
                x, y
            )
        else:
            splits = KFold(n_splits=self.num_splits, shuffle=False).split(x)

        for train_index, test_index in splits:
            b_model = (
                LinearRegression(fit_intercept=True)
                if self.alpha == 0.0
                else Lasso(alpha=self.alpha, fit_intercept=True)
            )
            b_model.fit(x[train_index], y[train_index])
            b[test_index] = b_model.predict(x[test_index])

            X = np.hstack(
                [z[train_index], x[train_index], z[train_index] * x[train_index]]
            )
            s_model = (
                LinearRegression(fit_intercept=True)
                if self.alpha == 0.0
                else Lasso(alpha=self.alpha, fit_intercept=True)
            )
            s_model.fit(X, y[train_index])

            z0 = np.zeros_like(z[test_index])
            X0 = np.hstack([z0, x[test_index], z0 * x[test_index]])
            m0 = s_model.predict(X0).ravel()
            z1 = np.ones_like(z[test_index])
            X1 = np.hstack([z1, x[test_index], z1 * x[test_index]])
            m1 = s_model.predict(X1).ravel()

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
        p_val_het = (
            1 - stats.t.cdf(results.tvalues[-1], results.df_resid)
            if s.var() > 1e-5
            else 1.0
        )

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


class OLSInteractLassoTest(OLSInteractTest):

    def __init__(
        self,
        w_hat=0.5,
        verbose=False,
        num_splits=10,
        alpha=1.0,
        stratify=False,
    ) -> None:
        super().__init__(
            verbose=verbose,
            w_hat=w_hat,
            num_splits=num_splits,
            alpha=alpha,
            stratify=stratify,
        )


class FTest(Test):
    def __init__(
        self,
        w_hat=0.5,
        verbose=False,
        order=1,
    ) -> None:
        self.order = order
        super().__init__(
            verbose=verbose,
            w_hat=w_hat,
        )

    def run(self, x, z, y, tau=None):
        warnings.filterwarnings("error")
        if x.ndim == 1:
            x = x.copy()
            x = x.reshape(-1, 1)
        if z.ndim == 1:
            z = z.copy()
            z = z.reshape(-1, 1)

        n_vars = x.shape[-1]
        names = ["z"] + [f"x{i}" for i in range(1, n_vars + 1)]

        df = pd.DataFrame(np.hstack([z, x]), columns=names)

        basis = preprocessing.PolynomialFeatures(
            degree=self.order + 1, interaction_only=True, include_bias=True
        )
        X = pd.DataFrame(
            basis.fit_transform(df),
            columns=basis.get_feature_names_out(),
        )
        # interaction_features = []
        # for col in X.columns:
        #     if ("x" in col) and ("z" in col):
        #         interaction_features.append(col)

        # to_drop = drop_highly_correlated_features(X[interaction_features])
        # X = X.drop(columns=to_drop, axis=1)

        results = sm.OLS(y, X).fit()
        try:
            hypotheses = ""
            num_constraints = 0
            for k in results.params.keys():
                if ("x" in k) and ("z" in k):
                    hypotheses += f"({k} = 0), "
                    num_constraints += 1
            hypotheses = hypotheses[:-2]
            p_val_het = results.f_test(hypotheses).pvalue
        except:
            return OLSTest(w_hat=self.w_hat).run(x=x, z=z, y=y, tau=tau)

        return {
            "ate": np.nan,
            "p-val-ate": np.nan,
            "p-val-het": p_val_het,
            "mse": np.nan,
            "CATE": np.nan,
        }


def drop_highly_correlated_features(df, threshold=0.99):
    """
    Drop features from the DataFrame that are highly correlated to others.

    Parameters:
    - df: pandas DataFrame containing the features.
    - threshold: float, the correlation threshold to identify highly correlated features (default is 0.9).

    Returns:
    - df_reduced: DataFrame with the highly correlated features dropped.
    """
    # Calculate the correlation matrix and get the absolute value
    corr_matrix = df.corr().abs()

    # Identify features that are highly correlated to others
    to_drop = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (
                corr_matrix.iloc[i, j] >= threshold
            ):  # If correlation is above the threshold
                to_drop.add(corr_matrix.columns[i])  # Add the feature to the drop list

    return to_drop
