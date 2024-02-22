from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import preprocessing


class GSS(object):
    ## Data preparation by Yanji Du
    def __init__(
        self,
        data_dir,
        years=None,
        n=None,
        balance=0.5,
    ):
        df = pd.read_csv(
            Path(data_dir) / "GSS.dat",
            sep="\s+",
            names=[
                "ballot",
                "age",
                "educ",
                "partyid",
                "polviews",
                "natfare",
                "natfarey",
                "racdif1",
                "racdif2",
                "racdif3",
                "racdif4",
                "year",
                "id",
            ],
        )
        self.n = n
        self.balance = balance
        # Drop years before 1986
        df = df[df["year"] >= 1986].reset_index(drop=True)
        self.all_years = df["year"].unique()
        # If specific years are provided, filter the DataFrame to include only those years
        if years is not None:
            df = df[df["year"].isin(years)].reset_index(drop=True)
        # Drop ballot column
        df.drop("ballot", axis=1, inplace=True)
        # Replace negative values with np.nan
        df.replace([-100, -99, -98, -97], np.nan, inplace=True)
        # Replace partyid = 7 with np.nan ("Other Party")
        df["partyid"] = df["partyid"].replace(7, np.nan)
        # Construct outcome variable
        # natfare or natfarey = 3 means "Too much spent on {welfare, assistance to the poor}" respectively
        df["y"] = df.apply(
            lambda x: (
                1
                if x["natfare"] == 3 or x["natfarey"] == 3
                else (
                    np.nan if np.isnan(x["natfare"]) and np.isnan(x["natfarey"]) else 0
                )
            ),
            axis=1,
        )
        # Construct treatment variable
        # natfare asked 'Welfare'
        # natfarey asked 'Assistance to the poor'
        # 'Welfare' is treatment = 1
        df["w"] = df.apply(
            lambda x: (
                np.nan
                if np.isnan(x["natfare"]) & np.isnan(x["natfarey"])
                else (1 if ~np.isnan(x["natfare"]) else 0)
            ),
            axis=1,
        )
        # racdif1 data: 1 = yes, 2 = no
        # code as yes = 0, no = 1
        df["racdif1_recode"] = df["racdif1"].replace({1: 0, 2: 1})
        # racdif2 data: 1 = yes, 2 = no
        # code as yes = 1, no = 0
        df["racdif2_recode"] = df["racdif2"].replace({1: 1, 2: 0})

        # racdif3 data: 1 = yes, 2 = no
        # code as yes = 0, no = 1
        df["racdif3_recode"] = df["racdif3"].replace({1: 0, 2: 1})

        # racdif4 data: 1 = yes, 2 = no
        # code as yes = 1, no = 0
        df["racdif4_recode"] = df["racdif4"].replace({1: 1, 2: 0})

        df["attblack"] = df[
            ["racdif1_recode", "racdif2_recode", "racdif3_recode", "racdif4_recode"]
        ].mean(axis=1)
        # Select final columns for df
        df = df[
            ["id", "year", "w", "y", "age", "educ", "partyid", "polviews", "attblack"]
        ]
        # Drop any rows with nulls (try doing the analysis without doing this also)
        df = df.dropna(how="any", axis=0).reset_index(drop=True)
        # Convert floats to ints (can do this if we drop all nulls)
        df[["w", "y", "age", "educ", "partyid", "polviews"]] = df[
            ["w", "y", "age", "educ", "partyid", "polviews"]
        ].astype(int)
        numeric_columns = ["age", "educ", "partyid", "polviews", "attblack"]
        self.x = pd.concat(
            [
                df[numeric_columns],
                pd.get_dummies(df["year"], prefix="d"),
            ],
            axis=1,
        )
        self.y = df["y"]
        self.z = df["w"]
        self.df = pd.concat([self.x, self.z, self.y], axis=1)
        # if n is not None and n < len(df):
        #     # Sample n random rows from the DataFrame
        #     random_indices = df.sample(n).index
        #     self.x = self.x.loc[random_indices]
        #     self.y = self.y.loc[random_indices]
        #     self.z = self.z.loc[random_indices]
        self.propensity = self.z.mean()

    def stratified_sample(self, n, balance):
        if n is not None and n < len(self.df):
            # Calculate the number of samples for z == 1 and z == 0 based on balance
            n_1 = int(n * balance)  # Number of samples where z == 1
            n_0 = n - n_1  # Number of samples where z == 0

            # Perform stratified sampling
            sampled_df = pd.concat(
                [
                    self.df[self.df["w"] == 1].sample(
                        n=min(n_1, len(self.df[self.df["w"] == 1]))
                    ),
                    self.df[self.df["w"] == 0].sample(
                        n=min(n_0, len(self.df[self.df["w"] == 0]))
                    ),
                ]
            )

            # Update x, y, and z with sampled indices
            x = self.x.loc[sampled_df.index].to_numpy()
            z = self.z.loc[sampled_df.index].to_numpy()
            y = self.y.loc[sampled_df.index].to_numpy()
            return x, z, y
        else:
            x = self.x.to_numpy()
            z = self.z.to_numpy()
            y = self.y.to_numpy()
            return x, z, y

    def sample(self, seed=None):
        return self.stratified_sample(self.n, self.balance)

    def resample(self):
        return self.stratified_sample(self.n, self.balance)

    def retreat(self):
        return self.stratified_sample(self.n, self.balance)


class Polynomial(object):
    def __init__(
        self,
        n,
        d=1,
        slope=1.0,
        sigma=0.1,
        balance=0.5,
        degree=1,
        beta_seed=None,
    ) -> None:
        self.n = n
        self.d = d
        self.slope = slope
        self.sigma = sigma
        self.balance = balance
        self.degree = degree
        self.xfm = preprocessing.PolynomialFeatures(
            degree=self.degree,
            include_bias=False,
        )
        self.rng_beta = np.random.default_rng(beta_seed)
        _ = self.xfm.fit_transform(np.zeros((1, self.d), dtype="float32"))
        self.beta_0 = -self.slope * self.sample_weights(self.xfm.n_output_features_)
        self.beta_1 = self.slope * self.sample_weights(self.xfm.n_output_features_)

    def sample(self, seed=None):
        self.rng = np.random.default_rng(seed)
        # sample covariate
        self.x = self.rng.uniform(-2, 2, (self.n, self.d)).astype("float32")
        # sample outcome
        self.eps = self.rng.normal(0, self.sigma, size=self.n).astype("float32")
        return self.retreat()

    def resample(self):
        # sample covariate
        self.x = self.rng.uniform(-2, 2, (self.n, self.d)).astype("float32")
        # sample outcome
        self.eps = self.rng.normal(0, self.sigma, size=self.n).astype("float32")
        return self.retreat()

    def retreat(self):
        z = np.zeros(self.n).astype("float32")
        idx_treated = self.rng.choice(
            np.arange(self.n),
            int(self.n * self.balance),
            replace=False,
        )
        z[idx_treated] = 1
        y = z * self.mu1(self.x) + (1 - z) * self.mu0(self.x) + self.eps
        return self.x, z, y

    def mu0(self, x):
        if x is None:
            x = self.x
        X = self.xfm.fit_transform(x)
        return X @ self.beta_0

    def mu1(self, x=None):
        if x is None:
            x = self.x
        X = self.xfm.fit_transform(x)
        return X @ self.beta_1

    @property
    def tau(self):
        return self.mu1(self.x) - self.mu0(self.x)

    def sample_weights(self, d):
        beta = self.rng_beta.uniform(0, 1, d).astype("float32")
        beta /= np.linalg.norm(beta) + 1e-6
        return beta


class SinInteraction(object):
    def __init__(
        self,
        n,
        d=1,
        slope=1.0,
        sigma=0.1,
        beta_t=1.0,
        balance=0.5,
        degree=1,
        sparsity=0.5,
        beta_seed=None,
    ) -> None:
        self.n = n
        self.d = d
        self.slope = slope
        self.sigma = sigma
        self.balance = balance
        self.degree = degree
        self.sparsity = sparsity
        self.beta_t = beta_t
        self.rng_beta = np.random.default_rng(beta_seed)
        zero_indices = self.rng_beta.choice(
            self.d, size=int(self.d * self.sparsity), replace=False
        )
        self.beta_x = self.sample_weights(self.d)
        self.beta_x[zero_indices] = 0.0
        self.beta_x /= np.linalg.norm(self.beta_x) + 1e-6
        self.beta_xz = self.sample_weights(self.d)
        self.beta_xz[zero_indices] = 0.0
        self.beta_xz /= np.linalg.norm(self.beta_xz) + 1e-6
        self.beta_xz *= self.slope
        if self.degree == 2:
            d_xx = d * (d + 1) // 2
            self.beta_xx = self.sample_weights(d_xx)
            self.beta_xxz = self.slope * self.sample_weights(d_xx)

    def sample(self, seed=None):
        self.rng = np.random.default_rng(seed)
        # sample covariate
        self.x = self.rng.uniform(-2, 2, (self.n, self.d)).astype("float32")
        # build response surfaces
        self.mu0 = self.mu(self.x, -np.ones((self.n, 1), dtype="float32"))
        self.mu1 = self.mu(self.x, np.ones((self.n, 1), dtype="float32"))
        assert self.mu0.ndim == 1
        assert self.mu1.ndim == 1
        # sample outcome
        self.eps = self.rng.normal(0, self.sigma, size=self.n)
        return self.retreat()

    def resample(self):
        # sample covariate
        self.x = self.rng.uniform(-2, 2, (self.n, self.d)).astype("float32")
        # build response surfaces
        self.mu0 = self.mu(self.x, -np.ones((self.n, 1), dtype="float32"))
        self.mu1 = self.mu(self.x, np.ones((self.n, 1), dtype="float32"))
        assert self.mu0.ndim == 1
        assert self.mu1.ndim == 1
        # sample outcome
        self.eps = self.rng.normal(0, self.sigma, size=self.n)
        return self.retreat()

    def retreat(self):
        z = np.zeros(self.n).astype("float32")
        idx_treated = self.rng.choice(
            np.arange(self.n),
            int(self.n * self.balance),
            replace=False,
        )
        z[idx_treated] = 1
        y = z * self.mu1 + (1 - z) * self.mu0 + self.eps
        return self.x, z, y

    def mu(self, x, z):
        mu = (
            1
            + self.beta_t * z.ravel()
            + x @ self.beta_x
            + (z * x) @ self.beta_xz
            - 2 * np.sin(2 * (z * x) @ self.beta_xz)
        )
        if self.degree == 2:
            xx = pairwise_interactions(x)
            mu += (
                xx @ self.beta_xx
                + (z * xx) @ self.beta_xxz
                - 2 * np.sin(2 * (z * xx) @ self.beta_xxz)
            )
        return mu

    @property
    def tau(self):
        return self.mu1 - self.mu0

    def sample_weights(self, d):
        beta = self.rng_beta.uniform(0, 1, d).astype("float32")
        return beta


def pairwise_interactions(x):
    n, d = x.shape
    # Calculate the number of pairwise interaction terms
    num_interactions = d * (d + 1) // 2

    # Initialize an array to store the interaction terms
    interactions = np.empty((n, num_interactions))

    # Fill the interactions array with the pairwise products
    k = 0
    for i in range(d):
        for j in range(i, d):
            interactions[:, k] = x[:, i] * x[:, j]
            k += 1

    return interactions
