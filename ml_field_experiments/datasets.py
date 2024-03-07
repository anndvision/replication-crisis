from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import preprocessing


class Facebook(object):
    def __init__(
        self,
        data_dir,
        n=None,
    ):
        continuous_covariates = [
            "age",  # Age
            "hhld_inc",  # Household Income
            "educyears",  # Years of Education
            "repdem",  # Scale of party identification [0 (dem), 1 (rep)]
            "fb_active",  # Active minutes on FB
            "fb_activepassive",  # Active and passive minutes on FB
            "fb_minutes_prescreen",  # FB minutes at pre-screen
        ]
        binary_covariates = [
            "female",  # Identify as femal
            "male",  # Identify as male
            "white",  # Identify as white
            "black",  # Identify as black
            "republican",  # Identify as republican
            "democrat",  # Identify as democrat
            "hhld_inc_under50",  # Household income under $50,000
            "college",  # [Attended/Graduated] College
            "ageunder30",  # Under 30 years old
            "agenda",  # Believes experimenters have an agenda
            "opinion_b",  # Has positive opinion of facebook
            "news_fb",  # News from facebook above median
            "minutes_fb",  # Minutes on facebook above median
            "age_dummy",  # Age above median
        ]
        data_path = Path(data_dir) / "final_data.csv"
        df = pd.read_csv(data_path, index_col=0, low_memory=False)
        in_sample = df["sample_main"] == 1
        outcome_family = {"swb": "Subjective well-being"}
        outcome_names = {
            "swb_happy": "Happiness",
            "swb_swl": "Life satisfaction",
            "swb_lns": "Loneliness",
            "swb_eurhappsvy_4": "Depressed",
            "swb_eurhappsvy_5": "Anxious",
            "swb_eurhappsvy_6": "Absorbed",
            "swb_eurhappsvy_7": "Bored",
            "happy_sms_summary": "SMS happiness",
            "pos_emotion_sms_summary": "SMS positive emotion",
            "lonely_sms_summary": "SMS not lonely",
        }
        k = "index_swb"
        this_sample = in_sample & df.loc[in_sample, k].notna()
        self.y = df.loc[this_sample, k].values
        self.d = df.loc[this_sample, "D"].values
        self.z = df.loc[this_sample, "T"].values

        X_cont = df.loc[this_sample, continuous_covariates].copy().astype("float32")
        X_cont = (X_cont - X_cont.mean(0)) / (X_cont.std(0) + 1e-6)
        X_bin = df.loc[this_sample, binary_covariates].copy().astype("float32")
        X_bin = X_bin.fillna(0)
        X_bin = 2 * X_bin - 1
        # Get outcome baselines
        # baselines = [f"{k}_b"]
        baselines = []
        for k in outcome_names.keys():
            baselines.append(f"{k}_b")
        X_baseline = df.loc[this_sample, baselines].copy().astype("float32")
        X_baseline = (X_baseline - X_baseline.mean(0)) / (X_baseline.std(0) + 1e-6)
        self.x = np.hstack(
            [
                X_baseline.values,
                X_cont.values,
                X_bin.values,
            ]
        )
        self.n = n

    def sample(self, seed=None):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(self.y), size=self.n, replace=False)
        return self.x[idx], self.z[idx], self.y[idx]


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
            x = self.x.loc[sampled_df.index].to_numpy().astype("float32")
            z = self.z.loc[sampled_df.index].to_numpy().astype("float32")
            y = self.y.loc[sampled_df.index].to_numpy().astype("float32")
            return x, z, y
        else:
            x = self.x.to_numpy().astype("float32")
            z = self.z.to_numpy().astype("float32")
            y = self.y.to_numpy().astype("float32")
            return x, z, y

    def sample(self, seed=None):
        return self.stratified_sample(self.n, self.balance)

    def resample(self):
        return self.stratified_sample(self.n, self.balance)

    def retreat(self):
        return self.stratified_sample(self.n, self.balance)


class GSSInteraction(object):
    def __init__(
        self,
        data_dir,
        n=-1,
        degree=1,
        sparsity=0.0,
        sigma_beta=1.0,
        sigma_y=1.0,
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
        # Drop years before 1986
        df = df[df["year"] >= 1986].reset_index(drop=True)
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
        numeric_columns = ["year", "age", "educ", "partyid", "polviews", "attblack"]
        df = df[numeric_columns]
        scaler = preprocessing.StandardScaler()
        self.x = scaler.fit_transform(df)
        self.sparsity = sparsity
        self.sigma_beta = sigma_beta
        self.sigma_y = sigma_y
        self.basis = preprocessing.PolynomialFeatures(
            degree=degree, interaction_only=True, include_bias=False
        )
        self.balance = balance
        self.n = n

    def sample(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.beta_0 = self.rng.normal(0, 1.0)
        self.beta_t = 0.3
        idx_sample = self.rng.choice(np.arange(len(self.x)), self.n, replace=False)
        z = np.zeros(self.n).astype("float32")
        idx_treated = self.rng.choice(
            np.arange(self.n), int(self.n * self.balance), replace=False
        )
        z[idx_treated] = 1

        X = self.basis.fit_transform(self.x[idx_sample])

        self.beta_x = self.rng.normal(0, 1, size=X.shape[-1])
        self.beta_xz = self.rng.normal(0, 1, size=X.shape[-1])

        if self.sparsity > 0.0:
            non_zeros = self.rng.binomial(
                1, p=1.0 - self.sparsity, size=(1, self.x.shape[-1])
            ).astype("float32")
            non_zeros = self.basis.fit_transform(non_zeros).ravel()
            self.beta_x *= non_zeros
            self.beta_xz *= non_zeros

        self.beta_x /= np.linalg.norm(self.beta_x) + 1e-6
        self.beta_xz /= np.linalg.norm(self.beta_xz) + 1e-6
        self.beta_xz *= self.sigma_beta
        self.mu0 = self.beta_0 + X @ self.beta_x
        self.mu1 = self.beta_0 + self.beta_t + X @ self.beta_x + X @ self.beta_xz
        self.tau = self.mu1 - self.mu0
        self.eps = self.rng.normal(0, self.sigma_y, size=len(z))
        y = z * self.mu1 + (1 - z) * self.mu0 + self.eps
        return self.x[idx_sample], z, y

    def retreat(self):
        z = np.zeros(self.n).astype("float32")
        idx_treated = self.rng.choice(
            np.arange(self.n), int(self.n * self.balance), replace=False
        )
        z[idx_treated] = 1
        y = z * self.mu1 + (1 - z) * self.mu0 + self.eps
        return self.x, z, y


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
        # sample
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
        mu = 1 + self.beta_t * z.ravel() + x @ self.beta_x
        if self.slope != 0:
            mu += (z * x) @ self.beta_xz - 2 * np.sin(2 * (z * x) @ self.beta_xz)
        if self.degree == 2:
            xx = pairwise_interactions(x)
            mu += xx @ self.beta_xx
            if self.slope != 0:
                mu += (z * xx) @ self.beta_xxz - 2 * np.sin(
                    2 * (z * xx) @ self.beta_xxz
                )

        return mu

    @property
    def tau(self):
        return self.mu(self.x, np.ones_like(self.x[:, :1])) - self.mu(
            self.x, np.zeros_like(self.x[:, :1])
        )

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


class BinomialFactorial(object):
    def __init__(
        self,
        n,
        p_t=0.5,
        k=1,
        sigma=0.1,
        order=1,
        sparsity=0.5,
        beta_seed=42,
    ) -> None:
        self.n = n
        self.p_t = p_t
        self.k = k
        self.sigma = sigma
        self.order = order
        self.sparsity = sparsity
        self.beta_seed = beta_seed
        # initialize beta random number generator
        self.rng_beta = np.random.default_rng(beta_seed)
        # initialize interaction expansion transformation
        self.xfm = preprocessing.PolynomialFeatures(
            degree=self.order + 1, interaction_only=True, include_bias=True
        )
        _ = self.xfm.fit_transform(np.zeros((1, self.k), dtype="float32"))
        # sample ground truth betas
        self.beta = self.rng_beta.normal(0, 1, self.xfm.n_output_features_).astype(
            "float32"
        )
        zero_indices = self.rng_beta.choice(
            self.xfm.n_output_features_,
            size=int(self.xfm.n_output_features_ * self.sparsity),
            replace=False,
        )
        self.beta[zero_indices] = 0.0

    def sample(self, seed=None):
        self.rng = np.random.default_rng(seed)
        # sample treatment array
        t = self.rng.binomial(1, self.p_t, (self.n, self.k)).astype("float32")
        # expand treatment array
        T = self.xfm.fit_transform(t)
        # build response surface
        self.mu = T @ self.beta
        # sample outcome
        self.eps = self.rng.normal(0, self.sigma, size=self.n)
        y = self.mu + self.eps
        return t, y
