import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn import preprocessing

from ml_field_experiments.datasets import Polynomial
from ml_field_experiments.covariate_adjustment.methods.gp import GP, GPSingle
from ml_field_experiments.covariate_adjustment.methods.regression import (
    DIM,
    Regression,
)
from ml_field_experiments.covariate_adjustment.methods.lin import (
    LinRegression,
)
from ml_field_experiments.covariate_adjustment.methods.grf import (
    GRFInteract,
    GRFAdjusted,
    CausalForest,
    GRF,
)

METHODS = {
    "OLS-Adjusted": Regression,
    "OLS-Interact": LinRegression,
    "OLS-Unadjusted": DIM,
    "CausalForest": CausalForest,
    "GRF": GRF,
    "GRF-Interact": GRFInteract,
    "GRF-Adjusted": GRFAdjusted,
    "GP-Interact": GP,
    "GP-Adjusted": GPSingle,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        help="ATE estimation method",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/",
        help="base directory to write results",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=2000,
        help="size of sample",
    )
    parser.add_argument(
        "--d",
        type=int,
        default=1,
        help="dimension of covariates",
    )
    parser.add_argument(
        "--slope",
        type=float,
        default=0.1,
        help="degree of heterogeneity",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.1,
        help="outcome variance",
    )
    parser.add_argument(
        "--balance",
        type=float,
        default=0.5,
        help="probability of treatment",
    )
    parser.add_argument(
        "--degree",
        type=int,
        default=3,
        help="degree of response surface polynomials",
    )
    parser.add_argument(
        "--num-reps",
        type=int,
        default=100,
        help="number of replications",
    )
    parser.add_argument(
        "--beta-seed",
        type=int,
        default=1331,
        help="random seed for response surface parameters",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1331,
        help="random seed",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    r = {
        "method": [],
        "n": [],
        "p(t)": [],
        "ate_var": [],
        "ate": [],
        "r2": [],
        "r2-adj": [],
    }
    r_sem = {
        "n": [],
        "p(t)": [],
        "ate_se": [],
        "bias": [],
        "variance": [],
        "method": [],
    }
    ds_full = Polynomial(
        n=int(1e6),
        d=args.d,
        slope=args.slope,
        sigma=args.sigma,
        balance=args.balance,
        degree=args.degree,
        beta_seed=args.beta_seed,
    )
    x, z, y = ds_full.sample(seed=args.seed)
    ate_true = ds_full.tau.mean()
    method = METHODS[args.method](w_hat=args.balance)
    dataset = Polynomial(
        n=args.n,
        d=args.d,
        slope=args.slope,
        sigma=args.sigma,
        balance=args.balance,
        degree=args.degree,
        beta_seed=args.beta_seed,
    )
    x, z, y = dataset.sample(seed=args.seed)
    ates = []
    for _ in range(args.num_reps):
        if args.method in ["OLS-Adjusted", "OLS-Interact"]:
            xfm = preprocessing.PolynomialFeatures(
                degree=args.degree,
                include_bias=False,
            )
            x = xfm.fit_transform(x)
        results = method.run(x=x, z=z, y=y)
        r["n"].append(args.n)
        r["p(t)"].append(args.balance)
        r["ate_var"].append(results["ate_var"])
        r["ate"].append(results["ate"])
        ates.append(results["ate"])
        r["r2"].append(results["r2"])
        r["r2-adj"].append(results["r2-adj"])
        r["method"].append(args.method)
        x, z, y = dataset.resample()
    r_sem["n"].append(args.n)
    r_sem["p(t)"].append(args.balance)
    r_sem["ate_se"].append(stats.sem(ates))
    r_sem["bias"].append(np.square(np.mean(ates) - ate_true))
    r_sem["variance"].append(np.var(ates))
    r_sem["method"].append(args.method)

    df = pd.DataFrame(r)
    df_sem = pd.DataFrame(r_sem)
    exp_str = (
        f"n={args.n}_balance={args.balance}_num-reps={args.num_reps}_seed={args.seed}"
    )
    output_dir = (
        Path(args.output_dir) / "ATE" / "Polynomial" / f"{args.method}" / exp_str
    )
    output_dir.mkdir(exist_ok=True, parents=True)
    df.to_csv(output_dir / "results.csv")
    df_sem.to_csv(output_dir / "results_sem.csv")


if __name__ == "__main__":
    main()
