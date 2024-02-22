import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from ml_field_experiments.datasets import GSS
from ml_field_experiments.covariate_adjustment.methods.gp import GPClassifier, GP
from ml_field_experiments.covariate_adjustment.methods.regression import (
    DIM,
    LassoRegression,
    Regression,
)
from ml_field_experiments.covariate_adjustment.methods.lin import (
    LinRegression,
    LinLassoRegression,
)
from ml_field_experiments.covariate_adjustment.methods.grf import GRF, CausalForest

METHODS = {
    "GP": GP,
    "GPC": GPClassifier,
    "OLS": Regression,
    "OLS-Lasso": LassoRegression,
    "Lin": LinRegression,
    "Lin-Lasso": LinLassoRegression,
    "DIM": DIM,
    "GRF": GRF,
    "CausalForest": CausalForest,
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
        "--data-dir",
        type=str,
        default="data/",
        help="location of dataset",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=2000,
        help="size of sample",
    )
    parser.add_argument(
        "--balance",
        type=float,
        default=0.5,
        help="probability of treatment",
    )
    parser.add_argument(
        "--num-reps",
        type=int,
        default=32,
        help="number of replications",
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
    ds_full = GSS(
        data_dir=args.data_dir,
        balance=args.balance,
    )
    ate_true = ds_full.y[ds_full.z == 1].mean() - ds_full.y[ds_full.z == 0].mean()
    method = METHODS[args.method](w_hat=args.balance)
    dataset = GSS(
        data_dir=args.data_dir,
        n=args.n,
        balance=args.balance,
    )
    x, z, y = dataset.sample(seed=args.seed)
    ates = []
    for _ in range(args.num_reps):
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
    output_dir = Path(args.output_dir) / "ATE" / "GSS" / f"{args.method}" / exp_str
    output_dir.mkdir(exist_ok=True, parents=True)
    df.to_csv(output_dir / "results.csv")
    df_sem.to_csv(output_dir / "results_sem.csv")


if __name__ == "__main__":
    main()
