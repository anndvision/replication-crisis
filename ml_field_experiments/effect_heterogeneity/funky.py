import argparse
from pathlib import Path

import pandas as pd

from ml_field_experiments.datasets import SinInteraction
from ml_field_experiments.effect_heterogeneity.methods.gp import GPTest
from ml_field_experiments.effect_heterogeneity.methods.ols import (
    FTest,
    OLSTest,
    OLSLassoTest,
    OLSInteractTest,
    OLSInteractLassoTest,
)
from ml_field_experiments.effect_heterogeneity.methods.nn import NNTest
from ml_field_experiments.effect_heterogeneity.methods.grf import GRFTest
from ml_field_experiments.effect_heterogeneity.methods.svm import SVRTest

METHODS = {
    "GP": GPTest,
    "OLS": OLSTest,
    "OLS-Lasso": OLSLassoTest,
    "OLS-Interact": OLSInteractTest,
    "OLS-Interact-Lasso": OLSInteractLassoTest,
    "SVM": SVRTest,
    "GRF": GRFTest,
    "NN": NNTest,
    "FTest": FTest,
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
        default=8,
        help="dimension of covariates",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.1,
        help="outcome variance",
    )
    parser.add_argument(
        "--order",
        type=int,
        default=2,
        help="maximum order of interactions",
    )
    parser.add_argument(
        "--beta-t",
        type=float,
        default=0.1,
        help="maximum order of interactions",
    )
    parser.add_argument(
        "--sparsity",
        type=float,
        default=0.5,
        help="sparsity of ground truth response surface parameters",
    )
    parser.add_argument(
        "--balance",
        type=float,
        default=0.5,
        help="probability of treatment",
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
        "order": [],
        "slope": [],
        "var-cate": [],
        "ate": [],
        "p-val-ate": [],
        "p-val-het": [],
        "mse": [],
    }
    method = METHODS[args.method](w_hat=args.balance)
    for slope in [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]:
        dataset = SinInteraction(
            n=args.n,
            d=args.d,
            slope=slope,
            sigma=args.sigma,
            beta_t=args.beta_t,
            balance=args.balance,
            degree=args.order,
            sparsity=args.sparsity,
            beta_seed=args.seed,
        )
        x, z, y = dataset.sample(seed=args.seed)
        results = method.run(x=x, z=z, y=y, tau=dataset.tau)
        r["method"].append(args.method)
        r["n"].append(args.n)
        r["p(t)"].append(args.balance)
        r["order"].append(args.order)
        r["slope"].append(slope)
        r["var-cate"].append(dataset.tau.var())
        r["ate"].append(results["ate"])
        r["p-val-ate"].append(results["p-val-ate"])
        r["p-val-het"].append(results["p-val-het"])
        r["mse"].append(results["mse"])

    df = pd.DataFrame(r)
    exp_str = f"n={args.n}_d={args.d}_sigma={args.sigma}beta-t={args.beta_t}_balance={args.balance}_order={args.order}_sparsity={args.sparsity}_seed={args.seed}"
    output_dir = (
        Path(args.output_dir) / "CATE" / "SinInteraction" / f"{args.method}" / exp_str
    )
    output_dir.mkdir(exist_ok=True, parents=True)
    df.to_csv(output_dir / "results.csv")


if __name__ == "__main__":
    main()
