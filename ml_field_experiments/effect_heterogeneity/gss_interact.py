import argparse
from pathlib import Path

import pandas as pd

from ml_field_experiments.datasets import GSSInteraction
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
        "--order",
        type=int,
        default=1,
        help="maximum order of interactions",
    )
    parser.add_argument(
        "--sparsity",
        type=float,
        default=0.0,
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
        "sigma-beta": [],
        "var-cate": [],
        "ate": [],
        "p-val-ate": [],
        "p-val-het": [],
        "mse": [],
    }
    if args.method == "FTest":
        method = FTest(w_hat=args.balance, order=args.order)
    else:
        method = METHODS[args.method](w_hat=args.balance)
    for sigma_beta in [0.0, 0.2, 0.5, 1.0, 2.0, 4.0]:
        dataset = GSSInteraction(
            data_dir=args.data_dir,
            n=args.n,
            degree=args.order,
            sparsity=args.sparsity,
            sigma_beta=sigma_beta,
            sigma_y=1.0,
            balance=args.balance,
        )
        x, z, y = dataset.sample(seed=args.seed)
        results = method.run(x=x, z=z, y=y, tau=dataset.tau)
        r["method"].append(args.method)
        r["n"].append(args.n)
        r["p(t)"].append(args.balance)
        r["order"].append(args.order)
        r["sigma-beta"].append(sigma_beta)
        r["var-cate"].append(dataset.tau.var())
        r["ate"].append(results["ate"])
        r["p-val-ate"].append(results["p-val-ate"])
        r["p-val-het"].append(results["p-val-het"])
        r["mse"].append(results["mse"])

    df = pd.DataFrame(r)
    exp_str = f"n={args.n}_balance={args.balance}_order={args.order}_sparsity={args.sparsity}_seed={args.seed}"
    output_dir = (
        Path(args.output_dir) / "CATE" / "GSSInteract" / f"{args.method}" / exp_str
    )
    output_dir.mkdir(exist_ok=True, parents=True)
    df.to_csv(output_dir / "results.csv")


if __name__ == "__main__":
    main()
