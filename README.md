# ml-field-experiments

## installation
```.sh
$ git clone git@github.com:anndvision/ml-field-experiments.git
$ cd ml-field-experiemnts
$ conda env create -f environment.yaml
$ conda activate ml-field-experiments
$ pip install -e .
```


## Covariate Adjustment

### Run some simulations
To run a Lin regression simulation on a subset of the GSS data
```.sh
python3 ml_field_experiments/covariate_adjustment/gss.py --method Lin --output-dir experiments/ --n 2000 --balance 0.5 --num-reps 100
```

To run a GP simulation using the polynomial dataset (note Regression and Lin methods get access to true polynomial form)
```.sh
python3 ml_field_experiments/covariate_adjustment/polynomial.py --method GP-Interact --output-dir experiments/ --n 500 --balance 0.5 --num-reps 100
```

To run a GRF with one response surface simulation on the non-linear funky data
```.sh
python3 ml_field_experiments/covariate_adjustment/funky.py --method GRF-Adjusted --output-dir experiments/ --n 1000 --balance 0.5 --num-reps 100
```

### Batch deployment of experiments
see the `slurm/commands.txt` file for a full battery of simulations.

see `notebooks/evaluate_ate.ipynb` for plotting examples.
