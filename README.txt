Python replication (demand-side / Table 5) — how to run
=======================================================

Notes
-----
1) This repository does NOT ship with `data/` or `logs/`.
   Create them locally and place the required input files under `data/`.
2) Do NOT run `scripts/*.py` directly, because they use package-relative imports.
   Always use `python -m scripts.run_estimation` (or `python -m scripts.run_estimation_ip`).
3) The default paths inside the scripts assume a different project layout.
   To avoid path issues, always pass `--data/--initial/--output` explicitly (examples below).


0) Create a virtual environment + install dependencies
------------------------------------------------------

From the repo root (this folder):

  python -m venv .venv
  source .venv/bin/activate
  python -m pip install -U pip

Minimum:
  pip install numpy pandas scipy tqdm

Recommended (parquet + speed):
  pip install pyarrow numba

If you cannot install pyarrow, use CSV instead of parquet for `--data`.


1) Prepare local inputs
-----------------------

Create folders:
  mkdir -p data logs

Place your inputs under `data/`:
  data/preparedata_new.parquet    (or .csv)
  data/initial_value.csv          (or .mat)

Optional (for closer MATLAB alignment):
  data/randcustsample_seed123.csv
  data/eps_draws_seed123.csv


2) Quick sanity run (small / fast)
----------------------------------

This is a "does it run" check. It is NOT Table 5.

  python -m scripts.run_estimation \
    --data data/preparedata_new.parquet \
    --initial data/initial_value.csv \
    --output data/output_sanity.csv \
    --seed 123 \
    --rng numpy \
    --n-customers 2000 \
    --max-hotels 10 \
    --eps-rows 50 \
    --eps-cols 100 \
    --maxiter 200 \
    --maxfun 200 \
    --n-jobs 1 \
    --no-numba \
    --progress \
    --progress-every 5


3) Table 5 replication (recommended baseline)
---------------------------------------------

This mirrors the long-run settings used in our local helper (run.sh), but we do NOT
ship run.sh to GitHub. Prefer running this via `nohup` on a server.

  nohup python -m scripts.run_estimation \
    --data data/preparedata_new.parquet \
    --initial data/initial_value.csv \
    --output data/output_table5.csv \
    --seed 123 \
    --rng matlab \
    --n-customers 10000 \
    --max-hotels 15 \
    --eps-rows 100 \
    --eps-cols 500 \
    --maxiter 90000 \
    --maxfun 90000 \
    --n-jobs 16 \
    --numba \
    --progress \
    --progress-every 5 \
    > logs/run_estimation_table5.log 2>&1 &


4) Table 5 replication with MATLAB-provided sample + eps draws (closest match)
------------------------------------------------------------------------------

  nohup python -m scripts.run_estimation \
    --data data/preparedata_new.parquet \
    --initial data/initial_value.csv \
    --output data/output_table5_matlab_sample.csv \
    --seed 123 \
    --rng matlab \
    --n-customers 10000 \
    --max-hotels 15 \
    --eps-rows 100 \
    --eps-cols 500 \
    --maxiter 90000 \
    --maxfun 90000 \
    --n-jobs 16 \
    --numba \
    --progress \
    --progress-every 5 \
    --randcustsample data/randcustsample_seed123.csv \
    --randcustsample-base 1 \
    --eps-draws data/eps_draws_seed123.csv \
    > logs/run_estimation_table5_matlab_sample.log 2>&1 &


5) Alternative optimizer: trust-constr / "interior-point" style
--------------------------------------------------------------

  nohup python -m scripts.run_estimation_ip \
    --data data/preparedata_new.parquet \
    --initial data/initial_value.csv \
    --output data/output_table5_ip.csv \
    --seed 123 \
    --rng matlab \
    --n-customers 10000 \
    --max-hotels 15 \
    --eps-rows 100 \
    --eps-cols 500 \
    --maxiter 90000 \
    --maxfun 90000 \
    --function-tol 1e-6 \
    --step-tol 1e-6 \
    --n-jobs 16 \
    --numba \
    --randcustsample data/randcustsample_seed123.csv \
    --randcustsample-base 1 \
    --eps-draws data/eps_draws_seed123.csv \
    > logs/run_estimation_table5_ip.log 2>&1 &


6) Outputs
----------

All `--output` files are written under `data/` in the examples above.
These are local artifacts and are not intended to be version controlled.

