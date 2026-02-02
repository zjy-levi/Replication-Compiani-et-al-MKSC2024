nohup python -m python_replication.scripts.run_estimation \
    --n-customers 2000 \
    --eps-cols 100 \
    --eps-rows 50 \
    --max-hotels 10 \
    --maxiter 200 \
    --progress \
    --progress-every 5 \
    > python_replication/logs/run_estimation.log 2>&1 &

# Table 5 replication settings (match Replication/table5_estimates.m)
nohup python -m python_replication.scripts.run_estimation \
    --data python_replication/data/preparedata_new.parquet \
    --initial python_replication/data/initial_value.csv \
    --output python_replication/data/output.csv \
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
    > python_replication/logs/run_estimation_table5.log 2>&1 &

# Table 5 replication (L-BFGS-B) using MATLAB sample + eps draws
nohup python -m python_replication.scripts.run_estimation \
    --data python_replication/data/preparedata_new.parquet \
    --initial python_replication/data/initial_value.csv \
    --output python_replication/data/output_matlab_sample.csv \
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
    --randcustsample python_replication/data/randcustsample_seed123.csv \
    --randcustsample-base 1 \
    --eps-draws python_replication/data/eps_draws_seed123.csv \
    > python_replication/logs/run_estimation_table5_matlab_sample.log 2>&1 &

# Table 5 replication (trust-constr / interior-point style, MATLAB-like tolerances + MATLAB draws)
nohup python -m python_replication.scripts.run_estimation_ip \
    --data python_replication/data/preparedata_new.parquet \
    --initial python_replication/data/initial_value.csv \
    --output python_replication/data/output_ip.csv \
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
    --randcustsample python_replication/data/randcustsample_seed123.csv \
    --randcustsample-base 1 \
    --eps-draws python_replication/data/eps_draws_seed123.csv \
    > python_replication/logs/run_estimation_table5_ip.log 2>&1 &

# Same settings with a different random seed (sample perturbation check)
nohup python -m python_replication.scripts.run_estimation \
    --data python_replication/data/preparedata_new.parquet \
    --initial python_replication/data/initial_value.csv \
    --output python_replication/data/output_seed789.csv \
    --seed 789 \
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
    > python_replication/logs/run_estimation_table5_seed789.log 2>&1 &

nohup python -m python_replication.scripts.run_estimation \
    --data python_replication/data/preparedata_new.parquet \
    --initial python_replication/data/initial_value.csv \
    --output python_replication/data/output.csv \
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
    > python_replication/logs/run_estimation_table5.log &