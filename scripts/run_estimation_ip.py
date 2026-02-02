"""Run demand-side estimation with interior-point style optimization."""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from .data_utils import prepare_demand_data
from .estimation import (
    _init_worker,
    _split_demand_data_by_groups,
    _worker_neg_log_likelihood,
    generate_eps_draws,
    load_initial_params,
    save_estimates,
)
from .likelihood import negative_log_likelihood, set_numba_enabled
from .random_utils import make_rng


def parse_args() -> argparse.Namespace:
    """Define CLI flags for running the demand estimation."""
    parser = argparse.ArgumentParser(description="Estimate the Expedia demand-side model (interior-point).")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("python_replication/data/preparedata_new.parquet"),
        help="Path to preparedata_new (parquet/csv)",
    )
    parser.add_argument(
        "--initial",
        type=Path,
        default=Path("python_replication/data/initial_value.csv"),
        help="Path to initial_value (csv/mat)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("python_replication/data/output.csv"),
        help="Output .csv for estimates",
    )
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument(
        "--rng",
        type=str,
        default="numpy",
        choices=["numpy", "matlab"],
        help="RNG backend (matlab uses MT19937-compatible stream)",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=None,
        help="Seed for customer sampling (defaults to --seed)",
    )
    parser.add_argument(
        "--eps-seed",
        type=int,
        default=None,
        help="Seed for eps_draws (defaults to --seed)",
    )
    parser.add_argument(
        "--randcustsample",
        type=Path,
        default=None,
        help="Optional CSV/MATLAB-exported randcustsample indices",
    )
    parser.add_argument(
        "--randcustsample-base",
        type=int,
        default=1,
        choices=[0, 1],
        help="Index base for randcustsample (1 for MATLAB, 0 for Python)",
    )
    parser.add_argument(
        "--eps-draws",
        type=Path,
        default=None,
        help="Optional CSV of MATLAB eps_draws (rows x cols)",
    )
    parser.add_argument("--n-customers", type=int, default=10_000, help="Sample size")
    parser.add_argument("--max-hotels", type=int, default=15, help="Max hotels per customer")
    parser.add_argument("--eps-rows", type=int, default=100, help="Rows for eps_draws")
    parser.add_argument("--eps-cols", type=int, default=500, help="Columns for eps_draws")
    parser.add_argument("--maxiter", type=int, default=90_000, help="Optimizer iterations")
    parser.add_argument(
        "--maxfun",
        type=int,
        default=90_000,
        help="Max function/gradient evaluations (aligns with MATLAB MaxFunctionEvaluations)",
    )
    parser.add_argument(
        "--function-tol",
        type=float,
        default=1e-6,
        help="Function/optimality tolerance (matches MATLAB FunctionTolerance)",
    )
    parser.add_argument(
        "--step-tol",
        type=float,
        default=1e-6,
        help="Step tolerance (matches MATLAB StepTolerance)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Parallel worker processes for likelihood evaluation",
    )
    parser.add_argument(
        "--numba",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable numba acceleration (default: enabled when available)",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable per-iteration progress logging (MATLAB Display='iter')",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=5,
        help="Log parameter vector every N iterations",
    )
    return parser.parse_args()


def setup_logging() -> None:
    """Configure logging to stdout only (for nohup redirection)."""
    if logging.getLogger().handlers:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler()],
    )


def _build_objective(data, eps_draws, n_jobs: int):
    pool: mp.pool.Pool | None = None
    if n_jobs > 1:
        data_chunks = _split_demand_data_by_groups(data, n_jobs)
        ctx = mp.get_context()
        pool = ctx.Pool(
            processes=len(data_chunks),
            initializer=_init_worker,
            initargs=(data_chunks, eps_draws),
        )

        def objective_fn(x: np.ndarray) -> float:
            results = pool.map(_worker_neg_log_likelihood, [x] * len(data_chunks), chunksize=1)
            return float(np.sum(results))

    else:

        def objective_fn(x: np.ndarray) -> float:
            return negative_log_likelihood(x, data, eps_draws)

    return objective_fn, pool


def main() -> None:
    """Entry point for Table 5 estimation using trust-constr."""
    args = parse_args()
    setup_logging()
    logger = logging.getLogger(__name__)
    set_numba_enabled(args.numba)

    sample_seed = args.seed if args.sample_seed is None else args.sample_seed
    eps_seed = args.seed if args.eps_seed is None else args.eps_seed
    sample_rng = make_rng(sample_seed, args.rng)
    eps_rng = make_rng(eps_seed, args.rng)
    logger.info("rng=%s sample_seed=%s eps_seed=%s", args.rng, sample_seed, eps_seed)

    total_start = time.perf_counter()

    t0 = time.perf_counter()
    sample_indices = None
    if args.randcustsample is not None:
        sample_indices = np.loadtxt(args.randcustsample, delimiter=",", dtype=float)
        sample_indices = np.asarray(sample_indices, dtype=int).reshape(-1)
        logger.info(
            "loaded randcustsample from %s (n=%d, base=%d)",
            args.randcustsample,
            sample_indices.size,
            args.randcustsample_base,
        )

    data = prepare_demand_data(
        args.data,
        max_hotels=args.max_hotels,
        n_customers=args.n_customers,
        seed=sample_seed,
        rng=sample_rng,
        use_matlab_rng=args.rng == "matlab",
        sample_indices=sample_indices,
        sample_indices_base=args.randcustsample_base,
    )
    logger.info("loaded data in %.2fs", time.perf_counter() - t0)

    t1 = time.perf_counter()
    initial_params = load_initial_params(args.initial)
    if args.eps_draws is not None:
        eps_draws = np.loadtxt(args.eps_draws, delimiter=",", dtype=float)
        logger.info("loaded eps_draws from %s", args.eps_draws)
    else:
        eps_draws = generate_eps_draws(
            eps_seed,
            n_rows=args.eps_rows,
            n_cols=args.eps_cols,
            rng=eps_rng,
            rng_type=args.rng,
        )
    logger.info("loaded params/draws in %.2fs", time.perf_counter() - t1)

    objective_fn, pool = _build_objective(data, eps_draws, args.n_jobs)

    lb = np.full(len(initial_params), -30.0, dtype=float)
    ub = np.full(len(initial_params), 30.0, dtype=float)
    lb[-1] = 1.0
    ub[-1] = 1.0
    try:
        from scipy.optimize import Bounds
    except Exception as exc:  # pragma: no cover - environment specific
        raise ImportError(
            "SciPy is required for optimization. "
            "Install a compatible SciPy/Numpy pair before running."
        ) from exc
    bounds = Bounds(lb, ub, keep_feasible=True)

    header_printed = {"value": False}
    last_x = {"value": None}

    def callback(xk: np.ndarray, state: object) -> bool:
        nit = int(getattr(state, "nit", getattr(state, "niter", 0)))
        nfev = int(getattr(state, "nfev", 0))
        obj = float(getattr(state, "fun", np.nan))
        feas = float(getattr(state, "constr_violation", np.nan))
        opt = float(getattr(state, "optimality", np.nan))

        prev = last_x["value"]
        if prev is None:
            step_norm = 0.0
        else:
            step_norm = float(np.linalg.norm(xk - prev))
        last_x["value"] = np.copy(xk)

        if args.progress:
            if not header_printed["value"]:
                logger.info(
                    "Iter F-count            f(x)  Feasibility   1storderoptimality         normofstep"
                )
                header_printed["value"] = True
            logger.info(
                "%4d %7d %14.6e %12.3e %18.3e %18.3e",
                nit,
                nfev,
                obj,
                feas,
                opt,
                step_norm,
            )
            if args.progress_every > 0 and nit % args.progress_every == 0:
                params_str = np.array2string(xk, precision=6, separator=", ")
                logger.info("params=%s", params_str)

        return nfev >= args.maxfun

    try:
        from scipy.optimize import minimize
    except Exception as exc:  # pragma: no cover - environment specific
        raise ImportError(
            "SciPy is required for optimization. "
            "Install a compatible SciPy/Numpy pair before running."
        ) from exc

    t2 = time.perf_counter()
    try:
        result = minimize(
            objective_fn,
            initial_params,
            method="trust-constr",
            bounds=bounds,
            options={
                "maxiter": args.maxiter,
                "gtol": args.function_tol,
                "xtol": args.step_tol,
                "barrier_tol": args.function_tol,
                "verbose": 0,
            },
            callback=callback,
        )
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    logger.info("optimization finished in %.2fs", time.perf_counter() - t2)

    save_estimates(args.output, result.x, float(result.fun), bool(result.success))
    logger.info("saved output to %s", args.output)

    logger.info("success=%s objective=%.4f", result.success, float(result.fun))
    logger.info("message=%s", result.message)
    logger.info("total runtime %.2fs", time.perf_counter() - total_start)


if __name__ == "__main__":
    main()
