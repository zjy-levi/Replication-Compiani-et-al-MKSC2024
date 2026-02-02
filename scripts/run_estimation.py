"""Run demand-side estimation to replicate Table 5."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np

from .data_utils import prepare_demand_data
from .estimation import estimate_parameters, generate_eps_draws, load_initial_params, save_estimates
from .likelihood import set_numba_enabled
from .random_utils import make_rng


def parse_args() -> argparse.Namespace:
    """Define CLI flags for running the demand estimation."""
    parser = argparse.ArgumentParser(description="Estimate the Expedia demand-side model.")
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
        help="Optional CSV/MAT of MATLAB randcustsample indices",
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
        action="store_true",
        help="Enable per-iteration progress logging",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=5,
        help="Log progress every N iterations",
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


def main() -> None:
    """Entry point for Table 5 estimation using the Python replication."""
    args = parse_args()
    setup_logging()
    logger = logging.getLogger(__name__)
    set_numba_enabled(args.numba)

    total_start = time.perf_counter()

    t0 = time.perf_counter()
    sample_seed = args.seed if args.sample_seed is None else args.sample_seed
    eps_seed = args.seed if args.eps_seed is None else args.eps_seed

    sample_rng = make_rng(sample_seed, args.rng)
    eps_rng = make_rng(eps_seed, args.rng)

    logger.info("rng=%s sample_seed=%s eps_seed=%s", args.rng, sample_seed, eps_seed)

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

    t2 = time.perf_counter()
    result = estimate_parameters(
        data,
        initial_params,
        eps_draws,
        maxiter=args.maxiter,
        maxfun=args.maxfun,
        progress=args.progress,
        progress_every=args.progress_every,
        logger=logger,
        n_jobs=args.n_jobs,
    )
    logger.info("optimization finished in %.2fs", time.perf_counter() - t2)

    save_estimates(args.output, result.params, result.objective, result.success)
    logger.info("saved output to %s", args.output)

    logger.info("success=%s objective=%.4f", result.success, result.objective)
    logger.info("message=%s", result.message)
    logger.info("total runtime %.2fs", time.perf_counter() - total_start)


if __name__ == "__main__":
    main()
