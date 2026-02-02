"""Run bootstrap for demand-side estimation (Table 5 SEs)."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .data_utils import (
    build_demand_data_from_df,
    load_dataframe,
    preprocess_dataframe,
    sample_customers,
    sort_by_customer,
)
from .estimation import estimate_parameters, generate_eps_draws, load_initial_params, save_estimates
from .likelihood import set_numba_enabled


def parse_args() -> argparse.Namespace:
    """Define CLI flags for bootstrap standard errors."""
    parser = argparse.ArgumentParser(description="Bootstrap the Expedia demand-side model.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("python_replication/data/preparedata_new.parquet"),
        help="Path to preparedata_new (parquet/csv)",
    )
    parser.add_argument(
        "--initial",
        type=Path,
        default=Path("python_replication/data/output.mat"),
        help="Path to output (mat/csv) containing bb",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("python_replication/data/output_bootstrap.mat"),
        help="Output .mat for bootstrap draws",
    )
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument("--n-customers", type=int, default=10_000, help="Sample size")
    parser.add_argument("--max-hotels", type=int, default=15, help="Max hotels per customer")
    parser.add_argument("--eps-rows", type=int, default=100, help="Rows for eps_draws")
    parser.add_argument("--eps-cols", type=int, default=500, help="Columns for eps_draws")
    parser.add_argument("--maxiter", type=int, default=90_000, help="Optimizer iterations")
    parser.add_argument("--nboot", type=int, default=250, help="Number of bootstrap draws")
    parser.add_argument(
        "--numba",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable numba acceleration (default: enabled when available)",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Enable per-iteration progress logging inside each optimization",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Log progress every N iterations inside each optimization",
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
    """Entry point for bootstrap estimation (Table 5 SEs)."""
    args = parse_args()
    setup_logging()
    logger = logging.getLogger(__name__)
    set_numba_enabled(args.numba)

    total_start = time.perf_counter()

    t0 = time.perf_counter()
    df = load_dataframe(args.data)
    df = preprocess_dataframe(df)
    df = sort_by_customer(df)
    df_sample = sample_customers(
        df, max_hotels=args.max_hotels, n_customers=args.n_customers, seed=args.seed
    )

    groups = [g for _, g in df_sample.groupby("customeri", sort=False)]
    nind = len(groups)

    logger.info("loaded data in %.2fs", time.perf_counter() - t0)

    t1 = time.perf_counter()
    initial_params = load_initial_params(args.initial)
    eps_draws = generate_eps_draws(args.seed, n_rows=args.eps_rows, n_cols=args.eps_cols)
    logger.info("loaded params/draws in %.2fs", time.perf_counter() - t1)

    bb_boot = np.zeros((len(initial_params), args.nboot))
    rng = np.random.default_rng(args.seed)

    for i in range(args.nboot):
        iter_start = time.perf_counter()
        bootsample_idx = rng.choice(np.arange(nind), size=nind, replace=True)
        boot_groups = [groups[idx] for idx in bootsample_idx]
        df_boot = pd.concat(boot_groups, ignore_index=True)
        new_customer = np.concatenate(
            [np.full(len(groups[idx]), j + 1, dtype=int) for j, idx in enumerate(bootsample_idx)]
        )
        df_boot["customeri"] = new_customer

        boot_data = build_demand_data_from_df(df_boot)
        result = estimate_parameters(
            boot_data,
            initial_params,
            eps_draws,
            maxiter=args.maxiter,
            progress=args.progress,
            progress_every=args.progress_every,
            logger=logger,
        )
        bb_boot[:, i] = result.params
        logger.info(
            "bootstrap %d/%d: success=%s elapsed=%.2fs",
            i + 1,
            args.nboot,
            result.success,
            time.perf_counter() - iter_start,
        )

    try:
        from scipy.io import savemat
    except Exception as exc:  # pragma: no cover - environment specific
        raise ImportError(
            "SciPy is required to save MATLAB v5 .mat files. "
            "Install a compatible SciPy/Numpy pair before running."
        ) from exc

    savemat(args.output, {"bb_boot": bb_boot})
    logger.info("saved output to %s", args.output)
    logger.info("total runtime %.2fs", time.perf_counter() - total_start)


if __name__ == "__main__":
    main()
