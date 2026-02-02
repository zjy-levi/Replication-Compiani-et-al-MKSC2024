"""Estimation helpers for the Expedia demand-side model."""

from __future__ import annotations

from dataclasses import dataclass
import csv
from pathlib import Path
from typing import Callable, Iterable
import logging
import multiprocessing as mp
import time

import numpy as np

from .random_utils import RNG, make_rng, uniform


@dataclass(frozen=True)
class EstimationResult:
    """Optimization output from the demand estimation."""
    params: np.ndarray
    objective: float
    success: bool
    message: str


def load_initial_params(path: str | Path, var_name: str = "bb") -> np.ndarray:
    """Load initial parameter values for optimization.

    Args:
        path: CSV or MAT file containing initial parameters.
        var_name: MAT variable name (defaults to `bb` as in MATLAB).
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return np.loadtxt(path, delimiter=",", dtype=float).reshape(-1)

    if suffix == ".mat":
        try:
            from scipy.io import loadmat
        except Exception as exc:  # pragma: no cover - environment specific
            raise ImportError(
                "SciPy is required to load MATLAB v5 .mat files. "
                "Install a compatible SciPy/Numpy pair before running."
            ) from exc

        mat = loadmat(path, squeeze_me=True, struct_as_record=False)
        if var_name not in mat:
            raise KeyError(f"Variable '{var_name}' not found in {path}")

        return np.asarray(mat[var_name]).reshape(-1).astype(float)

    raise ValueError(f"Unsupported initial parameter format: {path.suffix}")


def save_estimates(path: str | Path, params: np.ndarray, objective: float, success: bool) -> None:
    """Save estimation output in a Python-friendly format (csv/parquet).

    Supported formats:
    - .csv / .parquet: one-row table with objective/success and param columns
    - .mat: MATLAB-compatible output (bb, fval, flag)
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in {".csv", ".parquet", ".pq"}:
        row: dict[str, float | int] = {"objective": float(objective), "success": int(success)}
        for idx, value in enumerate(params):
            row[f"param_{idx}"] = float(value)

        if suffix == ".csv":
            fieldnames = list(row.keys())
            with path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(row)
            return

        # Parquet writing requires an Arrow/Pandas stack which may not be available in all environments.
        try:
            import pandas as pd  # type: ignore
        except Exception as exc:  # pragma: no cover - environment specific
            raise ImportError(
                "Writing parquet requires pandas (and a parquet engine such as pyarrow). "
                "Either install compatible dependencies or write to .csv instead."
            ) from exc
        df = pd.DataFrame([row])
        df.to_parquet(path, index=False)
        return

    if suffix == ".mat":
        try:
            from scipy.io import savemat
        except Exception as exc:  # pragma: no cover - environment specific
            raise ImportError(
                "SciPy is required to save MATLAB v5 .mat files. "
                "Install a compatible SciPy/Numpy pair before running."
            ) from exc
        savemat(path, {"bb": params, "fval": objective, "flag": int(success)})
        return

    raise ValueError(f"Unsupported output format: {path.suffix}")


def generate_eps_draws(
    seed: int | None,
    n_rows: int = 100,
    n_cols: int = 500,
    rng: RNG | None = None,
    rng_type: str = "numpy",
) -> np.ndarray:
    """Generate Gumbel draws for eps_pre integration (Section 4.2)."""
    if rng is None:
        rng = make_rng(seed, rng_type)
    if rng_type == "matlab":
        # MATLAB (as used in Replication/table5_estimates.m):
        #
        #   rng(seed); U = rand(n_rows, n_cols); eps_draws = log(-log(U));
        #
        # Important subtlety: MATLAB fills matrices in column-major order.
        # To match MATLAB element-by-element, we draw a 1D stream then reshape with order='F'.
        u_flat = uniform(rng, n_rows * n_cols).reshape(-1)
        u = u_flat.reshape((n_rows, n_cols), order="F")
        return np.log(-np.log(u))
    return rng.gumbel(loc=0.0, scale=1.0, size=(n_rows, n_cols))


@dataclass
class _ProgressState:
    maxiter: int
    log_every: int
    logger: logging.Logger
    start_time: float
    objective_fn: Callable[[np.ndarray], float]
    iterations: int = 0
    nfev: int = 0
    last_obj: float | None = None

    def objective(self, x: np.ndarray) -> float:
        self.nfev += 1
        value = self.objective_fn(x)
        self.last_obj = float(value)
        return value

    def callback(self, xk: np.ndarray) -> None:
        self.iterations += 1
        if self.log_every <= 0 or self.iterations % self.log_every != 0:
            return
        elapsed = time.perf_counter() - self.start_time
        eta_msg = "eta=unknown"
        if self.iterations > 0 and self.maxiter > 0:
            avg = elapsed / self.iterations
            remaining = max(self.maxiter - self.iterations, 0)
            eta_msg = f"eta={remaining * avg:.2f}s"
        obj_msg = "obj=unknown" if self.last_obj is None else f"obj={self.last_obj:.6f}"
        self.logger.info(
            "iter=%d nfev=%d %s elapsed=%.2fs %s",
            self.iterations,
            self.nfev,
            obj_msg,
            elapsed,
            eta_msg,
        )
        params_str = np.array2string(xk, precision=6, separator=", ")
        self.logger.info("params=%s", params_str)


_WORKER_DATA: DemandData | None = None
_WORKER_EPS_DRAWS: np.ndarray | None = None


def _init_worker(data_chunks: list[DemandData], eps_draws: np.ndarray) -> None:
    """Initialize worker with its data chunk for parallel likelihood evaluation."""
    global _WORKER_DATA, _WORKER_EPS_DRAWS
    worker_id = mp.current_process()._identity[0] - 1
    _WORKER_DATA = data_chunks[worker_id]
    _WORKER_EPS_DRAWS = eps_draws


def _worker_neg_log_likelihood(params: np.ndarray) -> float:
    """Compute negative log-likelihood for the worker's data chunk."""
    from .likelihood import negative_log_likelihood

    if _WORKER_DATA is None or _WORKER_EPS_DRAWS is None:
        raise RuntimeError("Worker data not initialized.")
    return negative_log_likelihood(params, _WORKER_DATA, _WORKER_EPS_DRAWS)


def _split_demand_data_by_groups(data: DemandData, n_chunks: int) -> list[DemandData]:
    """Split DemandData into contiguous group chunks for parallel processing."""
    # Local import to avoid importing pandas-heavy modules when only using utilities
    # like `generate_eps_draws` in minimal environments.
    from .data_utils import DemandData

    n_groups = len(data.group_slices)
    if n_groups == 0:
        return [data]
    n_chunks = max(1, min(n_chunks, n_groups))
    group_indices = np.arange(n_groups)
    chunks: list[DemandData] = []
    for idxs in np.array_split(group_indices, n_chunks):
        if idxs.size == 0:
            continue
        g_start = int(idxs[0])
        g_end = int(idxs[-1]) + 1
        row_start = data.group_slices[g_start][0]
        row_end = data.group_slices[g_end - 1][1]

        click = data.click[row_start:row_end]
        book = data.book[row_start:row_end]
        customer = data.customer[row_start:row_end]
        search_matrix = data.search_matrix[row_start:row_end]
        page_matrix = data.page_matrix[row_start:row_end]
        group_slices = [
            (int(s - row_start), int(e - row_start))
            for (s, e) in data.group_slices[g_start:g_end]
        ]

        chunks.append(
            DemandData(
                click=click,
                book=book,
                customer=customer,
                search_matrix=search_matrix,
                page_matrix=page_matrix,
                group_slices=group_slices,
                n_hotels=data.n_hotels,
            )
        )
    return chunks


def estimate_parameters(
    data: DemandData,
    initial_params: np.ndarray,
    eps_draws: np.ndarray,
    maxiter: int = 90_000,
    maxfun: int | None = None,
    progress: bool = False,
    progress_every: int = 5,
    logger: logging.Logger | None = None,
    n_jobs: int = 1,
) -> EstimationResult:
    """Run maximum likelihood estimation (Table 5 in the paper).

    Args:
        data: DemandData with per-consumer arrays.
        initial_params: Starting values for optimization.
        eps_draws: Monte Carlo draws for eps_pre integration.
        maxiter: Optimizer iteration limit.
        maxfun: Maximum number of function/gradient evaluations. Defaults to maxiter.
        progress: Enable per-iteration progress logging.
        progress_every: Log progress every N iterations.
        logger: Optional logger to use for progress output.
    """
    expected_len = 25 + 2 * data.n_hotels
    if len(initial_params) != expected_len:
        raise ValueError(
            f"Parameter length {len(initial_params)} does not match expected "
            f"{expected_len} for n_hotels={data.n_hotels}."
        )

    try:
        from scipy.optimize import minimize
    except Exception as exc:  # pragma: no cover - environment specific
        raise ImportError(
            "SciPy is required for optimization. "
            "Install a compatible SciPy/Numpy pair before running."
        ) from exc

    bounds = [(-30.0, 30.0)] * len(initial_params)
    bounds[-1] = (1.0, 1.0)

    if logger is None:
        logger = logging.getLogger(__name__)

    if maxfun is None:
        maxfun = maxiter

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
        from .likelihood import negative_log_likelihood

        def objective_fn(x: np.ndarray) -> float:
            return negative_log_likelihood(x, data, eps_draws)

    callback = None
    objective = objective_fn
    if progress:
        state = _ProgressState(
            maxiter=maxiter,
            log_every=progress_every,
            logger=logger,
            start_time=time.perf_counter(),
            objective_fn=objective_fn,
        )
        objective = state.objective
        callback = state.callback

    try:
        result = minimize(
            objective,
            initial_params,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": maxiter, "maxfun": maxfun, "disp": True},
            callback=callback,
        )
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    return EstimationResult(
        params=result.x,
        objective=float(result.fun),
        success=bool(result.success),
        message=str(result.message),
    )
