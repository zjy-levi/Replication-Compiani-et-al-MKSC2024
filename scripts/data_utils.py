"""Data loading and preprocessing for the Expedia demand model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .random_utils import RNG, sample_with_replacement

@dataclass(frozen=True)
class DemandData:
    """Container for arrays used in estimation.

    This mirrors the estimation inputs in Section 4.2 of the paper, with
    clicks/bookings by product and design matrices for the search and utility
    indices defined in Eqs. (3) and (8).
    """

    # click: per-row indicator for whether product j was clicked in impression t.
    click: np.ndarray
    # book: per-row indicator for whether product j was booked in impression t.
    book: np.ndarray
    # customer: per-row customer/impression id used to group sequences.
    customer: np.ndarray
    # search_matrix: design matrix for s_ijt (Eq. 3), includes position dummies + x_jt + hotel FE + intercept.
    search_matrix: np.ndarray
    # page_matrix: design matrix for u_ijt (Eq. 8), includes x_jt + hotel FE + intercept.
    page_matrix: np.ndarray
    # group_slices: list of (start, end) row indices for each customer's block.
    group_slices: list[tuple[int, int]]
    # n_hotels: number of unique hotel ids after preprocessing (controls FE length).
    n_hotels: int


def load_dataframe(path: str | Path) -> pd.DataFrame:
    """Load the pre-exported Expedia data from parquet/csv.

    Args:
        path: File path to `preparedata_new` or `preparedata_test` in parquet/csv.

    Returns:
        DataFrame with the raw columns used to construct x_jt and rank r_jt.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported data format: {path.suffix}")


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Match the MATLAB preprocessing steps before estimation.

    This mirrors `table5_estimates.m`:
    - hotel IDs are capped at 21 and 0 is reassigned to 21
    - positions are capped at 13
    - review score is converted to numeric
    """
    df = df.copy()
    df["hotel"] = df["hotel"].astype(int)
    df.loc[df["hotel"] > 20, "hotel"] = 21
    df.loc[df["hotel"] == 0, "hotel"] = 21

    df["position"] = df["position"].astype(int)
    df.loc[df["position"] > 12, "position"] = 13

    df["prop_review_score"] = pd.to_numeric(df["prop_review_score"], errors="coerce")
    return df


def sort_by_customer(df: pd.DataFrame) -> pd.DataFrame:
    """Sort rows by customer id to align with per-consumer likelihood blocks."""
    return df.sort_values("customeri").reset_index(drop=True)


def sample_customers(
    df: pd.DataFrame,
    max_hotels: int = 15,
    n_customers: int = 10_000,
    seed: int = 123,
    rng: RNG | None = None,
    use_matlab_rng: bool = False,
    sample_indices: np.ndarray | None = None,
    sample_indices_base: int = 0,
) -> pd.DataFrame:
    """Sample customers with at most `max_hotels` products (MATLAB logic).

    Args:
        df: Preprocessed data for estimation (training sample).
        max_hotels: Maximum number of products per consumer (<=15 in MATLAB).
        n_customers: Number of sampled customers (10,000 in MATLAB).
        seed: RNG seed to match the paper's sampling step.
    """
    groups = [g for _, g in df.groupby("customeri", sort=False)]
    valid_idx = [i for i, g in enumerate(groups) if len(g) <= max_hotels]
    if not valid_idx:
        raise ValueError("No customers satisfy the max_hotels filter.")

    if sample_indices is not None:
        sample_idx = np.asarray(sample_indices, dtype=int).reshape(-1)
        if sample_indices_base not in (0, 1):
            raise ValueError("sample_indices_base must be 0 or 1.")
        if sample_indices_base == 1:
            sample_idx = sample_idx - 1
        if sample_idx.size == 0:
            raise ValueError("sample_indices is empty.")
        if sample_idx.min() < 0 or sample_idx.max() >= len(groups):
            raise ValueError("sample_indices out of range for customer groups.")
        if max_hotels is not None:
            invalid = [i for i in sample_idx if len(groups[i]) > max_hotels]
            if invalid:
                raise ValueError("sample_indices includes customers exceeding max_hotels.")
    else:
        if rng is None:
            rng = np.random.default_rng(seed)
        sample_idx = sample_with_replacement(
            valid_idx, size=n_customers, rng=rng, matlab_compat=use_matlab_rng
        )

    sampled_groups = [groups[i] for i in sample_idx]
    df_sample = pd.concat(sampled_groups, ignore_index=True)

    new_customer = np.concatenate(
        [np.full(len(groups[i]), idx + 1, dtype=int) for idx, i in enumerate(sample_idx)]
    )
    df_sample["customeri"] = new_customer
    return df_sample


def build_design_matrices(
    df: pd.DataFrame,
    n_hotels: int,
    position_levels: int = 13,
    hotel_levels: int = 21,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct design matrices for search and utility indices.

    Returns:
        search_matrix: [position dummies, x_jt, hotel FE, intercept] for Eq. (3).
        page_matrix: [x_jt, hotel FE, intercept] for Eq. (8).
    """
    position = df["position"].to_numpy(dtype=int)
    hotel = df["hotel"].to_numpy(dtype=int)

    pos_dummy = _categorical_dummies(position, list(range(1, position_levels + 1)))[
        :, : position_levels - 1
    ]
    hotel_dummy = _categorical_dummies(hotel, list(range(1, hotel_levels + 1)))[
        :, : hotel_levels - 1
    ]

    covariates = np.column_stack(
        [
            df["prop_starrating"].to_numpy(float),
            df["prop_review_score"].to_numpy(float),
            df["prop_brand_bool"].to_numpy(float),
            df["prop_location_score1"].to_numpy(float),
            df["price_usd"].to_numpy(float) / 1000.0,
            df["promotion_flag"].to_numpy(float),
        ]
    )

    n = len(df)
    intercept = np.ones((n, 1), dtype=float)
    search_matrix = np.column_stack([pos_dummy, covariates, hotel_dummy, intercept])
    page_matrix = np.column_stack([covariates, hotel_dummy, intercept])
    return search_matrix, page_matrix


def _categorical_dummies(values: np.ndarray, categories: list[int]) -> np.ndarray:
    """One-hot encode using pandas Categorical for stable column ordering."""
    series = pd.Series(values, dtype="int64")
    if not np.isin(series.to_numpy(), categories).all():
        raise ValueError("Values out of expected categories for one-hot encoding.")

    cat = pd.Categorical(series, categories=categories, ordered=True)
    dummies = pd.get_dummies(cat, dtype=float)
    if dummies.shape[1] != len(categories):
        raise ValueError("Unexpected dummy column count.")

    return dummies.to_numpy()


def build_group_slices(customer: np.ndarray) -> list[tuple[int, int]]:
    """Create (start, end) slices for each consumer's rows."""
    unique, counts = np.unique(customer, return_counts=True)
    starts = np.cumsum(np.concatenate([[0], counts[:-1]]))
    return [(int(s), int(s + c)) for s, c in zip(starts, counts)]


def prepare_demand_data(
    data_path: str | Path,
    max_hotels: int = 15,
    n_customers: int = 10_000,
    seed: int = 123,
    rng: RNG | None = None,
    use_matlab_rng: bool = False,
    sample_indices: np.ndarray | None = None,
    sample_indices_base: int = 0,
) -> DemandData:
    """Load and preprocess estimation data into arrays for likelihood.

    Args:
        data_path: Path to parquet/csv data.
        max_hotels: Maximum number of products per consumer.
        n_customers: Sample size.
        seed: RNG seed for sampling.
    """
    df = load_dataframe(data_path)
    df = preprocess_dataframe(df)

    df = sort_by_customer(df)
    df_sample = sample_customers(
        df,
        max_hotels=max_hotels,
        n_customers=n_customers,
        seed=seed,
        rng=rng,
        use_matlab_rng=use_matlab_rng,
        sample_indices=sample_indices,
        sample_indices_base=sample_indices_base,
    )

    return build_demand_data_from_df(df_sample)


def build_demand_data_from_df(df_sample: pd.DataFrame) -> DemandData:
    """Convert a sampled DataFrame into the DemandData structure."""
    n_hotels = 21
    search_matrix, page_matrix = build_design_matrices(
        df_sample,
        n_hotels=n_hotels,
        position_levels=13,
        hotel_levels=21,
    )
    click = df_sample["click"].to_numpy(float)
    book = df_sample["book"].to_numpy(float)
    customer = df_sample["customeri"].to_numpy(int)

    group_slices = build_group_slices(customer)

    return DemandData(
        click=click,
        book=book,
        customer=customer,
        search_matrix=search_matrix,
        page_matrix=page_matrix,
        group_slices=group_slices,
        n_hotels=n_hotels,
    )
