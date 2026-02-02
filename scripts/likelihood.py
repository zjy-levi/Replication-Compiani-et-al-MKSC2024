"""Likelihood implementation for the Expedia demand-side model."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable
from tqdm import tqdm

import numpy as np

from .data_utils import DemandData

try:  # Optional acceleration
    from numba import njit
    _HAS_NUMBA = True
except Exception:  # pragma: no cover - environment specific
    njit = None
    _HAS_NUMBA = False

_NUMBA_ENABLED = _HAS_NUMBA


def _build_permutation_cache(max_k: int = 6) -> dict[int, np.ndarray]:
    perms: dict[int, np.ndarray] = {}
    for k in range(1, max_k + 1):
        perms[k] = np.array(list(itertools.permutations(range(1, k + 1))), dtype=np.int64)
    return perms


_PERMUTATIONS_NUMBA = _build_permutation_cache() if _HAS_NUMBA else {}


def set_numba_enabled(enabled: bool) -> None:
    """Enable/disable numba acceleration at runtime."""
    global _NUMBA_ENABLED
    _NUMBA_ENABLED = bool(enabled) and _HAS_NUMBA


@dataclass(frozen=True)
class ParameterSlices:
    """Parameter blocks matching the paper's indices.

    gamma_s: position effects f(r_j) in Eq. (3)
    beta_s: search covariates beta_S in Eq. (3)
    beta_u: utility covariates beta_U in Eq. (8)
    w: search hotel fixed effects xi^S_j
    v: utility hotel fixed effects xi^U_j
    sig: common shock scale for eps_pre integration
    """

    gamma_s: np.ndarray
    beta_s: np.ndarray
    beta_u: np.ndarray
    w: np.ndarray
    v: np.ndarray
    sig: float


def split_parameters(params: np.ndarray, n_hotels: int) -> ParameterSlices:
    """Split the flat parameter vector into model blocks.

    The layout follows the MATLAB estimation in `mylikelihood_matrix_mixture_logit_GC.m`.
    """
    pos_len = 12 # position dummies (1-12)
    cov_len = 6 # number of search/utility covariates   
    w_len = n_hotels # search hotel fixed effects
    v_len = n_hotels # utility hotel fixed effects

    idx = 0
    gamma_s = params[idx : idx + pos_len]
    idx += pos_len
    beta_s = params[idx : idx + cov_len]
    idx += cov_len
    beta_u = params[idx : idx + cov_len]
    idx += cov_len
    w = params[idx : idx + w_len]
    idx += w_len
    v = params[idx : idx + v_len]
    idx += v_len
    sig = float(params[idx])

    return ParameterSlices(gamma_s=gamma_s, beta_s=beta_s, beta_u=beta_u, w=w, v=v, sig=sig)


def negative_log_likelihood(
    params: np.ndarray,
    data: DemandData,
    eps_draws: np.ndarray,
) -> float:
    """Compute the negative conditional log-likelihood (Eq. 11).

    Args:
        params: Parameter vector ordered as [gamma_s, beta_s, beta_u, w, v, sig].
        data: DemandData containing clicks/bookings and design matrices.
        eps_draws: Monte Carlo draws for eps_pre integration.
    """
    slices = split_parameters(params, n_hotels=data.n_hotels) # --> ParameterSlices
    search_params = np.concatenate([slices.gamma_s, slices.beta_s, slices.w])
    utility_params = np.concatenate([slices.beta_u, slices.v])

    pd_s = data.search_matrix @ search_params
    pd_u = data.page_matrix @ utility_params

    total = 0.0
    for start, end in data.group_slices:
        total += pathprobability_conditioning3_gc_components(
            data.click[start:end],
            data.book[start:end],
            pd_s[start:end],
            pd_u[start:end],
            slices.sig,
            eps_draws,
        )

    return -total


def pathprobability_conditioning3_gc(X: np.ndarray, sig: float, eps_draws: np.ndarray) -> float:
    """Integrate over eps_pre using Monte Carlo draws (Eq. 10 + integration).

    Args:
        X: Per-consumer rows [click, book, customer, pd_s, pd_u].
        sig: Scale of the common eps_pre shock.
        eps_draws: Gumbel draws used for simulation of eps_pre.
    """
    click = X[:, 0]
    book = X[:, 1]
    pd_s = X[:, 3]
    pd_u = X[:, 4]
    return pathprobability_conditioning3_gc_components(click, book, pd_s, pd_u, sig, eps_draws)


def pathprobability_conditioning3_gc_components(
    click: np.ndarray,
    book: np.ndarray,
    pd_s: np.ndarray,
    pd_u: np.ndarray,
    sig: float,
    eps_draws: np.ndarray,
) -> float:
    """Monte Carlo integration without rebuilding the full X matrix."""
    n_rows = int(pd_s.shape[0])
    kk = int(np.sum(click))
    if kk == 0:
        return float(np.log(1e-20))
    if n_rows > eps_draws.shape[0]:
        raise ValueError("eps_draws has fewer rows than required by the customer group.")

    if _NUMBA_ENABLED and kk <= 6:
        perms = _PERMUTATIONS_NUMBA[kk]
        return float(
            _pathprobability_conditioning3_gc_components_nb(
                click.astype(np.int64),
                book.astype(np.int64),
                pd_s.astype(np.float64),
                pd_u.astype(np.float64),
                float(sig),
                eps_draws.astype(np.float64),
                perms,
            )
        )

    return _pathprobability_conditioning3_gc_components_py(click, book, pd_s, pd_u, sig, eps_draws)


def _pathprobability_conditioning3_gc_components_py(
    click: np.ndarray,
    book: np.ndarray,
    pd_s: np.ndarray,
    pd_u: np.ndarray,
    sig: float,
    eps_draws: np.ndarray,
) -> float:
    """Python fallback for Monte Carlo integration."""
    kk = pd_s.shape[0]
    AA = -sig**2 * eps_draws[:kk, :]
    n_draws = eps_draws.shape[1]
    ppall = np.zeros(n_draws)

    tmp_s = np.empty_like(pd_s, dtype=float)
    tmp_u = np.empty_like(pd_u, dtype=float)
    for i in range(n_draws):
        np.add(pd_s, AA[:, i], out=tmp_s)
        np.add(pd_u, AA[:, i], out=tmp_u)
        ppall[i] = pathprobability_conditioning_components(click, book, tmp_s, tmp_u)

    return _log_mean_exp(ppall)


def pathprobability_conditioning(X: np.ndarray) -> float:
    """Compute the exploded-logit probability for a consumer's outcome.

    This follows `pathprobability_conditioning.m` and enumerates permutations
    of clicked items to account for unobserved search order.
    """
    click = X[:, 0]
    book = X[:, 1]
    pd_s = X[:, 3]
    pd_u = X[:, 4]
    return pathprobability_conditioning_components(click, book, pd_s, pd_u)


def pathprobability_conditioning_components(
    click: np.ndarray,
    book: np.ndarray,
    pd_s: np.ndarray,
    pd_u: np.ndarray,
) -> float:
    """Core exploded-logit computation using component arrays."""
    click = np.asarray(click, dtype=int)
    book = np.asarray(book, dtype=int)
    pd_s = np.asarray(pd_s, dtype=float)
    pd_u = np.asarray(pd_u, dtype=float)

    kk = int(click.sum())
    if kk == 0: # 如果没有点击
        return float(np.log(1e-20))

    if _NUMBA_ENABLED and kk <= 6:
        perms = _PERMUTATIONS_NUMBA[kk]
        return float(
            _pathprobability_conditioning_components_nb(
                click.astype(np.int64),
                book.astype(np.int64),
                pd_s.astype(np.float64),
                pd_u.astype(np.float64),
                perms,
            )
        )

    return _pathprobability_conditioning_components_py(click, book, pd_s, pd_u)


def _pathprobability_conditioning_components_py(
    click: np.ndarray,
    book: np.ndarray,
    pd_s: np.ndarray,
    pd_u: np.ndarray,
) -> float:
    """Python fallback for exploded-logit computation."""
    kk = int(click.sum())
    permutations = _get_permutations(kk)
    pp = 0.0

    if book.sum() == 0: # 如果没有预定
        for order in permutations:
            click_order = click.copy()
            click_order[click_order != 0] = order

            denomi = 1.0 + _sum_exp(pd_s[click_order == 0]) + _sum_exp(pd_u[click_order != 0])
            pp_tmp = -np.log(denomi)

            for ii in range(kk, 0, -1):
                denomi += np.exp(_scalar(pd_s[click_order == ii]))
                pp_tmp += _scalar(pd_s[click_order == ii]) - np.log(denomi)

            pp_tmp -= _log_click_condition(pd_s)
            pp += np.exp(pp_tmp)

    else:
        select = (book == 1) # 预定的酒店索引布尔数组
        for order in permutations:
            click_order = click.copy()
            click_order[click_order != 0] = order # 被买的酒店按顺序编号

            if book[click_order == kk][0] == 0:
                denomi = 1.0 + _sum_exp(pd_u[click_order != 0]) + _sum_exp(pd_s[click_order == 0])
                pp_tmp = _scalar(pd_u[select]) - np.log(denomi)
                for ii in range(kk, 0, -1):
                    denomi += np.exp(_scalar(pd_s[click_order == ii]))
                    pp_tmp += _scalar(pd_s[click_order == ii]) - np.log(denomi)
                pp_tmp -= _log_click_condition(pd_s)
                pp_tmp = np.exp(pp_tmp)
            else:
                pp_tmp = 0.0
                if kk > 1:
                    for iii in range(kk, 1, -1):
                        denomi = (
                            1.0
                            + _sum_exp(pd_u[(click_order != 0) & (~select)])
                            + _sum_exp(pd_s[click_order == 0])
                        )
                        pp0 = 0.0
                        for ii in range(kk, iii - 1, -1):
                            denomi += np.exp(_scalar(pd_s[click_order == ii]))
                            pp0 += _scalar(pd_s[click_order == ii]) - np.log(denomi)
                        denomi += np.exp(_scalar(pd_u[select]))
                        pp0 += _scalar(pd_u[select]) - np.log(denomi)
                        for ii in range(iii - 1, 0, -1):
                            denomi += np.exp(_scalar(pd_s[click_order == ii]))
                            pp0 += _scalar(pd_s[click_order == ii]) - np.log(denomi)
                        pp_tmp += np.exp(pp0)

                denomi = 1.0 + _sum_exp(pd_u[click_order != 0]) + _sum_exp(pd_s[click_order == 0])
                pp0 = _scalar(pd_u[select]) - np.log(denomi)
                for ii in range(kk, 0, -1):
                    denomi += np.exp(_scalar(pd_s[click_order == ii]))
                    pp0 += _scalar(pd_s[click_order == ii]) - np.log(denomi)
                pp_tmp += np.exp(pp0)

                denomi = (
                    1.0
                    + _sum_exp(pd_u[(click_order != 0) & (~select)])
                    + _sum_exp(pd_s[click_order == 0])
                )
                pp0 = 0.0
                for ii in range(kk, 0, -1):
                    denomi += np.exp(_scalar(pd_s[click_order == ii]))
                    pp0 += _scalar(pd_s[click_order == ii]) - np.log(denomi)
                denomi += np.exp(_scalar(pd_u[select]))
                pp0 += _scalar(pd_u[select]) - np.log(denomi)
                pp_tmp += np.exp(pp0)

                pp_tmp = np.log(pp_tmp + (pp_tmp == 0.0) * 1e-16)
                pp_tmp -= _log_click_condition(pd_s, eps=1e-16)

            pp += np.exp(pp_tmp)

    return float(np.log(pp + 1e-20))


if _HAS_NUMBA:
    @njit(cache=True)
    def _sum_exp_nb(values: np.ndarray) -> float:
        total = 0.0
        for i in range(values.shape[0]):
            total += np.exp(values[i])
        return total


    @njit(cache=True)
    def _sum_exp_eq_nb(values: np.ndarray, click_order: np.ndarray, target: int) -> float:
        total = 0.0
        for i in range(values.shape[0]):
            if click_order[i] == target:
                total += np.exp(values[i])
        return total


    @njit(cache=True)
    def _sum_exp_ne0_nb(values: np.ndarray, click_order: np.ndarray) -> float:
        total = 0.0
        for i in range(values.shape[0]):
            if click_order[i] != 0:
                total += np.exp(values[i])
        return total


    @njit(cache=True)
    def _sum_exp_ne0_not_select_nb(
        values: np.ndarray,
        click_order: np.ndarray,
        select: np.ndarray,
    ) -> float:
        total = 0.0
        for i in range(values.shape[0]):
            if click_order[i] != 0 and not select[i]:
                total += np.exp(values[i])
        return total


    @njit(cache=True)
    def _scalar_by_order_nb(values: np.ndarray, click_order: np.ndarray, target: int) -> float:
        for i in range(values.shape[0]):
            if click_order[i] == target:
                return values[i]
        return 0.0


    @njit(cache=True)
    def _scalar_by_select_nb(values: np.ndarray, select: np.ndarray) -> float:
        for i in range(values.shape[0]):
            if select[i]:
                return values[i]
        return 0.0


    @njit(cache=True)
    def _log_click_condition_nb(pd_s: np.ndarray, eps: float) -> float:
        total = _sum_exp_nb(pd_s)
        return np.log(total / (1.0 + total) + eps)


    @njit(cache=True)
    def _log_mean_exp_nb(values: np.ndarray) -> float:
        max_val = values[0]
        for i in range(1, values.shape[0]):
            if values[i] > max_val:
                max_val = values[i]
        total = 0.0
        for i in range(values.shape[0]):
            total += np.exp(values[i] - max_val)
        return max_val + np.log(total / values.shape[0])


    @njit(cache=True)
    def _pathprobability_conditioning_components_nb(
        click: np.ndarray,
        book: np.ndarray,
        pd_s: np.ndarray,
        pd_u: np.ndarray,
        perms: np.ndarray,
    ) -> float:
        n = click.shape[0]
        idx_clicked = np.empty(n, dtype=np.int64)
        kk = 0
        for i in range(n):
            if click[i] != 0:
                idx_clicked[kk] = i
                kk += 1
        if kk == 0:
            return np.log(1e-20)

        select = book == 1
        book_sum = 0
        for i in range(n):
            if select[i]:
                book_sum += 1

        pp = 0.0
        for p in range(perms.shape[0]):
            click_order = np.zeros(n, dtype=np.int64)
            for j in range(kk):
                click_order[idx_clicked[j]] = perms[p, j]

            if book_sum == 0:
                denomi = 1.0 + _sum_exp_ne0_nb(pd_u, click_order) + _sum_exp_eq_nb(pd_s, click_order, 0)
                pp_tmp = -np.log(denomi)
                for ii in range(kk, 0, -1):
                    val = _scalar_by_order_nb(pd_s, click_order, ii)
                    denomi += np.exp(val)
                    pp_tmp += val - np.log(denomi)
                pp_tmp -= _log_click_condition_nb(pd_s, 1e-20)
                pp += np.exp(pp_tmp)
            else:
                booked_last = False
                for i in range(n):
                    if click_order[i] == kk and select[i]:
                        booked_last = True
                        break

                if not booked_last:
                    denomi = 1.0 + _sum_exp_ne0_nb(pd_u, click_order) + _sum_exp_eq_nb(pd_s, click_order, 0)
                    val_u = _scalar_by_select_nb(pd_u, select)
                    pp_tmp = val_u - np.log(denomi)
                    for ii in range(kk, 0, -1):
                        val = _scalar_by_order_nb(pd_s, click_order, ii)
                        denomi += np.exp(val)
                        pp_tmp += val - np.log(denomi)
                    pp_tmp -= _log_click_condition_nb(pd_s, 1e-20)
                    pp_tmp = np.exp(pp_tmp)
                else:
                    pp_tmp = 0.0
                    if kk > 1:
                        for iii in range(kk, 1, -1):
                            denomi = (
                                1.0
                                + _sum_exp_ne0_not_select_nb(pd_u, click_order, select)
                                + _sum_exp_eq_nb(pd_s, click_order, 0)
                            )
                            pp0 = 0.0
                            for ii in range(kk, iii - 1, -1):
                                val = _scalar_by_order_nb(pd_s, click_order, ii)
                                denomi += np.exp(val)
                                pp0 += val - np.log(denomi)
                            val_u = _scalar_by_select_nb(pd_u, select)
                            denomi += np.exp(val_u)
                            pp0 += val_u - np.log(denomi)
                            for ii in range(iii - 1, 0, -1):
                                val = _scalar_by_order_nb(pd_s, click_order, ii)
                                denomi += np.exp(val)
                                pp0 += val - np.log(denomi)
                            pp_tmp += np.exp(pp0)

                    denomi = 1.0 + _sum_exp_ne0_nb(pd_u, click_order) + _sum_exp_eq_nb(pd_s, click_order, 0)
                    val_u = _scalar_by_select_nb(pd_u, select)
                    pp0 = val_u - np.log(denomi)
                    for ii in range(kk, 0, -1):
                        val = _scalar_by_order_nb(pd_s, click_order, ii)
                        denomi += np.exp(val)
                        pp0 += val - np.log(denomi)
                    pp_tmp += np.exp(pp0)

                    denomi = (
                        1.0
                        + _sum_exp_ne0_not_select_nb(pd_u, click_order, select)
                        + _sum_exp_eq_nb(pd_s, click_order, 0)
                    )
                    pp0 = 0.0
                    for ii in range(kk, 0, -1):
                        val = _scalar_by_order_nb(pd_s, click_order, ii)
                        denomi += np.exp(val)
                        pp0 += val - np.log(denomi)
                    val_u = _scalar_by_select_nb(pd_u, select)
                    denomi += np.exp(val_u)
                    pp0 += val_u - np.log(denomi)
                    pp_tmp += np.exp(pp0)

                    pp_tmp = np.log(pp_tmp + (pp_tmp == 0.0) * 1e-16)
                    pp_tmp -= _log_click_condition_nb(pd_s, 1e-16)

                pp += np.exp(pp_tmp)

        return np.log(pp + 1e-20)


    @njit(cache=True)
    def _pathprobability_conditioning3_gc_components_nb(
        click: np.ndarray,
        book: np.ndarray,
        pd_s: np.ndarray,
        pd_u: np.ndarray,
        sig: float,
        eps_draws: np.ndarray,
        perms: np.ndarray,
    ) -> float:
        kk = pd_s.shape[0]
        n_draws = eps_draws.shape[1]
        ppall = np.empty(n_draws, dtype=np.float64)
        tmp_s = np.empty_like(pd_s, dtype=np.float64)
        tmp_u = np.empty_like(pd_u, dtype=np.float64)
        sig2 = sig * sig
        for i in range(n_draws):
            for r in range(kk):
                adj = -sig2 * eps_draws[r, i]
                tmp_s[r] = pd_s[r] + adj
                tmp_u[r] = pd_u[r] + adj
            ppall[i] = _pathprobability_conditioning_components_nb(click, book, tmp_s, tmp_u, perms)
        return _log_mean_exp_nb(ppall)


@lru_cache(maxsize=None)
def _cached_permutations(k: int) -> tuple[tuple[int, ...], ...]:
    """Cache permutations to avoid regenerating them for each consumer."""
    return tuple(itertools.permutations(range(1, k + 1)))


def _get_permutations(k: int) -> Iterable[tuple[int, ...]]:
    """Return cached permutations for small k, iterator otherwise."""
    if k <= 6:
        return _cached_permutations(k)
    return itertools.permutations(range(1, k + 1))


def _sum_exp(values: np.ndarray) -> float:
    """Stable helper for sum(exp(x)) with empty-array guard."""
    if values.size == 0:
        return 0.0
    return float(np.sum(np.exp(values)))


def _scalar(values: np.ndarray) -> float:
    """Extract a scalar from a 1-element array (MATLAB-style indexing)."""
    return float(np.asarray(values).reshape(-1)[0])


def _log_click_condition(pd_s: np.ndarray, eps: float = 1e-20) -> float:
    """Log probability of at least one click (selection correction)."""
    total = _sum_exp(pd_s)
    return float(np.log(total / (1.0 + total) + eps))


def _log_mean_exp(values: np.ndarray) -> float:
    """Compute log(mean(exp(values))) in a numerically stable way."""
    max_val = np.max(values)
    return float(max_val + np.log(np.mean(np.exp(values - max_val))))
