"""Implied volatility by Newton-Raphson with a bisection fallback.

Newton converges in a handful of iterations from a decent starting guess, but it
is not safe on its own: vega collapses toward zero for deep in- and out-of-the-
money options, and dividing by a near-zero derivative throws the iterate far from
the root. So every contract Newton fails to solve is handed to bisection, which
cannot diverge because it only ever halves a bracket that already contains the
root. The two together solve the whole chain rather than just its liquid middle.
"""
from __future__ import annotations

import numpy as np

from . import black_scholes as bs
from .. import config as C


def brenner_subrahmanyam_guess(price, F, T):
    """A closed-form ATM approximation used as the Newton seed.

    sigma ~= sqrt(2*pi/T) * price / F

    It is only exact at the money, but it lands close enough that Newton needs
    far fewer steps than a fixed 20%-vol start.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        guess = np.sqrt(2.0 * np.pi / T) * (price / F)
    return np.clip(np.nan_to_num(guess, nan=0.2), 0.05, 1.5)


def implied_vol(market_price, F, K, T, r, option_type,
                tol: float = C.IV_SOLVER_TOL,
                max_iter: int = C.IV_SOLVER_MAX_ITER) -> np.ndarray:
    """Vectorised IV solve. Returns NaN where no arbitrage-free root exists."""
    market_price = np.asarray(market_price, dtype=float)
    F = np.asarray(F, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    r = np.full(np.broadcast(market_price, F, K, T).shape, r, dtype=float) \
        if np.isscalar(r) else np.asarray(r, dtype=float)
    is_call = bs._is_call(option_type)

    lo, hi = C.IV_BOUNDS
    df = np.exp(-r * T)

    # No volatility can price an option below intrinsic or above the forward, so
    # those quotes are rejected rather than solved to a junk root.
    intrinsic = df * np.where(is_call, np.maximum(F - K, 0.0), np.maximum(K - F, 0.0))
    upper = df * np.where(is_call, F, K)
    valid = (
        np.isfinite(market_price) & np.isfinite(F) & (T > 0)
        & (market_price >= C.MIN_OPTION_PRICE)
        & (market_price >= intrinsic - 1e-8) & (market_price <= upper + 1e-8)
    )

    sigma = brenner_subrahmanyam_guess(market_price, F, T)
    sigma = np.where(valid, sigma, np.nan)

    # ---- Newton-Raphson
    # The Newton step is evaluated across the whole array and then masked, so the
    # contracts with no vega divide by zero before being discarded. That is the
    # intended path, not a numerical accident, so the warning is silenced here.
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for _ in range(max_iter):
            active = valid & np.isfinite(sigma)
            if not active.any():
                break
            model = bs.price(F, K, T, sigma, r, option_type)
            diff = model - market_price
            vega = bs.greeks(F, K, T, sigma, r, option_type)["vega"] * 100.0  # raw
            converged = np.abs(diff) < tol
            step_ok = active & ~converged & (vega > 1e-8)
            if not step_ok.any():
                break
            sigma = np.where(step_ok, sigma - diff / vega, sigma)
            sigma = np.where(step_ok, np.clip(sigma, lo, hi), sigma)

    # ---- bisection for whatever Newton could not land
    model = bs.price(F, K, T, sigma, r, option_type)
    failed = valid & (~np.isfinite(sigma) | (np.abs(model - market_price) > 1e-3))
    if failed.any():
        sigma = np.where(failed, _bisect(market_price, F, K, T, r, option_type,
                                         failed, lo, hi, tol), sigma)

    return np.where(valid, sigma, np.nan)


def _bisect(market_price, F, K, T, r, option_type, mask, lo, hi, tol):
    a = np.full_like(np.asarray(market_price, dtype=float), lo)
    b = np.full_like(np.asarray(market_price, dtype=float), hi)
    mid = (a + b) / 2.0
    for _ in range(80):
        mid = (a + b) / 2.0
        val = bs.price(F, K, T, mid, r, option_type) - market_price
        too_low = val < 0                      # model cheaper than market -> raise vol
        a = np.where(mask & too_low, mid, a)
        b = np.where(mask & ~too_low, mid, b)
        if np.all((b - a) < tol):
            break
    return mid
