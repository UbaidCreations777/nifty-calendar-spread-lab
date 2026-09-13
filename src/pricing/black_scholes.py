"""Black-76 option pricing and Greeks, implemented from first principles.

Why Black-76 on a forward rather than textbook Black-Scholes on spot: NIFTY pays
an irregular dividend stream, so any spot-based model needs a dividend-yield
assumption that is wrong on most days. Pricing off the forward implied by the
option chain itself (see `forward.py`) removes that free parameter entirely - the
market tells us the carry instead of us guessing it.

Only the normal CDF is borrowed (via math.erf); the pricing and every Greek is
written out so each term can be checked against its derivation.
"""
from __future__ import annotations

import numpy as np

SQRT_2PI = np.sqrt(2.0 * np.pi)


def norm_pdf(x):
    """Standard normal density."""
    return np.exp(-0.5 * np.square(x)) / SQRT_2PI


def norm_cdf(x):
    """Standard normal CDF, built on the error function.

    N(x) = 0.5 * [1 + erf(x / sqrt(2))]
    """
    from scipy.special import erf          # vectorised erf; math.erf is scalar
    return 0.5 * (1.0 + erf(np.asarray(x, dtype=float) / np.sqrt(2.0)))


def _d1_d2(F, K, T, sigma):
    F = np.asarray(F, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        vol_sqrt_t = sigma * np.sqrt(T)
        d1 = (np.log(F / K) + 0.5 * np.square(sigma) * T) / vol_sqrt_t
        d2 = d1 - vol_sqrt_t
    return d1, d2, vol_sqrt_t


def price(F, K, T, sigma, r, option_type):
    """Black-76 price of a European option on a forward.

    call = e^{-rT} [F N(d1) - K N(d2)]
    put  = e^{-rT} [K N(-d2) - F N(-d1)]
    """
    d1, d2, vol_sqrt_t = _d1_d2(F, K, T, sigma)
    df = np.exp(-np.asarray(r, dtype=float) * np.asarray(T, dtype=float))
    is_call = _is_call(option_type)

    call = df * (F * norm_cdf(d1) - K * norm_cdf(d2))
    put = df * (K * norm_cdf(-d2) - F * norm_cdf(-d1))
    out = np.where(is_call, call, put)

    # At or past expiry, and for a zero-vol contract, the option is worth its
    # discounted intrinsic value - the formula above divides by zero there.
    intrinsic = df * np.where(is_call, np.maximum(F - K, 0.0), np.maximum(K - F, 0.0))
    degenerate = (vol_sqrt_t <= 0) | ~np.isfinite(out)
    return np.where(degenerate, intrinsic, out)


def greeks(F, K, T, sigma, r, option_type) -> dict:
    """Delta, gamma, vega, theta and rho.

    Delta and gamma are with respect to the forward. Vega is quoted per one
    volatility point (1.0 = 1%) and theta per calendar day, which is how both are
    read on a desk rather than the per-unit-of-sigma, per-year raw partials.
    """
    d1, d2, vol_sqrt_t = _d1_d2(F, K, T, sigma)
    F = np.asarray(F, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    r = np.asarray(r, dtype=float)
    df = np.exp(-r * T)
    is_call = _is_call(option_type)
    pdf_d1 = norm_pdf(d1)

    delta = np.where(is_call, df * norm_cdf(d1), -df * norm_cdf(-d1))
    gamma = df * pdf_d1 / (F * vol_sqrt_t)
    vega = df * F * pdf_d1 * np.sqrt(T)

    # theta = -F e^{-rT} n(d1) sigma / (2 sqrt(T)) + r e^{-rT} [...]
    decay = -df * F * pdf_d1 * sigma / (2.0 * np.sqrt(T))
    carry = np.where(
        is_call,
        r * df * (F * norm_cdf(d1) - K * norm_cdf(d2)),
        r * df * (K * norm_cdf(-d2) - F * norm_cdf(-d1)),
    )
    theta = decay + carry

    px = price(F, K, T, sigma, r, option_type)
    rho = -T * px

    degenerate = (vol_sqrt_t <= 0) | ~np.isfinite(d1)
    zero = np.zeros_like(np.asarray(delta, dtype=float))
    return {
        "delta": np.where(degenerate, np.where(is_call, (F > K) * 1.0, (F < K) * -1.0), delta),
        "gamma": np.where(degenerate, zero, gamma),
        "vega": np.where(degenerate, zero, vega / 100.0),      # per 1 vol point
        "theta": np.where(degenerate, zero, theta / 365.0),    # per calendar day
        "rho": np.where(degenerate, zero, rho / 100.0),        # per 1% rate move
    }


def _is_call(option_type):
    arr = np.asarray(option_type)
    if arr.dtype.kind in "US":
        return np.char.upper(arr.astype(str)) == "CE"
    return arr.astype(bool)
