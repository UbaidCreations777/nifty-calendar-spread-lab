"""Validates the hand-written pricer against py_vollib's Black-76 reference.

This is the test that makes the rest of the project trustworthy: if the pricing
or the IV solver is wrong, every signal downstream is wrong in a way that still
looks plausible on a chart.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.pricing import black_scholes as bs
from src.pricing.implied_vol import implied_vol

vollib_price = pytest.importorskip("py_vollib.black").black
vollib_greeks = pytest.importorskip("py_vollib.black.greeks.analytical")
vollib_iv = pytest.importorskip("py_vollib.black.implied_volatility")

R = 0.065
CASES = [
    # F,      K,      T,       sigma, type
    (23500, 23500, 7 / 365, 0.14, "CE"),    # ATM weekly
    (23500, 23500, 30 / 365, 0.13, "CE"),   # ATM monthly
    (23500, 24000, 7 / 365, 0.16, "CE"),    # OTM call
    (23500, 23000, 7 / 365, 0.18, "PE"),    # OTM put
    (23500, 22000, 30 / 365, 0.22, "PE"),   # deep OTM put
    (23500, 25000, 45 / 365, 0.15, "CE"),   # far OTM, longer dated
    (23500, 22500, 14 / 365, 0.20, "CE"),   # ITM call
    (19000, 19000, 3 / 365, 0.11, "CE"),    # low spot regime, near expiry
]


def _flag(option_type: str) -> str:
    return "c" if option_type == "CE" else "p"


@pytest.mark.parametrize("F,K,T,sigma,opt", CASES)
def test_price_matches_vollib(F, K, T, sigma, opt):
    mine = float(bs.price(F, K, T, sigma, R, opt))
    ref = vollib_price(_flag(opt), F, K, T, R, sigma)
    assert mine == pytest.approx(ref, rel=1e-9, abs=1e-8)


@pytest.mark.parametrize("F,K,T,sigma,opt", CASES)
def test_greeks_match_vollib(F, K, T, sigma, opt):
    mine = bs.greeks(F, K, T, sigma, R, opt)
    flag = _flag(opt)
    assert float(mine["delta"]) == pytest.approx(
        vollib_greeks.delta(flag, F, K, T, R, sigma), rel=1e-6, abs=1e-8)
    assert float(mine["gamma"]) == pytest.approx(
        vollib_greeks.gamma(flag, F, K, T, R, sigma), rel=1e-6, abs=1e-10)
    # vollib quotes vega per vol point and theta per day, same as this module.
    assert float(mine["vega"]) == pytest.approx(
        vollib_greeks.vega(flag, F, K, T, R, sigma), rel=1e-6, abs=1e-8)
    assert float(mine["theta"]) == pytest.approx(
        vollib_greeks.theta(flag, F, K, T, R, sigma), rel=1e-5, abs=1e-6)


@pytest.mark.parametrize("F,K,T,sigma,opt", CASES)
def test_implied_vol_round_trips(F, K, T, sigma, opt):
    """Price at a known vol, then solve it back out."""
    px = float(bs.price(F, K, T, sigma, R, opt))
    solved = float(implied_vol(px, F, K, T, R, opt))
    assert solved == pytest.approx(sigma, abs=1e-6)


@pytest.mark.parametrize("F,K,T,sigma,opt", CASES)
def test_implied_vol_matches_vollib(F, K, T, sigma, opt):
    px = float(bs.price(F, K, T, sigma, R, opt))
    mine = float(implied_vol(px, F, K, T, R, opt))
    ref = vollib_iv.implied_volatility(px, F, K, R, T, _flag(opt))
    assert mine == pytest.approx(ref, abs=1e-5)


def test_implied_vol_vectorised_matches_scalar():
    F = np.array([23500.0, 23500.0, 19000.0])
    K = np.array([23500.0, 24000.0, 19000.0])
    T = np.array([7 / 365, 30 / 365, 3 / 365])
    sigma = np.array([0.14, 0.16, 0.11])
    opt = np.array(["CE", "CE", "PE"])
    px = bs.price(F, K, T, sigma, R, opt)
    solved = implied_vol(px, F, K, T, R, opt)
    np.testing.assert_allclose(solved, sigma, atol=1e-6)


def test_rejects_arbitrage_violating_quotes():
    """A price below intrinsic has no implied vol and must return NaN, not a
    number that silently poisons the signal database."""
    F, K, T = 23500.0, 22000.0, 7 / 365
    intrinsic = np.exp(-R * T) * (F - K)
    assert np.isnan(float(implied_vol(intrinsic * 0.5, F, K, T, R, "CE")))
    assert np.isnan(float(implied_vol(0.0, F, K, T, R, "CE")))


def test_rejects_sub_tick_prices():
    """A far OTM contract can price below NSE's 0.05 tick. Such a number is not a
    quote anyone could trade, and any vol over a wide range reproduces it, so the
    solver must refuse it instead of returning whichever root it landed on."""
    F, K, T = 23500.0, 30000.0, 3 / 365
    px = float(bs.price(F, K, T, 0.35, R, "CE"))
    assert px < 0.05
    assert np.isnan(float(implied_vol(px, F, K, T, R, "CE")))


def test_high_vol_far_otm_round_trips():
    """A vol regime far from the ATM seed, still a tradeable price."""
    F, K, T, sigma = 23500.0, 28000.0, 7 / 365, 1.2
    px = float(bs.price(F, K, T, sigma, R, "CE"))
    assert px > 0.05
    assert float(implied_vol(px, F, K, T, R, "CE")) == pytest.approx(sigma, abs=1e-5)


def test_bisection_solves_without_newton():
    """The fallback has to stand on its own, since it is what catches the cases
    Newton walks away from."""
    from src.pricing.implied_vol import _bisect

    F, K, T, sigma = np.array([23500.0]), np.array([24500.0]), np.array([10 / 365]), 0.28
    px = bs.price(F, K, T, sigma, R, "CE")
    solved = _bisect(px, F, K, T, np.array([R]), np.array(["CE"]),
                     np.array([True]), 0.01, 3.0, 1e-8)
    assert solved.item() == pytest.approx(sigma, abs=1e-5)
