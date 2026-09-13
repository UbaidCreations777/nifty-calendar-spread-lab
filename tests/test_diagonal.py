"""Tests for the diagonal explorer.

The thing to pin down is what makes a past day comparable. A diagonal is
described by five numbers rather than three, and the one most easily got wrong is
moneyness: matching on the strike number instead of its distance from the forward
would compare a 24,000 call struck 3% out of the money against one struck 25% out
of the money, purely because the index moved between them.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from src.signal import diagonal as dg
from src.signal import fair_value as fv


def make_chain(days, spot=20000.0, strikes=(19800, 19900, 20000, 20100, 20200),
               expiry_dtes=(4, 11, 18), px=lambda dte, k, spot: 50.0 + dte):
    """A synthetic chain: every day carries the same expiry structure."""
    rows = []
    for d in days:
        for dte in expiry_dtes:
            expiry = d + dt.timedelta(days=dte)
            for k in strikes:
                rows.append({
                    "date": d, "expiry": expiry, "dte": dte, "strike": float(k),
                    "option_type": "CE", "px": float(px(dte, k, spot)),
                    "spot": float(spot), "forward": float(spot),
                    "iv": 0.16, "moneyness": k / spot - 1.0,
                    "tradeable": True,
                })
    return pd.DataFrame(rows)


DAYS = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(20)]


def test_leg_is_matched_on_moneyness_not_strike_number():
    """The index moved 20% between the two sessions. The comparable contract is
    the one the same distance from the forward, not the same strike."""
    early = make_chain([DAYS[0]], spot=20000.0,
                       strikes=(19800, 20000, 20200))
    late = make_chain([DAYS[1]], spot=24000.0,
                      strikes=(23800, 24000, 24200))
    lookup = dg.Lookup(pd.concat([early, late], ignore_index=True))

    # Ask for at-the-money on each day.
    a = lookup.leg_at(DAYS[0], DAYS[0] + dt.timedelta(days=4), "CE", 0.0)
    b = lookup.leg_at(DAYS[1], DAYS[1] + dt.timedelta(days=4), "CE", 0.0)
    assert a["strike"] == 20000.0
    assert b["strike"] == 24000.0


def test_expiry_is_matched_within_tolerance():
    lookup = dg.Lookup(make_chain([DAYS[0]]))
    day = DAYS[0]
    assert lookup.expiry_near(day, 11, tolerance=0)[1] == 11
    assert lookup.expiry_near(day, 12, tolerance=3)[1] == 11
    assert lookup.expiry_near(day, 30, tolerance=3) is None


def test_structure_needs_two_different_expiries():
    """Both legs landing on the same expiry is a vertical, not a diagonal."""
    lookup = dg.Lookup(make_chain([DAYS[0]]))
    out = dg.structure_on(lookup, DAYS[0], "CE", front_dte=11, back_dte=12,
                          front_moneyness=0.0, back_moneyness=0.0,
                          dte_tolerance=3)
    assert out is None


def test_debit_is_back_leg_minus_front_leg():
    lookup = dg.Lookup(make_chain([DAYS[0]]))
    out = dg.structure_on(lookup, DAYS[0], "CE", front_dte=4, back_dte=18,
                          front_moneyness=0.0, back_moneyness=0.0)
    assert out["front_px"] == pytest.approx(54.0)     # 50 + dte
    assert out["back_px"] == pytest.approx(68.0)
    assert out["debit"] == pytest.approx(14.0)
    assert out["debit_pct"] == pytest.approx(14.0 / 20000.0 * 100.0)


def test_history_stops_before_the_evaluation_date():
    """Point in time, exactly as in the calendar model."""
    lookup = dg.Lookup(make_chain(DAYS))
    as_of = DAYS[10]
    target = dg.structure_on(lookup, as_of, "CE", 4, 18, 0.0, 0.0)
    past = dg.history(lookup, target, as_of, k=100)
    assert not past.empty
    assert past["date"].max() < as_of


def test_distance_grows_with_each_dimension():
    base = {"front_dte": 4, "back_dte": 18, "front_iv": 16.0,
            "front_moneyness": 0.0, "back_moneyness": 0.0}
    assert dg._distance(base, base) == pytest.approx(0.0)

    for field, shift in [("front_dte", 1), ("back_dte", 3), ("front_iv", 0.5),
                         ("front_moneyness", dg.STATE_SCALE_MONEYNESS),
                         ("back_moneyness", dg.STATE_SCALE_MONEYNESS)]:
        moved = {**base, field: base[field] + shift}
        # Each scale is defined so one step is one unit of distance.
        assert dg._distance(moved, base) == pytest.approx(1.0, abs=1e-9)


def test_a_far_state_is_dropped_by_the_cap():
    lookup = dg.Lookup(make_chain(DAYS))
    as_of = DAYS[10]
    target = dg.structure_on(lookup, as_of, "CE", 4, 18, 0.0, 0.0)
    target["front_iv"] = 60.0            # nothing in history is near this vol
    past = dg.history(lookup, target, as_of)
    assert past.empty


def test_evaluate_reports_insufficient_rather_than_guessing():
    lookup = dg.Lookup(make_chain(DAYS[:3]))
    as_of = DAYS[2]
    verdict, past = dg.evaluate(
        lookup, "CE", as_of,
        front_expiry=as_of + dt.timedelta(days=4), front_strike=20000.0,
        back_expiry=as_of + dt.timedelta(days=18), back_strike=20100.0,
        min_comparables=8)
    assert verdict["signal"] == fv.INSUFFICIENT
    assert np.isnan(verdict["z_score"])


def test_evaluate_prices_a_diagonal_against_its_history():
    """History sits flat, today is dear: the structure reads rich."""
    chain = make_chain(DAYS)
    # Make the last day's back leg much more expensive.
    last = DAYS[-1]
    mask = (chain["date"] == last) & (chain["dte"] == 18)
    chain.loc[mask, "px"] = chain.loc[mask, "px"] + 40.0
    # Give the history some dispersion so the z-score is trusted.
    wobble = chain["date"].map({d: (i % 3) - 1 for i, d in enumerate(DAYS)})
    chain.loc[chain["dte"] == 18, "px"] += wobble[chain["dte"] == 18] * 2.0

    lookup = dg.Lookup(chain)
    verdict, past = dg.evaluate(
        lookup, "CE", last,
        front_expiry=last + dt.timedelta(days=4), front_strike=20000.0,
        back_expiry=last + dt.timedelta(days=18), back_strike=20000.0,
        min_comparables=8)

    assert verdict["n_comparables"] >= 8
    assert verdict["z_score"] > 0
    assert verdict["raw_signal"] in (fv.SELL, fv.FLAT)


def test_same_strike_diagonal_is_just_a_calendar():
    """The calendar is the special case, and must come out identical."""
    lookup = dg.Lookup(make_chain(DAYS))
    day = DAYS[5]
    out = dg.structure_on(lookup, day, "CE", 4, 18, 0.0, 0.0)
    assert out["front_strike"] == out["back_strike"]
