"""Tests for the fair-value signal, built on synthetic spreads with known answers.

The look-ahead test is the one that matters most. A signal that can see the
future is the standard way a backtest of a mean-reversion idea comes out
profitable and then loses money live, and it is invisible in the equity curve -
the curve looks better, not wrong.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from src.signal import fair_value as fv


def make_spreads(debits, ivs=None, front_dte=4, back_dte=11,
                 option_type="CE", start=date(2024, 1, 1), spot=20000.0):
    """Build a spread series with a chosen debit path, in % of spot terms."""
    n = len(debits)
    ivs = [16.0] * n if ivs is None else ivs
    rows = []
    for i, (deb, iv) in enumerate(zip(debits, ivs)):
        rows.append({
            "date": start + timedelta(days=i),
            "option_type": option_type,
            "strike": spot,
            "spot": spot,
            "front_expiry": start + timedelta(days=i + front_dte),
            "back_expiry": start + timedelta(days=i + back_dte),
            "front_dte": front_dte,
            "back_dte": back_dte,
            "front_kind": "weekly",
            "front_px": 100.0,
            "back_px": 100.0 + deb,
            "debit": deb,
            "debit_pct": deb / spot * 100.0,
            "front_iv": iv,
            "back_iv": iv - 1.0,          # front richer: a long calendar wants this
            "term_structure": 1.0,
            "front_tradeable": True,
            "back_tradeable": True,
            "lot_size": 50.0,
        })
    return pd.DataFrame(rows)


def test_uses_only_past_observations():
    """A large debit late in the series must not change the verdict early in it.

    Two series share their first 20 days and differ only afterwards. If any row
    in the shared prefix comes out different, the signal is reading forward.
    """
    prefix = [100.0] * 20
    a = make_spreads(prefix + [100.0] * 10)
    b = make_spreads(prefix + [900.0] * 10)

    ev_a = fv.evaluate(a, min_comparables=5)
    ev_b = fv.evaluate(b, min_comparables=5)

    shared = len(prefix)
    pd.testing.assert_series_equal(
        ev_a["z_score"].iloc[:shared], ev_b["z_score"].iloc[:shared])
    pd.testing.assert_series_equal(
        ev_a["signal"].iloc[:shared], ev_b["signal"].iloc[:shared])


def test_first_rows_report_insufficient_data():
    """With nothing behind them, early rows cannot have a distribution."""
    ev = fv.evaluate(make_spreads([100.0] * 20), min_comparables=8)
    assert (ev["signal"].iloc[:8] == fv.INSUFFICIENT).all()
    assert ev["z_score"].iloc[:8].isna().all()


def test_cheap_spread_is_a_buy():
    """History at 100, today at 70: below its own distribution, so a buy."""
    debits = [97.0, 103.0, 99.0, 101.0, 96.0, 104.0, 98.0, 102.0, 70.0]
    ev = fv.evaluate(make_spreads(debits), min_comparables=8)
    last = ev.iloc[-1]
    assert last["n_comparables"] == 8
    assert last["z_score"] < -fv.C.Z_ENTRY
    assert last["signal"] == fv.BUY


def test_rich_spread_is_a_sell():
    """Selling the calendar needs the back leg carrying the higher vol, which is
    the mirror of what a long calendar wants."""
    debits = [97.0, 103.0, 99.0, 101.0, 96.0, 104.0, 98.0, 102.0, 130.0]
    spreads = make_spreads(debits)
    spreads["term_structure"] = -1.0
    ev = fv.evaluate(spreads, min_comparables=8)
    last = ev.iloc[-1]
    assert last["z_score"] > fv.C.Z_ENTRY
    assert last["signal"] == fv.SELL


def test_spread_in_line_with_history_is_flat():
    debits = [97.0, 103.0, 99.0, 101.0, 96.0, 104.0, 98.0, 102.0, 100.2]
    ev = fv.evaluate(make_spreads(debits), min_comparables=8)
    assert ev.iloc[-1]["signal"] == fv.FLAT


def test_iv_bucket_excludes_other_regimes():
    """Days in a different vol regime are not comparable, however similar the
    debit looks."""
    debits = [100.0] * 8 + [70.0]
    ivs = [25.0] * 8 + [16.0]          # history sat at 25 vol, today is 16
    ev = fv.evaluate(make_spreads(debits, ivs=ivs), min_comparables=8)
    assert ev.iloc[-1]["signal"] == fv.INSUFFICIENT
    assert ev.iloc[-1]["n_comparables"] == 0


def test_back_dte_tolerance_widens_the_match():
    """The back leg's slack is what makes a distribution reachable at all."""
    history = [97.0, 103.0, 99.0, 101.0, 96.0, 104.0, 98.0, 102.0]
    rows = []
    for i, deb in enumerate(history + [70.0]):
        back = 11 + (i % 3)            # back leg wanders 11-13
        rows.append(make_spreads([deb], back_dte=back,
                                 start=date(2024, 1, 1) + timedelta(days=i)))
    spreads = pd.concat(rows, ignore_index=True)

    strict = fv.evaluate(spreads, method=fv.BUCKET, back_dte_tolerance=0,
                         min_comparables=8)
    loose = fv.evaluate(spreads, method=fv.BUCKET, back_dte_tolerance=2,
                        min_comparables=8)

    assert strict.iloc[-1]["signal"] == fv.INSUFFICIENT
    assert loose.iloc[-1]["n_comparables"] == 8
    assert loose.iloc[-1]["signal"] == fv.BUY


def test_term_structure_filter_blocks_unsupported_buys():
    """Cheap is not enough: a long calendar wants the sold leg carrying the
    higher vol. When it does not, the filter stands the trade down."""
    debits = [97.0, 103.0, 99.0, 101.0, 96.0, 104.0, 98.0, 102.0, 70.0]
    spreads = make_spreads(debits)
    spreads["term_structure"] = -1.0         # back leg richer than front

    with_filter = fv.evaluate(spreads, min_comparables=8,
                              use_term_structure_filter=True)
    without = fv.evaluate(spreads, min_comparables=8,
                          use_term_structure_filter=False)

    assert with_filter.iloc[-1]["signal"] == fv.FLAT
    assert without.iloc[-1]["signal"] == fv.BUY


def test_percent_of_spot_normalisation_survives_an_index_rally():
    """The same trade at a higher index level must not read as a richer spread.

    This is the failure the normalisation exists to prevent: NIFTY rose roughly
    35% over the sample, and premium scales with it, so raw points would drift
    upward on their own and every late-sample spread would look expensive.
    """
    early = make_spreads([97.0, 103.0, 99.0, 101.0, 96.0, 104.0, 98.0, 102.0],
                         spot=20000.0)
    late = make_spreads([135.0], spot=27000.0,
                        start=date(2024, 2, 1))       # same 0.5% of spot
    ev = fv.evaluate(pd.concat([early, late], ignore_index=True),
                     min_comparables=8)
    last = ev.iloc[-1]
    assert last["debit_pct"] == pytest.approx(0.5, abs=1e-9)
    assert abs(last["z_score"]) < 0.5      # sits on its own historical mean
    assert last["signal"] == fv.FLAT

    # The same debit in points would have read as far too rich at the new level.
    assert last["debit"] > early["debit"].max()


def test_degenerate_distribution_does_not_trade():
    """Eight identical comparables have no width, so today cannot be scored
    against them however far away it is."""
    ev = fv.evaluate(make_spreads([100.0] * 8 + [70.0]), min_comparables=8)
    last = ev.iloc[-1]
    assert last["n_comparables"] == 8
    assert last["signal"] == fv.INSUFFICIENT
    assert np.isnan(last["z_score"])


def test_z_score_and_percentile_agree_on_direction():
    debits = [97.0, 103.0, 99.0, 101.0, 96.0, 104.0, 98.0, 102.0, 85.0]
    ev = fv.evaluate(make_spreads(debits), min_comparables=8)
    last = ev.iloc[-1]
    assert last["z_score"] < 0
    assert last["percentile"] == 0.0        # cheaper than every comparable
    assert last["fv_mean_pct"] > last["debit_pct"]


def test_insufficient_data_never_produces_a_trade():
    ev = fv.evaluate(make_spreads([100.0] * 5), min_comparables=8)
    assert set(ev["signal"]) == {fv.INSUFFICIENT}
    assert not ev["signal"].isin([fv.BUY, fv.SELL]).any()


# --------------------------------------------------------------- kNN matching

def test_distance_cap_rejects_dissimilar_states():
    """Nearest is not the same as similar. A day whose closest neighbours are all
    far away in state space has no comparables, and must say so rather than
    scoring itself against whatever happened to be least distant."""
    history = make_spreads([97.0, 103.0, 99.0, 101.0, 96.0, 104.0, 98.0, 102.0],
                           front_dte=4, ivs=[16.0] * 8)
    # Today sits 20 vol points away - 40 scaled units, far outside the cap.
    today = make_spreads([70.0], front_dte=4, ivs=[36.0],
                         start=date(2024, 2, 1))
    ev = fv.evaluate(pd.concat([history, today], ignore_index=True),
                     min_comparables=8)
    last = ev.iloc[-1]
    assert last["n_comparables"] == 0
    assert last["signal"] == fv.INSUFFICIENT


def test_knn_uses_near_states_when_no_exact_match_exists():
    """The case the bucket rule fails on: nothing matches exactly, but the
    neighbouring days-to-expiry are close enough to be informative."""
    rows = []
    for i, deb in enumerate([97.0, 103.0, 99.0, 101.0, 96.0, 104.0, 98.0, 102.0]):
        rows.append(make_spreads([deb], front_dte=5, back_dte=12,
                                 start=date(2024, 1, 1) + timedelta(days=i)))
    # Today's front leg has one day less than anything in history.
    rows.append(make_spreads([70.0], front_dte=4, back_dte=12,
                             start=date(2024, 2, 1)))
    spreads = pd.concat(rows, ignore_index=True)

    bucket = fv.evaluate(spreads, method=fv.BUCKET, min_comparables=8)
    knn = fv.evaluate(spreads, method=fv.KNN, min_comparables=8)

    assert bucket.iloc[-1]["signal"] == fv.INSUFFICIENT
    assert knn.iloc[-1]["n_comparables"] == 8
    assert knn.iloc[-1]["signal"] == fv.BUY


def test_closer_neighbours_carry_more_weight():
    """A near state should move the fair value more than a distant one."""
    near = make_spreads([100.0] * 8, front_dte=4, ivs=[16.0] * 8)
    far = make_spreads([200.0] * 8, front_dte=4, ivs=[17.0] * 8,
                       start=date(2024, 3, 1))
    today = make_spreads([150.0], front_dte=4, ivs=[16.1],
                         start=date(2024, 6, 1))
    ev = fv.evaluate(pd.concat([near, far, today], ignore_index=True),
                     k=16, min_comparables=8)
    last = ev.iloc[-1]
    # Both clusters are in range, but today sits nearer the 100 group, so the
    # weighted fair value must fall below the midpoint of 100 and 200.
    midpoint = make_spreads([150.0])["debit_pct"].iloc[0]
    assert last["fv_mean_pct"] < midpoint


def test_match_distance_is_reported():
    """Every scored day carries how close its comparables actually were, so a
    weak match is visible rather than implied."""
    spreads = make_spreads([97.0, 103.0, 99.0, 101.0, 96.0, 104.0, 98.0, 102.0,
                            70.0])
    ev = fv.evaluate(spreads, min_comparables=8)
    assert np.isfinite(ev.iloc[-1]["match_distance"])
    assert ev.iloc[-1]["match_distance"] >= 0


def test_knn_still_only_looks_backwards():
    """The look-ahead guarantee must survive the change of matching method."""
    prefix = [97.0, 103.0, 99.0, 101.0, 96.0, 104.0, 98.0, 102.0] * 3
    a = make_spreads(prefix + [100.0] * 10)
    b = make_spreads(prefix + [900.0] * 10)
    ev_a = fv.evaluate(a, min_comparables=5)
    ev_b = fv.evaluate(b, min_comparables=5)
    n = len(prefix)
    pd.testing.assert_series_equal(ev_a["z_score"].iloc[:n],
                                   ev_b["z_score"].iloc[:n])


def test_comparables_match_what_the_verdict_used():
    """The dashboard plots `comparables()` while the verdict comes from
    `evaluate()`. If those two ever disagree, the chart is explaining a signal
    that was not the one produced."""
    debits = [97.0, 103.0, 99.0, 101.0, 96.0, 104.0, 98.0, 102.0, 70.0]
    spreads = make_spreads(debits)
    ev = fv.evaluate(spreads, min_comparables=8)
    last = ev.iloc[-1]

    comps = fv.comparables(spreads, "CE", last["date"])
    assert len(comps) == last["n_comparables"]

    # Re-deriving the fair value from the returned rows must reproduce it.
    w = comps["weight"].to_numpy()
    mean = (comps["debit_pct"].to_numpy() * w).sum() / w.sum()
    assert mean == pytest.approx(last["fv_mean_pct"], rel=1e-9)


def test_comparables_respects_the_bucket_method():
    debits = [97.0, 103.0, 99.0, 101.0, 96.0, 104.0, 98.0, 102.0, 70.0]
    spreads = make_spreads(debits)
    ev = fv.evaluate(spreads, method=fv.BUCKET, min_comparables=8)
    comps = fv.comparables(spreads, "CE", ev.iloc[-1]["date"], method=fv.BUCKET)
    assert len(comps) == ev.iloc[-1]["n_comparables"]
