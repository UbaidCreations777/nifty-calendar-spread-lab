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


# ------------------------------------------------- several pairs per session

def make_pair_row(deb, front_dte, back_dte, day, iv=16.0, spot=20000.0,
                  option_type="CE"):
    return {
        "date": day, "option_type": option_type, "strike": spot, "spot": spot,
        "front_expiry": day + timedelta(days=front_dte),
        "back_expiry": day + timedelta(days=back_dte),
        "front_dte": front_dte, "back_dte": back_dte, "front_kind": "weekly",
        "back_kind": "monthly", "front_px": 100.0, "back_px": 100.0 + deb,
        "debit": deb, "debit_pct": deb / spot * 100.0, "front_iv": iv,
        "back_iv": iv - 1.0, "term_structure": 1.0,
        "front_tradeable": True, "back_tradeable": True, "lot_size": 50.0,
    }


def test_other_pairs_from_today_are_not_comparables():
    """With several expiry pairs per session, the rows sitting above this one in
    the frame include today's other pairs. Those were not knowable when today
    began, so they must not enter the distribution."""
    history = [97.0, 103.0, 99.0, 101.0, 96.0, 104.0, 98.0, 102.0]
    rows = [make_pair_row(d, 4, 11, date(2024, 1, 1) + timedelta(days=i))
            for i, d in enumerate(history)]

    today = date(2024, 1, 20)
    # Three pairs priced on the same session, the target one last.
    rows.append(make_pair_row(500.0, 4, 11, today))
    rows.append(make_pair_row(600.0, 4, 11, today))
    rows.append(make_pair_row(70.0, 4, 11, today))
    spreads = pd.DataFrame(rows)

    ev = fv.evaluate(spreads, min_comparables=8)
    scored = ev[ev["date"] == today]

    # Each of today's rows sees the same eight historical days - never each other.
    assert (scored["n_comparables"] == 8).all()
    assert scored["fv_mean_pct"].nunique() == 1


def test_evaluate_one_matches_evaluate_for_the_same_pair():
    """The explorer and the series must agree, or the dashboard is describing a
    different model from the one being backtested."""
    history = [97.0, 103.0, 99.0, 101.0, 96.0, 104.0, 98.0, 102.0]
    rows = [make_pair_row(d, 4, 11, date(2024, 1, 1) + timedelta(days=i))
            for i, d in enumerate(history)]
    today = date(2024, 1, 20)
    rows.append(make_pair_row(70.0, 4, 11, today))
    spreads = pd.DataFrame(rows)

    from_series = fv.evaluate(spreads, min_comparables=8).iloc[-1]
    one = fv.evaluate_one(spreads, "CE", today, front_dte=4, back_dte=11,
                          min_comparables=8)

    assert one["signal"] == from_series["signal"]
    assert one["z_score"] == pytest.approx(from_series["z_score"])
    assert one["n_comparables"] == from_series["n_comparables"]


def test_evaluate_one_rejects_a_pair_that_does_not_exist():
    spreads = pd.DataFrame([make_pair_row(100.0, 4, 11, date(2024, 1, 1))])
    out = fv.evaluate_one(spreads, "CE", date(2024, 1, 1), front_dte=9,
                          back_dte=40)
    assert out["signal"] == fv.INSUFFICIENT


# ----------------------------------------------------- the unfiltered verdict

def test_raw_signal_records_what_the_z_score_alone_said():
    """The filter only ever vetoes, so the dashboard needs the call it vetoed in
    order to explain a 'no trade' the reader would otherwise not be able to
    account for."""
    debits = [97.0, 103.0, 99.0, 101.0, 96.0, 104.0, 98.0, 102.0, 70.0]
    spreads = make_spreads(debits)
    spreads["term_structure"] = -1.0         # does not support a long calendar

    ev = fv.evaluate(spreads, min_comparables=8)
    last = ev.iloc[-1]
    assert last["raw_signal"] == fv.BUY      # cheap on the numbers
    assert last["signal"] == fv.FLAT         # but stood down by the filter


def test_sell_side_is_produced_not_suppressed():
    """Selling a rich spread is a signal the model emits; only the backtest
    declines to trade it."""
    debits = [97.0, 103.0, 99.0, 101.0, 96.0, 104.0, 98.0, 102.0, 130.0]
    spreads = make_spreads(debits)
    spreads["term_structure"] = -1.0         # supports a short calendar
    ev = fv.evaluate(spreads, min_comparables=8)
    assert ev.iloc[-1]["raw_signal"] == fv.SELL
    assert ev.iloc[-1]["signal"] == fv.SELL


def test_raw_signal_equals_signal_when_the_filter_is_off():
    debits = [97.0, 103.0, 99.0, 101.0, 96.0, 104.0, 98.0, 102.0, 70.0]
    spreads = make_spreads(debits)
    spreads["term_structure"] = -1.0
    ev = fv.evaluate(spreads, min_comparables=8,
                     use_term_structure_filter=False)
    assert ev.iloc[-1]["signal"] == ev.iloc[-1]["raw_signal"] == fv.BUY


def test_evaluate_one_also_reports_the_raw_signal():
    debits = [97.0, 103.0, 99.0, 101.0, 96.0, 104.0, 98.0, 102.0, 130.0]
    spreads = make_spreads(debits)
    spreads["term_structure"] = -1.0
    one = fv.evaluate_one(spreads, "CE", spreads["date"].max(),
                          front_dte=4, back_dte=11, min_comparables=8)
    assert one["raw_signal"] == fv.SELL


# ------------------------------------------------------------------- basis

def test_basis_separates_days_that_otherwise_look_identical():
    """Two legs are priced off two different forwards, and the gap between them
    moves the debit on its own. A day that agrees on vol and on both expiries but
    not on the basis is pricing a different trade, and must not be treated as a
    comparable."""
    history = [97.0, 103.0, 99.0, 101.0, 96.0, 104.0, 98.0, 102.0]
    rows = [make_pair_row(d, 4, 11, date(2024, 1, 1) + timedelta(days=i))
            for i, d in enumerate(history)]
    today = date(2024, 1, 20)
    rows.append(make_pair_row(70.0, 4, 11, today))
    spreads = pd.DataFrame(rows)

    # History sat at one basis; today is far away from it.
    spreads["basis"] = 80.0
    spreads.loc[spreads["date"] == today, "basis"] = 400.0

    ignored = fv.evaluate(spreads, min_comparables=8, use_basis=False)
    counted = fv.evaluate(spreads, min_comparables=8, use_basis=True)

    assert ignored.iloc[-1]["n_comparables"] == 8      # basis invisible
    assert counted.iloc[-1]["signal"] == fv.INSUFFICIENT
    assert counted.iloc[-1]["n_comparables"] < 8       # too far in state space


def test_matching_basis_still_counts_as_comparable():
    """The dimension must only exclude days that genuinely differ."""
    history = [97.0, 103.0, 99.0, 101.0, 96.0, 104.0, 98.0, 102.0]
    rows = [make_pair_row(d, 4, 11, date(2024, 1, 1) + timedelta(days=i))
            for i, d in enumerate(history)]
    today = date(2024, 1, 20)
    rows.append(make_pair_row(70.0, 4, 11, today))
    spreads = pd.DataFrame(rows)
    spreads["basis"] = 80.0                            # same basis throughout

    out = fv.evaluate(spreads, min_comparables=8, use_basis=True)
    assert out.iloc[-1]["n_comparables"] == 8
    assert out.iloc[-1]["signal"] == fv.BUY


def test_basis_column_absent_is_not_an_error():
    """Older cached spread files predate the column; the model should fall back
    rather than crash on them."""
    history = [97.0, 103.0, 99.0, 101.0, 96.0, 104.0, 98.0, 102.0]
    rows = [make_pair_row(d, 4, 11, date(2024, 1, 1) + timedelta(days=i))
            for i, d in enumerate(history)]
    rows.append(make_pair_row(70.0, 4, 11, date(2024, 1, 20)))
    spreads = pd.DataFrame(rows)
    assert "basis" not in spreads.columns

    out = fv.evaluate(spreads, min_comparables=8, use_basis=True)
    assert out.iloc[-1]["n_comparables"] == 8
