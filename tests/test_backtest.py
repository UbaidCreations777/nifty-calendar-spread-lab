"""Tests for the backtest engine and the cost model.

The sign convention is the thing worth pinning down. A long calendar is short the
front and long the back, so it makes money when the front decays faster than the
back - and it is easy to write that backwards and get a strategy that looks
profitable because the P&L is inverted rather than because the signal works.
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from src.backtest import costs, engine, metrics


def test_long_calendar_profits_when_front_decays_faster():
    pos = {"entry_front_px": 100.0, "entry_back_px": 150.0,
           "quantity": 50, "is_long": True}
    # Front collapses by 40, back only gives up 10: the structure widened.
    pnl = engine._gross_pnl(pos, front_px=60.0, back_px=140.0)
    assert pnl == pytest.approx((40.0 - 10.0) * 50)
    assert pnl > 0


def test_long_calendar_loses_when_the_back_leg_collapses():
    pos = {"entry_front_px": 100.0, "entry_back_px": 150.0,
           "quantity": 50, "is_long": True}
    pnl = engine._gross_pnl(pos, front_px=95.0, back_px=110.0)
    assert pnl == pytest.approx((5.0 - 40.0) * 50)
    assert pnl < 0


def test_short_calendar_is_the_exact_mirror():
    long_pos = {"entry_front_px": 100.0, "entry_back_px": 150.0,
                "quantity": 50, "is_long": True}
    short_pos = {**long_pos, "is_long": False}
    assert (engine._gross_pnl(long_pos, 60.0, 140.0)
            == -engine._gross_pnl(short_pos, 60.0, 140.0))


def test_costs_are_charged_on_all_four_orders():
    c = costs.round_trip_cost(front_px_in=100.0, back_px_in=150.0,
                              front_px_out=60.0, back_px_out=140.0,
                              quantity=50, is_long_calendar=True)
    assert c["brokerage"] == pytest.approx(4 * 20.0)
    assert c["total"] > 0
    for leg in ("front_entry", "back_entry", "front_exit", "back_exit"):
        assert c[leg] > 0


def test_stt_falls_only_on_the_sell_side():
    """Buying a leg pays stamp duty, selling it pays STT. Charging both on both
    sides would roughly double the tax line."""
    buy = costs.leg_cost(premium=100.0, quantity=50, is_buy=True)
    sell = costs.leg_cost(premium=100.0, quantity=50, is_buy=False)
    assert buy.stt == 0.0
    assert sell.stt == pytest.approx(100.0 * 50 * 0.001)
    assert sell.stamp == 0.0
    assert buy.stamp > 0


def test_costs_scale_with_size():
    small = costs.round_trip_cost(100.0, 150.0, 60.0, 140.0, 50, True)
    large = costs.round_trip_cost(100.0, 150.0, 60.0, 140.0, 500, True)
    assert large["total"] > small["total"]
    # Brokerage is flat per order, so cost per unit falls as size rises.
    assert large["total"] / 500 < small["total"] / 50


def test_net_pnl_is_gross_less_costs():
    pos = {"entry_date": date(2024, 1, 1), "entry_front_px": 100.0,
           "entry_back_px": 150.0, "quantity": 50, "is_long": True}
    trade = engine._close(pos, date(2024, 1, 5), 60.0, 140.0, "signal_reverted")
    assert trade["net_pnl"] == pytest.approx(trade["gross_pnl"] - trade["cost"])
    assert trade["cost"] > 0
    assert trade["exit_reason"] == "signal_reverted"
    assert trade["days_held"] == 4


def test_metrics_handle_a_flat_equity_curve():
    equity = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=50),
                           "equity": [1_000_000.0] * 50,
                           "in_position": [False] * 50})
    stats = metrics.summarise(equity, pd.DataFrame(), 1_000_000.0)
    assert stats["total_return_pct"] == pytest.approx(0.0)
    assert stats["max_drawdown_pct"] == pytest.approx(0.0)
    assert stats["n_trades"] == 0


def test_max_drawdown_is_measured_from_the_peak():
    curve = [100.0, 120.0, 90.0, 110.0]        # peak 120, trough 90 => -25%
    equity = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=4),
                           "equity": curve, "in_position": [True] * 4})
    stats = metrics.summarise(equity, pd.DataFrame(), 100.0)
    assert stats["max_drawdown_pct"] == pytest.approx(-25.0)


def test_lot_size_prefers_the_data_over_the_schedule():
    """The UDiFF files carry the real lot size; the hardcoded schedule is only a
    fallback for older files and must not override it."""
    lots = pd.Series({date(2026, 9, 10): 65.0})
    assert engine._lot_size_for(date(2026, 9, 10), lots) == 65.0
    # A date with no data falls back to the schedule.
    assert engine._lot_size_for(date(2024, 1, 10), pd.Series(dtype=float)) == 50.0
