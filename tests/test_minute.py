"""Tests for the minute-bar data layer and minute-level execution.

The staleness guard is the one that earns its keep. Minute data invites the
assumption that a price exists at every minute, and for an option chain that is
false - most contracts print in bursts. Filling a trade off a bar from twenty
minutes ago looks like precision and is fiction.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from src.backtest import intraday_minute as im
from src.data.minute import parse_ticker
from src.signal import fair_value as fv


# ------------------------------------------------------------------- tickers

@pytest.mark.parametrize("ticker,expected", [
    ("NIFTY04JAN24CE18300", (dt.date(2024, 1, 4), "CE", 18300.0)),
    ("NIFTY09APR25PE20400", (dt.date(2025, 4, 9), "PE", 20400.0)),
    # A minority of rows use the single-letter spelling.
    ("NIFTY10MAR26P24000", (dt.date(2026, 3, 10), "PE", 24000.0)),
    ("NIFTY10MAR26C24000", (dt.date(2026, 3, 10), "CE", 24000.0)),
])
def test_parse_ticker(ticker, expected):
    assert parse_ticker(ticker) == expected


@pytest.mark.parametrize("ticker", ["Nifty 50", "BANKNIFTY04JAN24CE18300", ""])
def test_parse_ticker_rejects_non_options(ticker):
    assert parse_ticker(ticker) == (None, None, None)


# ------------------------------------------------------------------- pricing

SESSION = dt.date(2024, 1, 5)


def bars(times_and_prices):
    return pd.DataFrame([
        {"ts": pd.Timestamp.combine(SESSION, t), "time": t, "close": p}
        for t, p in times_and_prices])


def test_price_at_uses_the_last_trade_at_or_before_the_target():
    b = bars([(dt.time(9, 18), 100.0), (dt.time(9, 20), 101.0),
              (dt.time(9, 25), 105.0)])
    assert im._price_at(b, dt.time(9, 20), im.MAX_STALENESS) == 101.0
    assert im._price_at(b, dt.time(9, 22), im.MAX_STALENESS) == 101.0


def test_price_at_rejects_a_stale_bar():
    """The contract last printed twenty minutes ago - there is no price here."""
    b = bars([(dt.time(9, 0), 100.0)])
    assert im._price_at(b, dt.time(9, 20), im.MAX_STALENESS) is None


def test_price_at_returns_none_before_the_first_trade():
    b = bars([(dt.time(10, 0), 100.0)])
    assert im._price_at(b, dt.time(9, 20), im.MAX_STALENESS) is None


# ----------------------------------------------------------------- execution

FRONT, BACK = dt.date(2024, 1, 11), dt.date(2024, 1, 25)
D1 = dt.date(2024, 1, 4)


def make_minute(front_prices, back_prices, session=SESSION):
    rows = []
    for expiry, series in ((FRONT, front_prices), (BACK, back_prices)):
        for t, p in series:
            rows.append({"date": session, "ts": pd.Timestamp.combine(session, t),
                         "time": t, "expiry": expiry, "strike": 20000.0,
                         "option_type": "CE", "close": p})
    return pd.DataFrame(rows)


def make_signals(signal=fv.BUY, z=-2.0):
    return pd.DataFrame([
        {"date": D1, "option_type": "CE", "signal": signal, "z_score": z,
         "strike": 20000.0, "front_expiry": FRONT, "back_expiry": BACK,
         "spot": 20000.0, "lot_size": 50.0},
        {"date": SESSION, "option_type": "CE", "signal": fv.FLAT, "z_score": 0.0,
         "strike": 20000.0, "front_expiry": FRONT, "back_expiry": BACK,
         "spot": 20000.0, "lot_size": 50.0},
    ])


TIMES = [(dt.time(9, 20), None), (dt.time(15, 15), None)]


def test_both_legs_are_priced_at_the_same_instant():
    minute = make_minute(
        front_prices=[(dt.time(9, 20), 100.0), (dt.time(15, 15), 90.0)],
        back_prices=[(dt.time(9, 20), 150.0), (dt.time(15, 15), 150.0)])
    trade = im.run(make_signals(), minute, verbose=False)["trades"].iloc[0]

    assert trade["entry_debit"] == pytest.approx(50.0)
    assert trade["exit_debit"] == pytest.approx(60.0)
    assert trade["gross_pnl"] == pytest.approx(10.0 * trade["quantity"])


def test_a_stale_leg_cancels_the_whole_trade():
    """One leg filled is a naked option, not a calendar."""
    minute = make_minute(
        front_prices=[(dt.time(9, 20), 100.0), (dt.time(15, 15), 90.0)],
        back_prices=[(dt.time(8, 30), 150.0), (dt.time(15, 15), 150.0)])
    result = im.run(make_signals(), minute, verbose=False)
    assert result["trades"].empty
    assert result["skipped"]["stale"] == 1


def test_a_missing_leg_cancels_the_whole_trade():
    minute = make_minute(
        front_prices=[(dt.time(9, 20), 100.0), (dt.time(15, 15), 90.0)],
        back_prices=[])
    result = im.run(make_signals(), minute, verbose=False)
    assert result["trades"].empty
    assert result["skipped"]["no_bars"] == 1


def test_the_trade_lands_in_the_session_after_the_signal():
    minute = make_minute(
        front_prices=[(dt.time(9, 20), 100.0), (dt.time(15, 15), 90.0)],
        back_prices=[(dt.time(9, 20), 150.0), (dt.time(15, 15), 150.0)])
    trade = im.run(make_signals(), minute, verbose=False)["trades"].iloc[0]
    assert trade["signal_date"] == D1
    assert trade["session"] == SESSION
    assert trade["days_held"] == 0


def test_short_signals_need_shorting_enabled():
    minute = make_minute(
        front_prices=[(dt.time(9, 20), 100.0), (dt.time(15, 15), 90.0)],
        back_prices=[(dt.time(9, 20), 150.0), (dt.time(15, 15), 150.0)])
    sig = make_signals(signal=fv.SELL, z=2.0)
    assert im.run(sig, minute, allow_short=False, verbose=False)["trades"].empty
    assert not im.run(sig, minute, allow_short=True,
                      verbose=False)["trades"].empty


def test_costs_are_charged_and_reduce_the_result():
    minute = make_minute(
        front_prices=[(dt.time(9, 20), 100.0), (dt.time(15, 15), 90.0)],
        back_prices=[(dt.time(9, 20), 150.0), (dt.time(15, 15), 150.0)])
    trade = im.run(make_signals(), minute, verbose=False)["trades"].iloc[0]
    assert trade["cost"] > 0
    assert trade["net_pnl"] == pytest.approx(trade["gross_pnl"] - trade["cost"])
