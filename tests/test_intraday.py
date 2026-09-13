"""Tests for the intraday variant.

The timing rule is the thing to pin down. The signal is computed on one day's
settlement prices and acted on in the *next* session — if that ever slips to
acting on the same day's data, the variant is reading prices that were not known
when the decision was made, and its edge is manufactured.
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from src.backtest import intraday
from src.signal import fair_value as fv


def make_chain(rows):
    """rows: (date, expiry, strike, option_type, open_px, close_px)"""
    return pd.DataFrame(
        [{"date": d, "expiry": e, "strike": k, "option_type": o,
          "open_px": op, "close_px": cl, "px": cl, "tradeable": True}
         for d, e, k, o, op, cl in rows])


def make_signal(signal_date, signal=fv.BUY, z=-2.0, strike=20000.0,
                front=date(2024, 1, 8), back=date(2024, 1, 25)):
    return pd.DataFrame([{
        "date": signal_date, "option_type": "CE", "signal": signal,
        "z_score": z, "strike": strike, "front_expiry": front,
        "back_expiry": back, "spot": 20000.0, "lot_size": 50.0,
        "front_tradeable": True, "back_tradeable": True,
    }])


D1, D2 = date(2024, 1, 4), date(2024, 1, 5)
FRONT, BACK = date(2024, 1, 8), date(2024, 1, 25)


def _chain_for_session(f_open, b_open, f_close, b_close):
    return make_chain([
        (D2, FRONT, 20000.0, "CE", f_open, f_close),
        (D2, BACK, 20000.0, "CE", b_open, b_close),
    ])


def test_signal_is_acted_on_in_the_following_session():
    """A signal on day one trades on day two, never on day one itself."""
    signals = pd.concat([make_signal(D1), make_signal(D2, signal=fv.FLAT)],
                        ignore_index=True)
    chain = _chain_for_session(100.0, 150.0, 95.0, 150.0)

    result = intraday.run(signals, chain, verbose=False)
    assert len(result["trades"]) == 1
    assert result["trades"].iloc[0]["signal_date"] == D1
    assert result["trades"].iloc[0]["session"] == D2


def test_a_signal_on_the_last_day_cannot_be_traded():
    """There is no next session to act in, so it is dropped rather than filled
    at some price from the same day."""
    signals = make_signal(D1)
    chain = _chain_for_session(100.0, 150.0, 95.0, 150.0)
    result = intraday.run(signals, chain, verbose=False)
    assert result["trades"].empty


def test_long_calendar_profits_when_the_debit_widens_intraday():
    signals = pd.concat([make_signal(D1), make_signal(D2, signal=fv.FLAT)],
                        ignore_index=True)
    # Debit opens at 50, closes at 60.
    chain = _chain_for_session(100.0, 150.0, 90.0, 150.0)
    trade = intraday.run(signals, chain, verbose=False)["trades"].iloc[0]

    assert trade["entry_debit"] == pytest.approx(50.0)
    assert trade["exit_debit"] == pytest.approx(60.0)
    assert trade["gross_pnl"] == pytest.approx(10.0 * trade["quantity"])
    assert trade["net_pnl"] < trade["gross_pnl"]        # costs were charged


def test_short_calendar_is_the_mirror():
    long_sig = pd.concat([make_signal(D1, signal=fv.BUY, z=-2.0),
                          make_signal(D2, signal=fv.FLAT)], ignore_index=True)
    short_sig = pd.concat([make_signal(D1, signal=fv.SELL, z=2.0),
                           make_signal(D2, signal=fv.FLAT)], ignore_index=True)
    chain = _chain_for_session(100.0, 150.0, 90.0, 150.0)

    a = intraday.run(long_sig, chain, allow_short=True, verbose=False)["trades"]
    b = intraday.run(short_sig, chain, allow_short=True, verbose=False)["trades"]
    assert a.iloc[0]["gross_pnl"] == pytest.approx(-b.iloc[0]["gross_pnl"])


def test_position_never_survives_the_session():
    signals = pd.concat([make_signal(D1), make_signal(D2, signal=fv.FLAT)],
                        ignore_index=True)
    chain = _chain_for_session(100.0, 150.0, 90.0, 150.0)
    result = intraday.run(signals, chain, verbose=False)
    assert (result["trades"]["days_held"] == 0).all()
    assert result["trades"].iloc[0]["exit_reason"] == "session_close"


def test_untraded_legs_are_not_filled():
    """A contract with no volume could not have been entered at any price."""
    signals = pd.concat([make_signal(D1), make_signal(D2, signal=fv.FLAT)],
                        ignore_index=True)
    chain = _chain_for_session(100.0, 150.0, 90.0, 150.0)
    chain.loc[chain["expiry"] == BACK, "tradeable"] = False
    assert intraday.run(signals, chain, verbose=False)["trades"].empty


def test_missing_open_price_blocks_the_trade():
    signals = pd.concat([make_signal(D1), make_signal(D2, signal=fv.FLAT)],
                        ignore_index=True)
    chain = _chain_for_session(100.0, 150.0, 90.0, 150.0)
    chain.loc[chain["expiry"] == FRONT, "open_px"] = None
    assert intraday.run(signals, chain, verbose=False)["trades"].empty


def test_short_signals_are_skipped_when_shorting_is_off():
    signals = pd.concat([make_signal(D1, signal=fv.SELL, z=2.0),
                         make_signal(D2, signal=fv.FLAT)], ignore_index=True)
    chain = _chain_for_session(100.0, 150.0, 90.0, 150.0)
    assert intraday.run(signals, chain, allow_short=False,
                        verbose=False)["trades"].empty
    assert not intraday.run(signals, chain, allow_short=True,
                            verbose=False)["trades"].empty
