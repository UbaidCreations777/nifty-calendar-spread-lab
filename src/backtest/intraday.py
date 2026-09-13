"""The same signal, traded and closed inside one session.

The overnight version holds a calendar for around four days and carries the risk
that hurts it most: a calendar is short gamma, so a gap in NIFTY moves against
the leg it sold before the desk can do anything about it. Closing before the bell
removes every gap from the sample.

What it cannot remove is the cost. The round trip is the same four orders whether
the position is held four days or four hours, and one session's move is smaller
than four sessions' - so this variant pays the same toll to capture less
distance. Whether that trade-off is worth making is the question, and it is
measurable.

**The timing, and why it is not look-ahead.** The signal for a given session is
computed from the *previous* day's settlement prices, on the legs chosen then.
The decision is therefore complete before the session opens. Entry is at that
day's open, exit at its close, on those same contracts.

**The limitation that matters.** Bhavcopy gives one open and one close per
contract, not a time-stamped book. Each leg's "open" is its own first trade of
the day, and two contracts do not print their first trade at the same instant -
so a spread entered at "the open" here is really two fills at two unknown
moments. On a structure whose edge is a few tenths of a percent that legging gap
is not a rounding error, and it is the strongest argument for redoing this on
intraday data before believing any of it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config as C
from ..signal import fair_value as fv
from .costs import round_trip_cost


def _lookup(chain: pd.DataFrame, column: str) -> dict:
    sub = chain[chain[column].notna()]
    keys = zip(sub["date"], sub["expiry"], sub["strike"], sub["option_type"])
    return dict(zip(keys, sub[column]))


def run(evaluated: pd.DataFrame, chain: pd.DataFrame,
        initial_capital: float = C.INITIAL_CAPITAL,
        allow_short: bool = C.ALLOW_SHORT,
        verbose: bool = True) -> dict:
    """One decision per session: act on yesterday's signal, flat by the close."""
    opens = _lookup(chain, "open_px")
    closes = _lookup(chain, "close_px")
    traded = {(d, e, s, o) for d, e, s, o in
              zip(chain.loc[chain["tradeable"], "date"],
                  chain.loc[chain["tradeable"], "expiry"],
                  chain.loc[chain["tradeable"], "strike"],
                  chain.loc[chain["tradeable"], "option_type"])}

    dates = sorted(evaluated["date"].unique())
    next_session = {d: dates[i + 1] for i, d in enumerate(dates[:-1])}

    capital = float(initial_capital)
    trades: list[dict] = []
    equity: list[dict] = []

    for d in dates:
        signals_today = evaluated[evaluated["date"] == d]
        candidates = signals_today[signals_today["signal"].isin([fv.BUY, fv.SELL])]
        if not allow_short:
            candidates = candidates[candidates["signal"] == fv.BUY]

        session = next_session.get(d)
        trade = None

        if session is not None and not candidates.empty:
            pick = candidates.iloc[candidates["z_score"].abs().argmax()]
            legs = [(pick["front_expiry"], pick["strike"], pick["option_type"]),
                    (pick["back_expiry"], pick["strike"], pick["option_type"])]
            keys = [(session, *leg) for leg in legs]

            # Both legs must have opened and closed, and both must have traded -
            # a contract with no volume could not have been entered at any price.
            if (all(k in opens and k in closes for k in keys)
                    and all(k in traded for k in keys)):
                f_open, b_open = opens[keys[0]], opens[keys[1]]
                f_close, b_close = closes[keys[0]], closes[keys[1]]
                trade = _execute(pick, session, f_open, b_open, f_close, b_close,
                                 capital)

        if trade is not None:
            capital += trade["net_pnl"]
            trades.append(trade)

        equity.append({"date": d, "capital": capital, "unrealised": 0.0,
                       "equity": capital,
                       "in_position": trade is not None})

    if verbose:
        print(f"  intraday: {len(trades)} sessions traded over {len(dates)} days",
              flush=True)

    return {"trades": pd.DataFrame(trades), "equity": pd.DataFrame(equity),
            "settings": {"initial_capital": initial_capital,
                         "allow_short": allow_short, "mode": "intraday"}}


def _execute(pick: pd.Series, session, f_open: float, b_open: float,
             f_close: float, b_close: float, capital: float) -> dict | None:
    lot = float(pick["lot_size"]) if np.isfinite(pick["lot_size"]) else C.LOT_SIZE_DEFAULT
    is_long = pick["signal"] == fv.BUY
    spot = float(pick["spot"])

    entry_debit = b_open - f_open
    exit_debit = b_close - f_close

    margin_per_lot = C.MARGIN_PER_LOT_PCT * spot * lot
    debit_per_lot = abs(entry_debit) * lot
    cost_per_lot = margin_per_lot + (debit_per_lot if is_long else 0.0)
    lots_by_margin = int((capital * C.MAX_MARGIN_UTILISATION) // cost_per_lot) \
        if cost_per_lot > 0 else 0

    risk_per_lot = C.STRESS_MOVE_PCT_OF_SPOT / 100.0 * spot * lot
    lots_by_risk = int((capital * C.RISK_PER_TRADE_PCT) // risk_per_lot) \
        if risk_per_lot > 0 else 0

    n_lots = max(0, min(lots_by_margin, lots_by_risk))
    if n_lots < 1:
        return None

    quantity = int(n_lots * lot)
    move = exit_debit - entry_debit
    gross = move * quantity * (1.0 if is_long else -1.0)
    costs = round_trip_cost(f_open, b_open, f_close, b_close, quantity, is_long)

    return {
        "signal_date": pick["date"],
        "session": session,
        "option_type": pick["option_type"],
        "strike": pick["strike"],
        "is_long": is_long,
        "entry_z": float(pick["z_score"]),
        "front_expiry": pick["front_expiry"],
        "back_expiry": pick["back_expiry"],
        "entry_debit": entry_debit,
        "exit_debit": exit_debit,
        "debit_move": move,
        "spot": spot,
        "lot_size": lot,
        "n_lots": n_lots,
        "quantity": quantity,
        "gross_pnl": gross,
        "cost": costs["total"],
        "net_pnl": gross - costs["total"],
        "days_held": 0,
        "exit_reason": "session_close",
    }
