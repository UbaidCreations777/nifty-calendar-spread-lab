"""Intraday execution on minute bars, with both legs priced at the same instant.

This replaces the guess the daily version had to make. Bhavcopy gives one open
per contract, and each leg's open is its own first trade at an unknown moment -
so the "spread at the open" it computed was two fills that never coexisted. Here
both legs are read off the same timestamp, which is the price a desk could
actually have worked.

The decision still comes from the previous session's settlement prices, so
nothing known only after the open influences whether the trade is taken. What the
minute data changes is not the signal but the fill.

Two guards matter:

**Staleness.** A contract that has not printed for several minutes has no current
price, only an old one. Carrying a stale bar forward and calling it a fill is the
same error as marking a position off an untraded strike, so a leg whose last bar
is older than `max_staleness` disqualifies the trade rather than being filled at
whatever it last traded at.

**Both legs or neither.** A calendar with one leg filled is a naked option, not a
spread. If either leg cannot be priced at the entry instant, the trade does not
happen.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from .. import config as C
from ..signal import fair_value as fv
from .costs import round_trip_cost

ENTRY_TIME = dt.time(9, 20)
EXIT_TIME = dt.time(15, 15)
MAX_STALENESS = pd.Timedelta("5min")


def _bar_index(minute: pd.DataFrame) -> dict:
    """(date, expiry, strike, option_type) -> that contract's bars for the day."""
    out = {}
    for key, grp in minute.groupby(["date", "expiry", "strike", "option_type"],
                                   sort=False):
        out[key] = grp.sort_values("ts")
    return out


def _price_at(bars: pd.DataFrame, when: dt.time,
              max_staleness: pd.Timedelta) -> float | None:
    """Last trade at or before `when`, provided it is recent enough to be real."""
    if bars is None or bars.empty:
        return None
    cutoff = bars[bars["time"] <= when]
    if cutoff.empty:
        return None
    last = cutoff.iloc[-1]
    target = pd.Timestamp.combine(pd.Timestamp(last["ts"]).date(), when)
    if target - pd.Timestamp(last["ts"]) > max_staleness:
        return None
    return float(last["close"])


def run(evaluated: pd.DataFrame, minute: pd.DataFrame,
        initial_capital: float = C.INITIAL_CAPITAL,
        entry_time: dt.time = ENTRY_TIME,
        exit_time: dt.time = EXIT_TIME,
        max_staleness: pd.Timedelta = MAX_STALENESS,
        allow_short: bool = C.ALLOW_SHORT,
        verbose: bool = True) -> dict:
    """One trade per session at most, entered and closed inside it."""
    bars = _bar_index(minute)
    sessions = sorted(minute["date"].unique())
    session_after = {}
    signal_days = sorted(evaluated["date"].unique())
    for i, d in enumerate(signal_days[:-1]):
        nxt = signal_days[i + 1]
        if nxt in set(sessions):
            session_after[d] = nxt

    capital = float(initial_capital)
    trades: list[dict] = []
    equity: list[dict] = []
    skipped = {"no_bars": 0, "stale": 0, "too_small": 0}

    for d in signal_days:
        candidates = evaluated[(evaluated["date"] == d)
                               & evaluated["signal"].isin([fv.BUY, fv.SELL])]
        if not allow_short:
            candidates = candidates[candidates["signal"] == fv.BUY]

        session = session_after.get(d)
        trade = None

        if session is not None and not candidates.empty:
            pick = candidates.iloc[candidates["z_score"].abs().argmax()]
            front_key = (session, pick["front_expiry"], float(pick["strike"]),
                         pick["option_type"])
            back_key = (session, pick["back_expiry"], float(pick["strike"]),
                        pick["option_type"])

            f_bars, b_bars = bars.get(front_key), bars.get(back_key)
            if f_bars is None or b_bars is None:
                skipped["no_bars"] += 1
            else:
                f_in = _price_at(f_bars, entry_time, max_staleness)
                b_in = _price_at(b_bars, entry_time, max_staleness)
                f_out = _price_at(f_bars, exit_time, max_staleness)
                b_out = _price_at(b_bars, exit_time, max_staleness)

                if None in (f_in, b_in, f_out, b_out):
                    skipped["stale"] += 1
                else:
                    trade = _execute(pick, session, entry_time, exit_time,
                                     f_in, b_in, f_out, b_out, capital)
                    if trade is None:
                        skipped["too_small"] += 1

        if trade is not None:
            capital += trade["net_pnl"]
            trades.append(trade)

        equity.append({"date": d, "capital": capital, "unrealised": 0.0,
                       "equity": capital, "in_position": trade is not None})

    if verbose:
        print(f"  minute intraday: {len(trades)} trades, skipped "
              f"{skipped['no_bars']} no-bars / {skipped['stale']} stale / "
              f"{skipped['too_small']} undersized", flush=True)

    return {"trades": pd.DataFrame(trades), "equity": pd.DataFrame(equity),
            "skipped": skipped,
            "settings": {"initial_capital": initial_capital,
                         "entry_time": str(entry_time),
                         "exit_time": str(exit_time),
                         "allow_short": allow_short, "mode": "intraday_minute"}}


def _execute(pick: pd.Series, session, entry_time, exit_time,
             f_in: float, b_in: float, f_out: float, b_out: float,
             capital: float) -> dict | None:
    lot = float(pick["lot_size"]) if np.isfinite(pick["lot_size"]) else C.LOT_SIZE_DEFAULT
    is_long = pick["signal"] == fv.BUY
    spot = float(pick["spot"])

    entry_debit = b_in - f_in
    exit_debit = b_out - f_out

    margin_per_lot = C.MARGIN_PER_LOT_PCT * spot * lot
    cost_per_lot = margin_per_lot + (abs(entry_debit) * lot if is_long else 0.0)
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
    costs = round_trip_cost(f_in, b_in, f_out, b_out, quantity, is_long)

    return {
        "signal_date": pick["date"], "session": session,
        "entry_time": str(entry_time), "exit_time": str(exit_time),
        "option_type": pick["option_type"], "strike": pick["strike"],
        "is_long": is_long, "entry_z": float(pick["z_score"]),
        "front_expiry": pick["front_expiry"], "back_expiry": pick["back_expiry"],
        "entry_front_px": f_in, "entry_back_px": b_in,
        "exit_front_px": f_out, "exit_back_px": b_out,
        "entry_debit": entry_debit, "exit_debit": exit_debit,
        "debit_move": move, "spot": spot, "lot_size": lot, "n_lots": n_lots,
        "quantity": quantity, "gross_pnl": gross, "cost": costs["total"],
        "net_pnl": gross - costs["total"], "days_held": 0,
        "exit_reason": "session_close",
    }


def sweep_times(evaluated: pd.DataFrame, minute: pd.DataFrame,
                entry_times, exit_times, **kwargs) -> pd.DataFrame:
    """The same rule across entry and exit clocks.

    A result that only works at one pair of times is a result about those times.
    """
    from . import metrics

    rows = []
    for t_in in entry_times:
        for t_out in exit_times:
            if t_out <= t_in:
                continue
            r = run(evaluated, minute, entry_time=t_in, exit_time=t_out,
                    verbose=False, **kwargs)
            s = metrics.summarise(r["equity"], r["trades"], C.INITIAL_CAPITAL)
            rows.append({"entry": str(t_in), "exit": str(t_out),
                         "trades": s.get("n_trades", 0),
                         "return_pct": s.get("total_return_pct"),
                         "sharpe": s.get("sharpe"),
                         "win_rate_pct": s.get("win_rate_pct"),
                         "gross_pnl": s.get("gross_pnl"),
                         "net_pnl": s.get("net_pnl"),
                         "max_drawdown_pct": s.get("max_drawdown_pct")})
    return pd.DataFrame(rows)
