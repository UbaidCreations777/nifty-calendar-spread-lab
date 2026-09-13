"""Event-driven backtest of the fair-value calendar signal.

The loop walks forward one trading day at a time. A position, once opened, keeps
the strike and the two expiries it was opened with - it does not roll with the
daily ATM series - and is marked each day off those exact contracts. That
distinction is the difference between backtesting a trade and backtesting an
index of trades.

Exits come from two places. The signal exit fires when the freshly computed ATM
spread for the same state has reverted inside `Z_EXIT`: the mispricing the trade
was opened against is no longer there, whatever the position itself is worth. The
time exit fires a day before the front leg expires, which keeps the book clear of
expiry-day pin risk and of the higher STT charged on exercised contracts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config as C
from ..signal import fair_value as fv
from .costs import round_trip_cost


def _price_lookup(chain: pd.DataFrame) -> dict:
    keys = zip(chain["date"], chain["expiry"], chain["strike"],
               chain["option_type"])
    return dict(zip(keys, chain["px"]))


def _lot_size_for(date, chain_lots: pd.Series) -> float:
    lot = chain_lots.get(date, np.nan)
    if np.isfinite(lot) and lot > 0:
        return float(lot)
    for cutoff, size in sorted(C.LOT_SIZE_SCHEDULE.items(), reverse=True):
        if date >= cutoff:
            return float(size)
    return float(C.LOT_SIZE_DEFAULT)


def run(evaluated: pd.DataFrame, chain: pd.DataFrame,
        initial_capital: float = C.INITIAL_CAPITAL,
        z_exit: float = C.Z_EXIT,
        allow_short: bool = C.ALLOW_SHORT,
        max_holding_days: int | None = C.MAX_HOLDING_DAYS,
        verbose: bool = True) -> dict:
    """Run the backtest. Returns trades, the daily equity curve and settings."""
    prices = _price_lookup(chain)
    lots_by_date = (chain.groupby("date")["lot_size"].first()
                    .astype(float))

    capital = float(initial_capital)
    open_pos: dict | None = None
    trades: list[dict] = []
    equity: list[dict] = []

    dates = sorted(evaluated["date"].unique())
    by_date = {d: g for d, g in evaluated.groupby("date")}

    for d in dates:
        today = by_date[d]

        # ---------------------------------------------------------- mark / exit
        if open_pos is not None:
            f_key = (d, open_pos["front_expiry"], open_pos["strike"],
                     open_pos["option_type"])
            b_key = (d, open_pos["back_expiry"], open_pos["strike"],
                     open_pos["option_type"])
            f_px, b_px = prices.get(f_key), prices.get(b_key)

            marked = f_px is not None and b_px is not None
            if marked:
                open_pos["last_front_px"] = float(f_px)
                open_pos["last_back_px"] = float(b_px)

            front_dte = (pd.Timestamp(open_pos["front_expiry"])
                         - pd.Timestamp(d)).days

            same_state = today[today["option_type"] == open_pos["option_type"]]
            z_now = (float(same_state["z_score"].iloc[0])
                     if not same_state.empty and np.isfinite(
                         same_state["z_score"].iloc[0]) else np.nan)

            held = (pd.Timestamp(d) - pd.Timestamp(open_pos["entry_date"])).days

            reason = None
            if front_dte <= 1:
                reason = "front_expiry"
            elif max_holding_days is not None and held >= max_holding_days:
                # The edge study measures the signal's information decaying to
                # nothing by the fifth day, so holding past it is carrying the
                # structure's risk without the reason the trade was opened for.
                reason = "max_holding"
            elif not marked:
                # The legs have dropped out of the chain - the strike has drifted
                # out of the window, or the day is missing. A position that
                # cannot be marked cannot be managed, so it is closed at its last
                # known marks rather than carried on stale prices. Without this
                # the book silently holds one trade to the end of the sample.
                reason = "lost_marks"
            elif np.isfinite(z_now) and abs(z_now) <= z_exit:
                reason = "signal_reverted"
            elif np.isfinite(z_now) and np.sign(z_now) != np.sign(open_pos["entry_z"]):
                reason = "signal_flipped"

            if reason is not None:
                trade = _close(open_pos, d, open_pos["last_front_px"],
                               open_pos["last_back_px"], reason)
                capital += trade["net_pnl"]
                trades.append(trade)
                open_pos = None

        # --------------------------------------------------------------- entry
        if open_pos is None:
            candidates = today[today["signal"].isin([fv.BUY, fv.SELL])]
            candidates = candidates[candidates["front_tradeable"]
                                    & candidates["back_tradeable"]]
            if not allow_short:
                candidates = candidates[candidates["signal"] == fv.BUY]

            if not candidates.empty:
                # The most stretched reading is the one worth the capital.
                pick = candidates.iloc[candidates["z_score"].abs().argmax()]
                lot = _lot_size_for(d, lots_by_date)
                is_long = pick["signal"] == fv.BUY

                margin_per_lot = C.MARGIN_PER_LOT_PCT * pick["spot"] * lot
                debit_per_lot = abs(pick["debit"]) * lot
                cost_per_lot = margin_per_lot + (debit_per_lot if is_long else 0.0)
                budget = capital * C.MAX_MARGIN_UTILISATION
                lots_by_margin = int(budget // cost_per_lot) if cost_per_lot > 0 else 0

                # What the deposit allows and what the risk budget allows are
                # different numbers, and the smaller one is the position.
                risk_per_lot = (C.STRESS_MOVE_PCT_OF_SPOT / 100.0
                                * pick["spot"] * lot)
                lots_by_risk = int((capital * C.RISK_PER_TRADE_PCT) // risk_per_lot) \
                    if risk_per_lot > 0 else 0
                n_lots = max(0, min(lots_by_margin, lots_by_risk))

                if n_lots >= 1:
                    open_pos = {
                        "entry_date": d,
                        "option_type": pick["option_type"],
                        "strike": pick["strike"],
                        "front_expiry": pick["front_expiry"],
                        "back_expiry": pick["back_expiry"],
                        "entry_front_px": float(pick["front_px"]),
                        "entry_back_px": float(pick["back_px"]),
                        "entry_debit": float(pick["debit"]),
                        "entry_debit_pct": float(pick["debit_pct"]),
                        "entry_z": float(pick["z_score"]),
                        "entry_percentile": float(pick["percentile"]),
                        "entry_fv_mean_pct": float(pick["fv_mean_pct"]),
                        "n_comparables": int(pick["n_comparables"]),
                        "entry_front_dte": int(pick["front_dte"]),
                        "entry_back_dte": int(pick["back_dte"]),
                        "entry_front_iv": float(pick["front_iv"]),
                        "entry_back_iv": float(pick["back_iv"]),
                        "entry_spot": float(pick["spot"]),
                        "is_long": bool(is_long),
                        "lot_size": lot,
                        "n_lots": n_lots,
                        "quantity": int(n_lots * lot),
                        "margin": margin_per_lot * n_lots,
                        "days_held": 0,
                        "last_front_px": float(pick["front_px"]),
                        "last_back_px": float(pick["back_px"]),
                    }

        # ------------------------------------------------------- equity marking
        unrealised = 0.0
        if open_pos is not None:
            unrealised = _gross_pnl(open_pos, open_pos["last_front_px"],
                                    open_pos["last_back_px"])
        equity.append({"date": d, "capital": capital,
                       "unrealised": unrealised,
                       "equity": capital + unrealised,
                       "in_position": open_pos is not None})

    # A position still open when the data runs out is a real trade with a real
    # mark, and leaving it out would drop its P&L from every statistic.
    if open_pos is not None:
        trade = _close(open_pos, dates[-1], open_pos["last_front_px"],
                       open_pos["last_back_px"], "end_of_sample")
        capital += trade["net_pnl"]
        trades.append(trade)
        equity[-1] = {**equity[-1], "capital": capital, "unrealised": 0.0,
                      "equity": capital, "in_position": False}

    if verbose:
        print(f"  backtest: {len(trades)} trades over {len(dates)} days", flush=True)

    return {
        "trades": pd.DataFrame(trades),
        "equity": pd.DataFrame(equity),
        "settings": {"initial_capital": initial_capital, "z_entry": C.Z_ENTRY,
                     "z_exit": z_exit, "allow_short": allow_short},
    }


def _gross_pnl(pos: dict, front_px: float, back_px: float) -> float:
    """Long calendar: short the front, long the back."""
    front_leg = (pos["entry_front_px"] - front_px)      # sold, so a fall is profit
    back_leg = (back_px - pos["entry_back_px"])         # bought
    gross = (front_leg + back_leg) * pos["quantity"]
    return gross if pos["is_long"] else -gross


def _close(pos: dict, exit_date, front_px: float, back_px: float,
           reason: str) -> dict:
    gross = _gross_pnl(pos, front_px, back_px)
    costs = round_trip_cost(pos["entry_front_px"], pos["entry_back_px"],
                            front_px, back_px, pos["quantity"], pos["is_long"])
    exit_debit = back_px - front_px
    return {
        **{k: v for k, v in pos.items()
           if k not in ("last_front_px", "last_back_px")},
        "exit_date": exit_date,
        "exit_reason": reason,
        "exit_front_px": front_px,
        "exit_back_px": back_px,
        "exit_debit": exit_debit,
        "gross_pnl": gross,
        "cost": costs["total"],
        "cost_stt": costs["stt"],
        "cost_slippage": costs["slippage"],
        "net_pnl": gross - costs["total"],
        "days_held": (pd.Timestamp(exit_date) - pd.Timestamp(pos["entry_date"])).days,
    }
