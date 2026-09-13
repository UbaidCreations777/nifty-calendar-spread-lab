"""Does the signal predict anything, before any question of trading it?

A backtest answers "would this have made money", which bundles the signal
together with the structure's carry, the exit rule, the sizing and the costs. If
the answer is no, that tells you nothing about which of those five was at fault.

This separates the first one out. For every scored day it measures what the same
legs - fixed strike, fixed expiries - were worth some days later, and groups that
by the z-score at the time. If cheap spreads subsequently widen toward their fair
value and rich ones narrow, the signal has information in it, whatever the P&L of
trading it turns out to be.

Two P&L conventions are reported side by side, because they answer different
questions:

  spread_change  - what the structure did, direction ignored
  signal_pnl     - the same move signed by what the signal said to do, so a
                   positive number means the signal was right
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import fair_value as fv


def _price_lookup(chain: pd.DataFrame) -> dict:
    keys = zip(chain["date"], chain["expiry"], chain["strike"],
               chain["option_type"])
    return dict(zip(keys, chain["px"]))


def forward_moves(evaluated: pd.DataFrame, chain: pd.DataFrame,
                  horizons=(1, 2, 3, 5)) -> pd.DataFrame:
    """What each scored day's spread was worth `h` trading days later."""
    prices = _price_lookup(chain)
    dates = sorted(chain["date"].unique())
    pos = {d: i for i, d in enumerate(dates)}

    scored = evaluated[evaluated["z_score"].notna()].copy()
    rows = []

    for _, r in scored.iterrows():
        i = pos.get(r["date"])
        if i is None:
            continue
        base = {"date": r["date"], "option_type": r["option_type"],
                "z_score": r["z_score"], "signal": r["signal"],
                "debit_pct": r["debit_pct"], "spot": r["spot"],
                "front_dte": r["front_dte"]}

        for h in horizons:
            j = i + h
            if j >= len(dates):
                continue
            later = dates[j]
            f = prices.get((later, r["front_expiry"], r["strike"],
                            r["option_type"]))
            b = prices.get((later, r["back_expiry"], r["strike"],
                            r["option_type"]))
            if f is None or b is None:
                continue

            # Same legs, same strike - only time and the market have moved.
            later_pct = (b - f) / r["spot"] * 100.0
            change = later_pct - r["debit_pct"]

            # A buy wants the spread to widen, a sell wants it to narrow.
            direction = -1.0 if r["z_score"] > 0 else 1.0
            rows.append({**base, "horizon": h, "later_debit_pct": later_pct,
                         "spread_change": change,
                         "signal_pnl": change * direction})

    return pd.DataFrame(rows)


def by_z_bucket(moves: pd.DataFrame, horizon: int = 3) -> pd.DataFrame:
    """Average subsequent move, grouped by how stretched the signal was.

    If the signal carries information, `mean_change` should fall monotonically
    across the buckets: spreads that looked cheap widen, spreads that looked rich
    narrow.
    """
    h = moves[moves["horizon"] == horizon]
    if h.empty:
        return pd.DataFrame()

    edges = [-np.inf, -2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2, np.inf]
    labels = ["< -2", "-2 to -1.5", "-1.5 to -1", "-1 to -0.5", "-0.5 to 0.5",
              "0.5 to 1", "1 to 1.5", "1.5 to 2", "> 2"]
    h = h.assign(bucket=pd.cut(h["z_score"], edges, labels=labels))

    return (h.groupby("bucket", observed=True)
            .agg(n=("spread_change", "size"),
                 mean_change=("spread_change", "mean"),
                 median_change=("spread_change", "median"),
                 mean_signal_pnl=("signal_pnl", "mean"),
                 hit_rate=("signal_pnl", lambda s: (s > 0).mean() * 100))
            .round(4).reset_index())


def by_horizon(moves: pd.DataFrame, z_entry: float) -> pd.DataFrame:
    """Whether the signal's edge shows up quickly or not at all."""
    traded = moves[moves["z_score"].abs() >= z_entry]
    if traded.empty:
        return pd.DataFrame()
    return (traded.groupby(["horizon", "signal"], observed=True)
            .agg(n=("signal_pnl", "size"),
                 mean_signal_pnl=("signal_pnl", "mean"),
                 hit_rate=("signal_pnl", lambda s: (s > 0).mean() * 100))
            .round(4).reset_index())


def decompose(moves: pd.DataFrame, horizon: int, z_entry: float) -> pd.DataFrame:
    """Split the move into the part the signal called and the part it did not.

    The structure carries regardless of the signal: a short calendar owns the
    fast-decaying leg and bleeds, a long one collects. Comparing the signalled
    trades against every other day at the same horizon shows how much of the
    result is the call and how much is the carry.
    """
    h = moves[moves["horizon"] == horizon]
    if h.empty:
        return pd.DataFrame()
    traded = h["z_score"].abs() >= z_entry
    return pd.DataFrame([
        {"group": "signalled (|z| >= entry)", "n": int(traded.sum()),
         "mean_spread_change": float(h.loc[traded, "spread_change"].mean()),
         "mean_signal_pnl": float(h.loc[traded, "signal_pnl"].mean())},
        {"group": "all other scored days", "n": int((~traded).sum()),
         "mean_spread_change": float(h.loc[~traded, "spread_change"].mean()),
         "mean_signal_pnl": float(h.loc[~traded, "signal_pnl"].mean())},
    ]).round(4)
