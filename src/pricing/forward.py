"""The forward implied by the option chain itself, via put-call parity.

Every expiry gets its own forward. That matters here: a calendar spread is a bet
across two expiries, and if the two legs are priced off different carry
assumptions the "spread" picks up a term that has nothing to do with volatility.

Parity gives C - P = e^{-rT}(F - K), so F = K + e^{rT}(C - P). Evaluating it at
the strike where |C - P| is smallest keeps the estimate on the most liquid part
of the chain, where both quotes are real rather than stale.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config as C


def synthetic_forward(chain: pd.DataFrame, T: float, r: float = C.RISK_FREE_RATE) -> float:
    """Forward for one (date, expiry) slice of the chain.

    `chain` needs columns: strike, option_type, px. Returns NaN when the slice has
    no strike with both a call and a put.
    """
    calls = chain[chain["option_type"] == "CE"].set_index("strike")["px"]
    puts = chain[chain["option_type"] == "PE"].set_index("strike")["px"]
    common = calls.index.intersection(puts.index)
    if len(common) == 0:
        return float("nan")

    diff = (calls[common] - puts[common]).astype(float)
    atm_strike = diff.abs().idxmin()
    return float(atm_strike + np.exp(r * T) * diff.loc[atm_strike])


def forwards_by_expiry(day: pd.DataFrame, r: float = C.RISK_FREE_RATE) -> pd.DataFrame:
    """One forward per expiry for a single trading day."""
    rows = []
    for expiry, grp in day.groupby("expiry"):
        T = grp["T"].iloc[0]
        if T <= 0:
            continue
        rows.append({"expiry": expiry, "T": T,
                     "forward": synthetic_forward(grp, T, r)})
    return pd.DataFrame(rows)
