"""Builds the daily ATM calendar spread series that the signal is computed on.

A calendar here is: sell the near expiry, buy the further-out monthly, same
strike, struck at the money. What gets recorded each day is the net debit - what
the structure costs - along with the state it was priced in: both legs' days to
expiry and the front leg's ATM implied vol.

The debit is stored as a percentage of spot as well as in points. NIFTY moved
from roughly 19,000 to 26,000 across this sample, and option premium scales with
the level of the underlying, so 100 points in 2023 and 100 points in 2026 are not
the same trade. Comparing raw points across that range would read a drift in the
index as a change in the richness of the spread.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config as C


def _pick_legs(day_expiries: pd.DataFrame) -> tuple[pd.Series, pd.Series] | None:
    """Nearest expiry as the front, nearest monthly beyond it as the back."""
    front_pool = day_expiries[
        (day_expiries["dte"] >= C.FRONT_DTE_MIN)
        & (day_expiries["dte"] <= C.FRONT_DTE_MAX)
    ].sort_values("dte")
    if front_pool.empty:
        return None
    front = front_pool.iloc[0]

    back_pool = day_expiries[
        (day_expiries["kind"] == "monthly")
        & (day_expiries["expiry"] > front["expiry"])
        & (day_expiries["dte"] >= C.BACK_DTE_MIN)
        & (day_expiries["dte"] <= C.BACK_DTE_MAX)
    ].sort_values("dte")
    if back_pool.empty:
        return None
    return front, back_pool.iloc[0]


def build_spreads(chain: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """One ATM calendar per (date, option_type)."""
    rows = []
    dates = sorted(chain["date"].unique())

    for i, d in enumerate(dates):
        day = chain[chain["date"] == d]
        expiries = (day[["expiry", "kind", "dte"]]
                    .drop_duplicates().sort_values("dte"))
        legs = _pick_legs(expiries)
        if legs is None:
            continue
        front, back = legs

        for opt in ("CE", "PE"):
            f = day[(day["expiry"] == front["expiry"]) & (day["option_type"] == opt)]
            b = day[(day["expiry"] == back["expiry"]) & (day["option_type"] == opt)]
            if f.empty or b.empty:
                continue

            # Both legs must exist at the same strike for this to be a calendar.
            common = np.intersect1d(f["strike"].to_numpy(), b["strike"].to_numpy())
            if common.size == 0:
                continue

            forward = float(f["forward"].iloc[0])
            atm_strike = float(common[np.argmin(np.abs(common - forward))])
            fr = f[f["strike"] == atm_strike].iloc[0]
            bk = b[b["strike"] == atm_strike].iloc[0]
            if not np.isfinite(fr["iv"]) or not np.isfinite(bk["iv"]):
                continue

            spot = float(fr["spot"])
            debit = float(bk["px"] - fr["px"])
            rows.append({
                "date": d,
                "option_type": opt,
                "strike": atm_strike,
                "spot": spot,
                "front_expiry": front["expiry"],
                "back_expiry": back["expiry"],
                "front_dte": int(front["dte"]),
                "back_dte": int(back["dte"]),
                "front_kind": front["kind"],
                "front_px": float(fr["px"]),
                "back_px": float(bk["px"]),
                "debit": debit,
                "debit_pct": debit / spot * 100.0,
                "front_iv": float(fr["iv"]) * 100.0,
                "back_iv": float(bk["iv"]) * 100.0,
                "term_structure": (float(fr["iv"]) - float(bk["iv"])) * 100.0,
                "front_tradeable": bool(fr["tradeable"]),
                "back_tradeable": bool(bk["tradeable"]),
                "lot_size": float(fr["lot_size"]) if np.isfinite(fr["lot_size"]) else np.nan,
            })

        if verbose and (i + 1) % 200 == 0:
            print(f"  spreads: {i + 1}/{len(dates)} days", flush=True)

    if not rows:
        raise ValueError("no calendar spreads could be built - check DTE windows")
    return pd.DataFrame(rows).sort_values(["date", "option_type"]).reset_index(drop=True)
