"""Turns raw bhavcopy rows into a dated option chain with forwards and IVs.

Three data decisions are made here, and each one changes what the strategy sees:

1. Settlement price, not close. NSE marks every contract at end of day, including
   ones that never traded; `close` on an untraded strike is whatever it last
   printed, which can be days stale. Backtesting a spread off stale closes
   manufactures profit that was never available.

2. A liquidity filter for anything tradeable. Settlement prices are fine for
   valuation but a contract with no volume could not actually have been entered,
   so entries are restricted to strikes that traded.

3. Strikes within +/-10% of spot. The wings are illiquid, their IVs are noisy,
   and no ATM calendar ever reaches them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config as C
from ..pricing.forward import synthetic_forward
from ..pricing.implied_vol import implied_vol

STRIKE_WINDOW = 0.10


def _dte(day: pd.Series, expiry: pd.Series) -> pd.Series:
    return (pd.to_datetime(expiry) - pd.to_datetime(day)).dt.days


def classify_expiries(expiries: pd.Series) -> pd.DataFrame:
    """Label each expiry weekly or monthly.

    An expiry is the monthly one when it is the last expiry in its calendar
    month. Deriving it from the data rather than hardcoding "last Thursday"
    matters because NSE moved NIFTY's expiry day during the sample period.
    """
    ex = pd.DataFrame({"expiry": pd.Series(sorted(set(expiries)))})
    ex["ym"] = pd.to_datetime(ex["expiry"]).dt.to_period("M")
    last = ex.groupby("ym")["expiry"].transform("max")
    ex["kind"] = np.where(ex["expiry"] == last, "monthly", "weekly")
    return ex[["expiry", "kind"]]


def build_day(raw_day: pd.DataFrame, expiry_kinds: pd.DataFrame) -> pd.DataFrame:
    """Enrich one trading day: forwards per expiry, then IV per contract."""
    opts = raw_day[raw_day["option_type"].isin(["CE", "PE"])].copy()
    if opts.empty:
        return pd.DataFrame()

    spot = opts["underlying"].dropna()
    if spot.empty:
        return pd.DataFrame()
    spot = float(spot.iloc[0])

    opts = opts[(opts["strike"] > spot * (1 - STRIKE_WINDOW))
                & (opts["strike"] < spot * (1 + STRIKE_WINDOW))]
    opts["px"] = opts["settle"].where(opts["settle"] > 0, opts["close"])
    opts = opts[opts["px"] > 0]
    if opts.empty:
        return pd.DataFrame()

    # The day's first and last traded prices, kept for the intraday variant.
    # Unlike settlement these exist only where the contract actually traded, so
    # they stay NaN rather than being filled with a stale number.
    opts["open_px"] = opts["open"].where(opts["open"] > 0)
    opts["close_px"] = opts["close"].where(opts["close"] > 0)

    opts["dte"] = _dte(opts["date"], opts["expiry"])
    opts = opts[opts["dte"] > 0]
    opts["T"] = opts["dte"] / 365.0
    opts["spot"] = spot

    fwd = {}
    for expiry, grp in opts.groupby("expiry"):
        fwd[expiry] = synthetic_forward(grp, float(grp["T"].iloc[0]))
    opts["forward"] = opts["expiry"].map(fwd)
    opts = opts[opts["forward"].notna() & (opts["forward"] > 0)]
    if opts.empty:
        return pd.DataFrame()

    opts["iv"] = implied_vol(
        opts["px"].to_numpy(), opts["forward"].to_numpy(), opts["strike"].to_numpy(),
        opts["T"].to_numpy(), C.RISK_FREE_RATE, opts["option_type"].to_numpy())

    opts["moneyness"] = opts["strike"] / opts["forward"] - 1.0
    opts = opts.merge(expiry_kinds, on="expiry", how="left")
    opts["tradeable"] = opts["volume"].fillna(0) > 0
    # Legacy-format files carry no lot size and arrive as pd.NA in an object
    # column, which will not survive arithmetic downstream.
    opts["lot_size"] = pd.to_numeric(opts["lot_size"], errors="coerce")

    cols = ["date", "expiry", "kind", "dte", "T", "strike", "option_type", "px",
            "open_px", "close_px", "spot", "forward", "iv", "moneyness",
            "open_interest", "volume", "lot_size", "tradeable"]
    return opts[cols]


def build_chain(raw: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Enrich every day in the raw frame."""
    expiry_kinds = classify_expiries(raw["expiry"])
    out = []
    days = sorted(raw["date"].unique())
    for i, d in enumerate(days):
        day = build_day(raw[raw["date"] == d], expiry_kinds)
        if not day.empty:
            out.append(day)
        if verbose and (i + 1) % 100 == 0:
            print(f"  chain: {i + 1}/{len(days)} days", flush=True)
    if not out:
        raise ValueError("no usable chain data was built from the raw bhavcopy")
    return pd.concat(out, ignore_index=True)


def atm_iv(chain: pd.DataFrame) -> pd.DataFrame:
    """ATM implied vol per (date, expiry, option_type).

    ATM is the strike nearest the forward, which is where the calendar sits and
    where the quote is most reliable.
    """
    df = chain[chain["iv"].notna()].copy()
    df["abs_moneyness"] = df["moneyness"].abs()
    idx = df.groupby(["date", "expiry", "option_type"])["abs_moneyness"].idxmin()
    out = df.loc[idx, ["date", "expiry", "kind", "dte", "option_type", "strike",
                       "iv", "spot", "forward", "px"]]
    return out.rename(columns={"iv": "atm_iv", "strike": "atm_strike"})
