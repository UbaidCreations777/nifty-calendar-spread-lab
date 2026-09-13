"""Minute bars for the handful of contracts the strategy actually touches.

The source file is 85 million rows covering every NIFTY option that traded
between 2024 and 2026. The strategy needs four of them per session - two legs,
two option types - so the whole file is scanned once, filtered to the contracts
named by the daily spread series, and cached. What comes out is a few hundred
thousand rows instead of eighty-five million.

**Provenance.** This data is third-party, so it is checked against NSE's own
bhavcopy rather than trusted: `validate_against_bhavcopy` reconstructs daily bars
from the minute series and compares them. Open, high and low match NSE exactly.
Close does not, and should not - NSE settles F&O at the volume-weighted average
of the last half hour, not the last trade - so the comparison uses that VWAP,
which matches to within a fifth of a percent. Volume differs by exactly the lot
size, because this file counts units where bhavcopy counts contracts.

The ticker carries the contract: NIFTY04JAN24CE18300 is an expiry, a right and a
strike. A minority of rows use the single-letter C/P spelling instead of CE/PE.
"""
from __future__ import annotations

import datetime as dt
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .. import config as C

MINUTE_FILE = C.ROOT / "data" / "minute" / "nifty_1min_options_2024_2026.parquet"
CACHE = C.DATA_PROCESSED / "minute_legs.parquet"
SPOT_CACHE = C.DATA_PROCESSED / "minute_spot.parquet"

SPOT_TICKER = "Nifty 50"
_TICKER_RE = re.compile(r"^NIFTY(\d{2}[A-Z]{3}\d{2})(CE|PE|C|P)(\d+)$")

# NSE's F&O closing price is the weighted average of the last half hour.
CLOSING_WINDOW_START = dt.time(15, 0)
SESSION_OPEN = dt.time(9, 15)
SESSION_CLOSE = dt.time(15, 29)


def parse_ticker(ticker: str) -> tuple:
    """NIFTY04JAN24CE18300 -> (date(2024,1,4), 'CE', 18300.0)."""
    m = _TICKER_RE.match(ticker)
    if not m:
        return (None, None, None)
    right = m.group(2)
    if right in ("C", "P"):
        right += "E"
    return (dt.datetime.strptime(m.group(1), "%d%b%y").date(),
            right, float(m.group(3)))


def _ticker_table(tickers: pd.Series) -> pd.DataFrame:
    uniq = pd.DataFrame({"Ticker": pd.unique(tickers)})
    parsed = [parse_ticker(t) for t in uniq["Ticker"]]
    uniq[["expiry", "option_type", "strike"]] = pd.DataFrame(parsed,
                                                             index=uniq.index)
    return uniq.dropna(subset=["expiry"])


def needed_contracts(spreads: pd.DataFrame) -> set:
    """The (session, expiry, strike, option_type) bars the backtest will ask for.

    A signal formed on one session is acted on in the next, so the bars needed
    are the following day's - on the legs chosen the day before.
    """
    days = sorted(spreads["date"].unique())
    following = {d: days[i + 1] for i, d in enumerate(days[:-1])}

    wanted = set()
    for _, r in spreads.iterrows():
        session = following.get(r["date"])
        if session is None:
            continue
        for expiry in (r["front_expiry"], r["back_expiry"]):
            wanted.add((session, expiry, float(r["strike"]), r["option_type"]))
    return wanted


def extract(spreads: pd.DataFrame, path: Path = MINUTE_FILE,
            rebuild: bool = False, verbose: bool = True) -> pd.DataFrame:
    """Scan the minute file once, keeping only the contracts the strategy uses."""
    if CACHE.exists() and not rebuild:
        return pd.read_parquet(CACHE)

    wanted = needed_contracts(spreads)
    if verbose:
        print(f"minute extract: looking for {len(wanted):,} contract-sessions",
              flush=True)

    pf = pq.ParquetFile(path)
    kept = []
    for i in range(pf.metadata.num_row_groups):
        chunk = pf.read_row_group(i).to_pandas()
        meta = _ticker_table(chunk["Ticker"])
        if meta.empty:
            continue
        chunk = chunk.merge(meta, on="Ticker")
        chunk["date"] = pd.to_datetime(chunk["Date"]).dt.date

        keys = list(zip(chunk["date"], chunk["expiry"], chunk["strike"],
                        chunk["option_type"]))
        mask = np.fromiter((k in wanted for k in keys), bool, len(keys))
        if mask.any():
            kept.append(chunk.loc[mask, ["date", "Timestamp", "expiry", "strike",
                                         "option_type", "Open", "High", "Low",
                                         "Close", "Volume", "OI"]])
        if verbose and (i + 1) % 40 == 0:
            print(f"  row group {i + 1}/{pf.metadata.num_row_groups}", flush=True)

    if not kept:
        raise ValueError("no minute bars matched the requested contracts")

    out = pd.concat(kept, ignore_index=True)
    out = out.rename(columns={"Timestamp": "ts", "Open": "open", "High": "high",
                              "Low": "low", "Close": "close",
                              "Volume": "volume", "OI": "oi"})
    out["ts"] = pd.to_datetime(out["ts"])
    out["time"] = out["ts"].dt.time
    out = out.sort_values(["date", "expiry", "strike", "option_type", "ts"])
    C.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out.to_parquet(CACHE, index=False)
    if verbose:
        print(f"minute extract: {len(out):,} bars cached", flush=True)
    return out


def extract_spot(path: Path = MINUTE_FILE, rebuild: bool = False) -> pd.DataFrame:
    """The index itself, minute by minute."""
    if SPOT_CACHE.exists() and not rebuild:
        return pd.read_parquet(SPOT_CACHE)
    tbl = pq.read_table(path, filters=[("Ticker", "=", SPOT_TICKER)])
    df = tbl.to_pandas().rename(columns={"Timestamp": "ts", "Close": "spot"})
    df["date"] = pd.to_datetime(df["Date"]).dt.date
    df["ts"] = pd.to_datetime(df["ts"])
    df["time"] = df["ts"].dt.time
    out = df[["date", "ts", "time", "spot"]].sort_values("ts")
    C.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out.to_parquet(SPOT_CACHE, index=False)
    return out


def validate_against_bhavcopy(minute: pd.DataFrame, raw_dir: Path = C.DATA_RAW,
                              min_volume: int = 1000) -> pd.DataFrame:
    """Reconstruct daily bars from the minute series and compare them to NSE.

    Returns one row per field with the median absolute difference. Open, high and
    low should match exactly; close is compared on NSE's own convention, the
    last-half-hour VWAP.
    """
    dates = sorted(minute["date"].unique())
    frames = []
    for d in dates:
        f = raw_dir / f"{d.isoformat()}.parquet"
        if f.exists():
            frames.append(pd.read_parquet(f))
    if not frames:
        raise FileNotFoundError("no bhavcopy files to validate against")

    bh = pd.concat(frames, ignore_index=True)
    bh = bh[bh["option_type"].isin(["CE", "PE"])].copy()
    bh["date"] = pd.to_datetime(bh["date"]).dt.date
    bh["expiry"] = pd.to_datetime(bh["expiry"]).dt.date

    grp = ["date", "expiry", "strike", "option_type"]
    daily = minute.groupby(grp).agg(
        m_open=("open", "first"), m_high=("high", "max"),
        m_low=("low", "min"), m_ltp=("close", "last"),
        m_volume=("volume", "sum")).reset_index()

    closing = minute[minute["time"] >= CLOSING_WINDOW_START]
    vwap = (closing.groupby(grp)
            .apply(lambda g: np.average(g["close"], weights=g["volume"])
                   if g["volume"].sum() > 0 else np.nan, include_groups=False)
            .rename("m_vwap30").reset_index())
    daily = daily.merge(vwap, on=grp, how="left")

    j = daily.merge(bh[grp + ["open", "high", "low", "close", "volume"]], on=grp)
    j = j[j["m_volume"] > min_volume]

    rows = []
    for mine, theirs, label in [("m_open", "open", "open"),
                                ("m_high", "high", "high"),
                                ("m_low", "low", "low"),
                                ("m_vwap30", "close", "close (vs last-30m VWAP)"),
                                ("m_ltp", "close", "close (vs last trade)")]:
        d = (j[mine] - j[theirs]).abs() / j[theirs].replace(0, np.nan) * 100
        rows.append({"field": label, "n": int(d.notna().sum()),
                     "median_diff_pct": float(d.median()),
                     "within_1pct": float((d < 1).mean() * 100),
                     "exact_match_pct": float((d < 0.001).mean() * 100)})

    ratio = (j["m_volume"] / j["volume"].replace(0, np.nan)).median()
    rows.append({"field": f"volume ratio (units/contracts) = {ratio:.0f}",
                 "n": len(j), "median_diff_pct": np.nan,
                 "within_1pct": np.nan, "exact_match_pct": np.nan})
    return pd.DataFrame(rows).round(3)
