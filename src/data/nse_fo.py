"""Downloads and normalises the NSE F&O bhavcopy.

NSE returns 401 to any request that has not first collected the cookies its
homepage sets, so every download primes a session and re-primes once on a 401.

The archive exists in two layouts: the UDiFF format from 2024-07-08 onward and
the legacy format before it. Both are tried per date and whichever answers wins,
so a moved path shows up as a named failure rather than a silently empty folder.
"""
from __future__ import annotations

import io
import time
import zipfile
from datetime import date
from pathlib import Path

import pandas as pd
import requests

from .. import config as C

_MONTHS = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN",
           "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")

_session: requests.Session | None = None
_primed_at = 0.0

# Unified schema every downloaded file is normalised into.
COLUMNS = ["date", "symbol", "instrument", "expiry", "strike", "option_type",
           "open", "high", "low", "close", "settle", "underlying",
           "open_interest", "volume", "lot_size"]


def _fmt(url: str, d: date) -> str:
    return url.format(
        dd=f"{d.day:02d}", mm=f"{d.month:02d}", yyyy=f"{d.year:04d}",
        MON=_MONTHS[d.month - 1], YYYYMMDD=f"{d.year:04d}{d.month:02d}{d.day:02d}",
    )


def _get_session() -> requests.Session:
    global _session
    if _session is None:
        s = requests.Session()
        s.headers.update({
            "User-Agent": C.USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": C.NSE_BASE + "/all-reports-derivatives",
            "Connection": "keep-alive",
        })
        _session = s
    return _session


def prime(force: bool = False) -> tuple[bool, str]:
    """Collect NSE's cookies. Returns (ok, message)."""
    global _primed_at
    if not force and (time.time() - _primed_at) < 600:
        return True, "already primed"
    try:
        s = _get_session()
        r = s.get(C.NSE_BASE + "/", timeout=C.TIMEOUT)
        time.sleep(0.5)
        s.get(C.NSE_BASE + "/all-reports-derivatives", timeout=C.TIMEOUT)
        _primed_at = time.time()
        if len(s.cookies) == 0:
            return False, (f"homepage answered HTTP {r.status_code} but set no "
                           "cookies - NSE may be blocking this network")
        return True, f"got {len(s.cookies)} cookies"
    except Exception as e:
        return False, f"{e.__class__.__name__}: {e}"


def _unzip(content: bytes) -> str | None:
    try:
        zf = zipfile.ZipFile(io.BytesIO(content))
    except zipfile.BadZipFile:
        return None
    names = [n for n in zf.namelist() if n.lower().endswith((".csv", ".dat"))]
    if not names:
        return None
    return zf.read(names[0]).decode("utf-8", "replace")


def _normalise_udiff(df: pd.DataFrame, d: date) -> pd.DataFrame:
    df = df[df["TckrSymb"] == C.UNDERLYING].copy()
    df = df[df["FinInstrmTp"].isin(["IDO", "IDF", "STO", "STF"])]
    out = pd.DataFrame({
        "date": d,
        "symbol": df["TckrSymb"],
        "instrument": df["FinInstrmTp"],
        "expiry": pd.to_datetime(df["XpryDt"]).dt.date,
        "strike": pd.to_numeric(df["StrkPric"], errors="coerce"),
        "option_type": df["OptnTp"].fillna("FUT"),
        "open": pd.to_numeric(df["OpnPric"], errors="coerce"),
        "high": pd.to_numeric(df["HghPric"], errors="coerce"),
        "low": pd.to_numeric(df["LwPric"], errors="coerce"),
        "close": pd.to_numeric(df["ClsPric"], errors="coerce"),
        "settle": pd.to_numeric(df["SttlmPric"], errors="coerce"),
        "underlying": pd.to_numeric(df["UndrlygPric"], errors="coerce"),
        "open_interest": pd.to_numeric(df["OpnIntrst"], errors="coerce"),
        "volume": pd.to_numeric(df["TtlTradgVol"], errors="coerce"),
        "lot_size": pd.to_numeric(df["NewBrdLotQty"], errors="coerce"),
    })
    return out


def _normalise_legacy(df: pd.DataFrame, d: date) -> pd.DataFrame:
    df.columns = [c.strip().upper() for c in df.columns]
    df = df[df["SYMBOL"] == C.UNDERLYING].copy()
    inst_map = {"OPTIDX": "IDO", "FUTIDX": "IDF",
                "OPTSTK": "STO", "FUTSTK": "STF"}
    df = df[df["INSTRUMENT"].isin(inst_map)]
    out = pd.DataFrame({
        "date": d,
        "symbol": df["SYMBOL"],
        "instrument": df["INSTRUMENT"].map(inst_map),
        "expiry": pd.to_datetime(df["EXPIRY_DT"], format="%d-%b-%Y").dt.date,
        "strike": pd.to_numeric(df["STRIKE_PR"], errors="coerce"),
        "option_type": df["OPTION_TYP"].replace({"XX": "FUT"}),
        "open": pd.to_numeric(df["OPEN"], errors="coerce"),
        "high": pd.to_numeric(df["HIGH"], errors="coerce"),
        "low": pd.to_numeric(df["LOW"], errors="coerce"),
        "close": pd.to_numeric(df["CLOSE"], errors="coerce"),
        "settle": pd.to_numeric(df["SETTLE_PR"], errors="coerce"),
        "underlying": pd.NA,   # legacy files carry no spot; filled from futures
        "open_interest": pd.to_numeric(df["OPEN_INT"], errors="coerce"),
        "volume": pd.to_numeric(df["CONTRACTS"], errors="coerce"),
        "lot_size": pd.NA,
    })
    return out


def _fill_underlying(df: pd.DataFrame) -> pd.DataFrame:
    """Legacy files omit spot. The near-month future's close is the best proxy
    available in the same file, so it stands in where spot is missing."""
    if df.empty or df["underlying"].notna().any():
        return df
    fut = df[df["instrument"].isin(["IDF", "STF"])]
    if fut.empty:
        return df
    near = fut.sort_values("expiry").iloc[0]
    df = df.copy()
    df["underlying"] = near["close"]
    return df


def fetch_day(d: date) -> tuple[pd.DataFrame | None, str]:
    """Download and normalise one trading day. Returns (frame, note)."""
    ok, msg = prime()
    if not ok:
        return None, f"cookie handshake failed - {msg}"

    tried = []
    for src in C.SOURCES:
        url = _fmt(src["url"], d)
        tried.append(src["name"])
        for attempt in (1, 2):
            try:
                r = _get_session().get(url, timeout=C.TIMEOUT)
            except Exception as e:
                if attempt == 1:
                    time.sleep(2)
                    continue
                break

            if r.status_code == 404:
                break
            if r.status_code in (401, 403) and attempt == 1:
                prime(force=True)
                continue
            if r.status_code != 200:
                break

            text = _unzip(r.content)
            if text is None:
                break
            try:
                raw = pd.read_csv(io.StringIO(text))
            except Exception as e:
                return None, f"{src['name']}: unreadable csv - {e}"

            raw.columns = [c.strip() for c in raw.columns]
            if src["name"] == "udiff":
                out = _normalise_udiff(raw, d)
            else:
                out = _normalise_legacy(raw, d)
            out = _fill_underlying(out)
            return out[COLUMNS], src["name"]
        time.sleep(C.REQUEST_DELAY)

    return None, f"no source answered (tried {', '.join(tried)})"


def cache_path(d: date) -> Path:
    return C.DATA_RAW / f"{d.isoformat()}.parquet"


def download_range(start: date, end: date, verbose: bool = True) -> dict:
    """Download every weekday in [start, end] that is not already cached."""
    C.DATA_RAW.mkdir(parents=True, exist_ok=True)
    stats = {"cached": 0, "downloaded": 0, "holiday": 0, "failed": 0}
    failures = []

    for d in pd.date_range(start, end, freq="B").date:
        path = cache_path(d)
        if path.exists():
            stats["cached"] += 1
            continue

        df, note = fetch_day(d)
        if df is None:
            # NSE has no file on exchange holidays, which is not an error.
            if "no source answered" in note:
                stats["holiday"] += 1
                path.with_suffix(".holiday").touch()
            else:
                stats["failed"] += 1
                failures.append((d, note))
                if verbose:
                    print(f"  {d} FAILED: {note}", flush=True)
            continue

        df.to_parquet(path, index=False)
        stats["downloaded"] += 1
        if verbose and stats["downloaded"] % 25 == 0:
            print(f"  {d}: {stats['downloaded']} days downloaded "
                  f"({note}, {len(df)} rows)", flush=True)
        time.sleep(C.REQUEST_DELAY)

    stats["failures"] = failures
    return stats


def load_cached(start: date | None = None, end: date | None = None) -> pd.DataFrame:
    """Load every cached day into one frame."""
    files = sorted(C.DATA_RAW.glob("*.parquet"))
    frames = []
    for f in files:
        d = date.fromisoformat(f.stem)
        if start and d < start:
            continue
        if end and d > end:
            continue
        frames.append(pd.read_parquet(f))
    if not frames:
        raise FileNotFoundError(
            f"no cached bhavcopy in {C.DATA_RAW}. Run: python -m src.data.download")
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["expiry"] = pd.to_datetime(df["expiry"]).dt.date
    return df
