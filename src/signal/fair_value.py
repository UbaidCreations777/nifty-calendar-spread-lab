"""The signal: is today's calendar spread cheap or rich against its own history?

The question asked is the one a desk actually asks. Not "what does a model say
this is worth", but: the last time the front leg had 4 days left, the back leg
11, and ATM vol was 16 - what did this structure cost? If it cost 0.53% of spot
then and it costs 0.37% now, it is cheap, and the reason is not a model's
opinion.

Finding those comparable days is the whole problem. An exact match on both legs'
days-to-expiry sounds right and does not work: NSE moved NIFTY's weekly expiry
from Thursday to Tuesday during 2025, which splits the state space into two eras
and leaves the current one matching almost nothing. Measured over three years,
an exact rule reaches eight comparables on 0.1% of days.

So comparability is a distance rather than a bucket. Neighbours are taken from a
scaled state space - one unit is about as different as one day on the front leg -
and the nearest K are used, weighted so closer days count for more. A cap keeps
that honest: past a certain distance a historical day is not the same trade, and
is dropped even when that leaves too few days to say anything.

Two rules apply whichever way the matching is done:

Point in time. Comparables for day t come only from days strictly before t. An
in-sample mean would let a 2026 trade be judged against 2026 data it could not
have seen, which is the most common way a backtest of this kind ends up
profitable and untradeable.

A minimum sample, and a minimum spread. Below `MIN_COMPARABLES` matching days
there is no distribution. And if those days barely vary, the z-score is not
trusted either - dividing by a near-zero standard deviation turns rounding into
conviction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config as C

INSUFFICIENT = "insufficient_data"
BUY = "buy"
SELL = "sell"
FLAT = "flat"

KNN = "knn"
BUCKET = "bucket"


def _state_distance(front_dte, back_dte, iv, i, past) -> np.ndarray:
    """Distance from day i to each earlier day, in scaled state units."""
    d_front = (front_dte[past] - front_dte[i]) / C.STATE_SCALE_FRONT_DTE
    d_back = (back_dte[past] - back_dte[i]) / C.STATE_SCALE_BACK_DTE
    d_iv = (iv[past] - iv[i]) / C.STATE_SCALE_IV
    return np.sqrt(d_front**2 + d_back**2 + d_iv**2)


def _weighted_stats(values: np.ndarray, weights: np.ndarray,
                    today: float) -> dict:
    """Weighted mean, standard deviation and percentile of the comparables.

    The standard deviation carries the reliability-weight correction, so a set of
    fifteen neighbours of differing closeness is not treated as fifteen equally
    good observations.
    """
    w_sum = weights.sum()
    mean = float((values * weights).sum() / w_sum)

    denom = w_sum - (weights**2).sum() / w_sum
    if denom > 1e-12:
        var = float((weights * (values - mean) ** 2).sum() / denom)
    else:
        var = 0.0
    std = float(np.sqrt(max(var, 0.0)))

    z = (today - mean) / std if std > 1e-12 else np.nan
    pct = float((weights * (values < today)).sum() / w_sum * 100.0)

    return {
        "n_comparables": int(values.size),
        "fv_mean_pct": mean,
        "fv_std_pct": std,
        "fv_min_pct": float(values.min()),
        "fv_max_pct": float(values.max()),
        "z_score": float(z) if np.isfinite(z) else np.nan,
        "percentile": pct,
        "cheapness_pct": ((today - mean) / mean * 100.0
                          if abs(mean) > 1e-12 else np.nan),
    }


def _classify(z: float, z_entry: float) -> str:
    """What the z-score alone says, before any confirmation."""
    if z <= -z_entry:
        return BUY                    # spread trades below its own history
    if z >= z_entry:
        return SELL                   # spread trades above its own history
    return FLAT


def _apply_filter(raw: str, term_structure: float, enabled: bool) -> str:
    """Confirmation only, never a trigger.

    A long calendar wants the leg it sells to carry more vol than the leg it
    buys; when the term structure says the opposite, the cheapness is more likely
    justified than mispriced. The reverse holds for a short. The unfiltered call
    is kept alongside so the dashboard can show what was overruled and why.
    """
    if not enabled:
        return raw
    if raw == BUY and term_structure <= 0:
        return FLAT
    if raw == SELL and term_structure >= 0:
        return FLAT
    return raw


def _blank(n: int) -> dict:
    return {"n_comparables": n, "signal": INSUFFICIENT,
            "raw_signal": INSUFFICIENT, "z_score": np.nan,
            "percentile": np.nan, "fv_mean_pct": np.nan, "fv_std_pct": np.nan,
            "fv_min_pct": np.nan, "fv_max_pct": np.nan,
            "cheapness_pct": np.nan, "match_distance": np.nan}


def evaluate(spreads: pd.DataFrame,
             method: str = KNN,
             k: int = C.KNN_K,
             max_distance: float = C.MAX_STATE_DISTANCE,
             iv_half_width: float = C.IV_BUCKET_HALF_WIDTH,
             back_dte_tolerance: int = C.BACK_DTE_TOLERANCE,
             min_comparables: int = C.MIN_COMPARABLES,
             z_entry: float = C.Z_ENTRY,
             use_term_structure_filter: bool = True) -> pd.DataFrame:
    """Attach a fair-value verdict to every row of the daily spread series.

    `method` is "knn" - nearest neighbours in scaled state space, capped by
    distance - or "bucket", the fixed-tolerance rule kept as the comparison
    baseline.
    """
    out = []
    spreads = spreads.sort_values("date").reset_index(drop=True)

    for opt in spreads["option_type"].unique():
        book = spreads[spreads["option_type"] == opt].reset_index(drop=True)
        debit = book["debit_pct"].to_numpy()
        front_dte = book["front_dte"].to_numpy()
        back_dte = book["back_dte"].to_numpy()
        iv = book["front_iv"].to_numpy()
        dates = book["date"].to_numpy()

        for i in range(len(book)):
            row = book.iloc[i].to_dict()
            # Strictly earlier *days*, not earlier rows. When the series holds
            # several expiry pairs per session, the rows above this one include
            # today's other pairs - which were not knowable when today opened.
            past = np.flatnonzero(dates < dates[i])

            if method == KNN:
                if past.size == 0:
                    out.append({**row, **_blank(0)})
                    continue
                dist = _state_distance(front_dte, back_dte, iv, i, past)
                within = dist <= max_distance
                idx, dist = past[within], dist[within]
                if idx.size > k:                 # keep the k closest
                    keep = np.argsort(dist)[:k]
                    idx, dist = idx[keep], dist[keep]
                weights = 1.0 / (1.0 + dist)
                match_distance = float(dist.mean()) if dist.size else np.nan
            else:
                idx = past[
                    (front_dte[past] == front_dte[i])
                    & (np.abs(back_dte[past] - back_dte[i]) <= back_dte_tolerance)
                    & (np.abs(iv[past] - iv[i]) <= iv_half_width)
                ]
                weights = np.ones(idx.size)
                match_distance = np.nan

            if idx.size < min_comparables:
                out.append({**row, **_blank(int(idx.size))})
                continue

            stats = _weighted_stats(debit[idx], weights, debit[i])
            stats["match_distance"] = match_distance

            # No usable width means no z-score, however far today sits from the
            # mean. Reporting it as flat would be a quiet wrong answer in the one
            # direction that matters.
            if (not np.isfinite(stats["z_score"])
                    or abs(stats["fv_mean_pct"]) < 1e-12
                    or stats["fv_std_pct"] / abs(stats["fv_mean_pct"])
                    < C.MIN_RELATIVE_DISPERSION):
                out.append({**row, **stats, "signal": INSUFFICIENT,
                            "raw_signal": INSUFFICIENT, "z_score": np.nan})
                continue

            raw = _classify(stats["z_score"], z_entry)
            signal = _apply_filter(raw, row["term_structure"],
                                   use_term_structure_filter)
            out.append({**row, **stats, "signal": signal, "raw_signal": raw})

    cols_first = ["date", "option_type", "signal", "raw_signal", "z_score",
                  "percentile",
                  "debit", "debit_pct", "fv_mean_pct", "n_comparables",
                  "match_distance"]
    df = pd.DataFrame(out).sort_values(["date", "option_type"]).reset_index(drop=True)
    rest = [c for c in df.columns if c not in cols_first]
    return df[cols_first + rest]


def comparables(spreads: pd.DataFrame, option_type: str, as_of,
                method: str = KNN, k: int = C.KNN_K,
                max_distance: float = C.MAX_STATE_DISTANCE,
                iv_half_width: float = C.IV_BUCKET_HALF_WIDTH,
                back_dte_tolerance: int = C.BACK_DTE_TOLERANCE,
                front_dte: int | None = None,
                back_dte: int | None = None) -> pd.DataFrame:
    """The historical days a given verdict was actually built from.

    The dashboard plots what the model used rather than re-deriving a similar set
    with its own filter, so the chart cannot drift away from the signal.

    `front_dte`/`back_dte` pick one expiry pair out of a session that holds
    several; without them the first pair for that date is used.
    """
    book = (spreads[spreads["option_type"] == option_type]
            .sort_values("date").reset_index(drop=True))
    loc = book.index[book["date"] == as_of]
    if front_dte is not None:
        loc = [j for j in loc if book.loc[j, "front_dte"] == front_dte
               and book.loc[j, "back_dte"] == back_dte]
    if len(loc) == 0:
        return book.iloc[0:0]
    i = int(loc[0])
    past = np.flatnonzero(book["date"].to_numpy() < book["date"].to_numpy()[i])
    if past.size == 0:
        return book.iloc[0:0]

    front_dte = book["front_dte"].to_numpy()
    back_dte = book["back_dte"].to_numpy()
    iv = book["front_iv"].to_numpy()

    if method == KNN:
        dist = _state_distance(front_dte, back_dte, iv, i, past)
        within = dist <= max_distance
        idx, dist = past[within], dist[within]
        if idx.size > k:
            keep = np.argsort(dist)[:k]
            idx, dist = idx[keep], dist[keep]
        out = book.iloc[idx].copy()
        out["state_distance"] = dist
        out["weight"] = 1.0 / (1.0 + dist)
        return out

    idx = past[
        (front_dte[past] == front_dte[i])
        & (np.abs(back_dte[past] - back_dte[i]) <= back_dte_tolerance)
        & (np.abs(iv[past] - iv[i]) <= iv_half_width)
    ]
    out = book.iloc[idx].copy()
    out["state_distance"] = np.nan
    out["weight"] = 1.0
    return out


def evaluate_one(spreads: pd.DataFrame, option_type: str, as_of,
                 front_dte: int, back_dte: int,
                 z_entry: float = C.Z_ENTRY,
                 use_term_structure_filter: bool = True,
                 min_comparables: int = C.MIN_COMPARABLES,
                 **match_kwargs) -> dict:
    """The verdict for one expiry pair on one day.

    `evaluate` scores an entire series; this scores a single structure a user has
    asked about. Both read their comparables through `comparables`, so an
    arbitrary pair picked in the dashboard is judged by exactly the rule the
    strategy is judged by.
    """
    book = spreads[(spreads["option_type"] == option_type)
                   & (spreads["date"] == as_of)
                   & (spreads["front_dte"] == front_dte)
                   & (spreads["back_dte"] == back_dte)]
    if book.empty:
        return {"signal": INSUFFICIENT, "raw_signal": INSUFFICIENT,
                "error": "no such pair on this date"}

    row = book.iloc[0].to_dict()
    comps = comparables(spreads, option_type, as_of, front_dte=front_dte,
                        back_dte=back_dte, **match_kwargs)

    if len(comps) < min_comparables:
        return {**row, **_blank(len(comps))}

    stats = _weighted_stats(comps["debit_pct"].to_numpy(),
                            comps["weight"].to_numpy(), row["debit_pct"])
    stats["match_distance"] = float(comps["state_distance"].mean()) \
        if comps["state_distance"].notna().any() else np.nan

    if (not np.isfinite(stats["z_score"])
            or abs(stats["fv_mean_pct"]) < 1e-12
            or stats["fv_std_pct"] / abs(stats["fv_mean_pct"])
            < C.MIN_RELATIVE_DISPERSION):
        return {**row, **stats, "signal": INSUFFICIENT,
                "raw_signal": INSUFFICIENT, "z_score": np.nan}

    raw = _classify(stats["z_score"], z_entry)
    signal = _apply_filter(raw, row["term_structure"], use_term_structure_filter)
    return {**row, **stats, "signal": signal, "raw_signal": raw}


def latest_view(evaluated: pd.DataFrame) -> pd.DataFrame:
    """The most recent day's verdict for each option type - what a desk would
    look at in the morning."""
    last = evaluated["date"].max()
    return evaluated[evaluated["date"] == last]
