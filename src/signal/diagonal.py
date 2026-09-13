"""Diagonals: two legs, different expiries *and* different strikes.

A calendar is the special case where both strikes are the same. Letting them
differ adds two dimensions to the state a day has to match on - where each leg
sits relative to the forward - and that is the whole difficulty. It is not a
compute problem: precomputing every strike-by-strike combination would run to
roughly 98 million rows, but nobody needs that table. The history for one chosen
structure is a few hundred lookups, so it is built when it is asked for.

The statistical cost is real, though. The calendar model matched on three things;
this matches on five, and a state space gains sparsity fast. So match quality is
reported with every verdict rather than assumed - if the nearest comparable days
are far away, the number on the screen is worth less, and the screen says so.

**Moneyness, not strikes.** A past day is comparable when its legs sat at the
same *distance from the forward*, not at the same strike number. NIFTY was near
19,000 at the start of the sample and near 23,400 at the end; a 24,000 strike is
a different trade in each.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config as C
from . import fair_value as fv

# One unit of distance is "about as different as one day on the front leg".
# NIFTY strikes are 50 apart on a ~23,400 index, so roughly 0.21% - a quarter of
# a percent is therefore close to one strike step.
STATE_SCALE_MONEYNESS = 0.0025

DEFAULT_DTE_TOLERANCE = 3


class Lookup:
    """Chain indexed for repeated per-day leg lookups.

    Built once and reused: a diagonal query walks every earlier session, and
    re-filtering the full chain each time is what would make this slow.
    """

    def __init__(self, chain: pd.DataFrame):
        chain = chain[chain["iv"].notna()].copy()
        self.by_day_expiry: dict = {}
        self.expiries_by_day: dict = {}

        for (d, expiry, opt), grp in chain.groupby(["date", "expiry",
                                                    "option_type"], sort=False):
            grp = grp.sort_values("strike")
            self.by_day_expiry[(d, expiry, opt)] = {
                "strike": grp["strike"].to_numpy(),
                "px": grp["px"].to_numpy(),
                "iv": grp["iv"].to_numpy(),
                "moneyness": grp["moneyness"].to_numpy(),
                "tradeable": grp["tradeable"].to_numpy(),
                "spot": float(grp["spot"].iloc[0]),
                "dte": int(grp["dte"].iloc[0]),
            }

        for (d, expiry), grp in chain.groupby(["date", "expiry"], sort=False):
            self.expiries_by_day.setdefault(d, []).append(
                (expiry, int(grp["dte"].iloc[0])))
        for d in self.expiries_by_day:
            self.expiries_by_day[d].sort(key=lambda t: t[1])

        self.dates = sorted(self.expiries_by_day)

    def expiry_near(self, day, dte: int, tolerance: int):
        """The expiry on `day` whose days-to-expiry is closest to `dte`."""
        best, best_gap = None, None
        for expiry, e_dte in self.expiries_by_day.get(day, []):
            gap = abs(e_dte - dte)
            if gap <= tolerance and (best_gap is None or gap < best_gap):
                best, best_gap = (expiry, e_dte), gap
        return best

    def leg_at(self, day, expiry, option_type: str, moneyness: float):
        """The contract closest to a given distance from the forward."""
        slot = self.by_day_expiry.get((day, expiry, option_type))
        if slot is None or slot["strike"].size == 0:
            return None
        i = int(np.argmin(np.abs(slot["moneyness"] - moneyness)))
        return {"strike": float(slot["strike"][i]), "px": float(slot["px"][i]),
                "iv": float(slot["iv"][i]) * 100.0,
                "moneyness": float(slot["moneyness"][i]),
                "tradeable": bool(slot["tradeable"][i]), "spot": slot["spot"],
                "dte": slot["dte"]}


def structure_on(lookup: Lookup, day, option_type: str, front_dte: int,
                 back_dte: int, front_moneyness: float, back_moneyness: float,
                 dte_tolerance: int = DEFAULT_DTE_TOLERANCE) -> dict | None:
    """The equivalent structure as it stood on one past session."""
    front = lookup.expiry_near(day, front_dte, dte_tolerance)
    back = lookup.expiry_near(day, back_dte, dte_tolerance)
    if front is None or back is None or front[0] == back[0]:
        return None

    f = lookup.leg_at(day, front[0], option_type, front_moneyness)
    b = lookup.leg_at(day, back[0], option_type, back_moneyness)
    if f is None or b is None:
        return None

    spot = f["spot"]
    debit = b["px"] - f["px"]
    return {
        "date": day, "option_type": option_type,
        "front_expiry": front[0], "back_expiry": back[0],
        "front_dte": f["dte"], "back_dte": b["dte"],
        "front_strike": f["strike"], "back_strike": b["strike"],
        "front_moneyness": f["moneyness"], "back_moneyness": b["moneyness"],
        "front_px": f["px"], "back_px": b["px"],
        "front_iv": f["iv"], "back_iv": b["iv"],
        "term_structure": f["iv"] - b["iv"],
        "front_tradeable": f["tradeable"], "back_tradeable": b["tradeable"],
        "spot": spot, "debit": debit, "debit_pct": debit / spot * 100.0,
    }


def _distance(row: dict, target: dict) -> float:
    """How unlike the target state a past day was, in scaled units."""
    return float(np.sqrt(
        ((row["front_dte"] - target["front_dte"]) / C.STATE_SCALE_FRONT_DTE) ** 2
        + ((row["back_dte"] - target["back_dte"]) / C.STATE_SCALE_BACK_DTE) ** 2
        + ((row["front_iv"] - target["front_iv"]) / C.STATE_SCALE_IV) ** 2
        + ((row["front_moneyness"] - target["front_moneyness"])
           / STATE_SCALE_MONEYNESS) ** 2
        + ((row["back_moneyness"] - target["back_moneyness"])
           / STATE_SCALE_MONEYNESS) ** 2))


def history(lookup: Lookup, target: dict, as_of,
            dte_tolerance: int = DEFAULT_DTE_TOLERANCE,
            max_distance: float = C.MAX_STATE_DISTANCE,
            k: int = C.KNN_K) -> pd.DataFrame:
    """What this structure cost on the comparable days before `as_of`."""
    rows = []
    for day in lookup.dates:
        if day >= as_of:                       # point in time, as everywhere
            break
        got = structure_on(lookup, day, target["option_type"],
                           target["front_dte"], target["back_dte"],
                           target["front_moneyness"], target["back_moneyness"],
                           dte_tolerance)
        if got is None:
            continue
        got["state_distance"] = _distance(got, target)
        rows.append(got)

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out = out[out["state_distance"] <= max_distance]
    if out.empty:
        return out
    out = out.nsmallest(k, "state_distance")
    out["weight"] = 1.0 / (1.0 + out["state_distance"])
    return out


def evaluate(lookup: Lookup, option_type: str, as_of,
             front_expiry, front_strike: float,
             back_expiry, back_strike: float,
             z_entry: float = C.Z_ENTRY,
             min_comparables: int = C.MIN_COMPARABLES,
             use_term_structure_filter: bool = True,
             **history_kwargs) -> tuple[dict, pd.DataFrame]:
    """Price one diagonal against its own history. Returns (verdict, history)."""
    f = lookup.by_day_expiry.get((as_of, front_expiry, option_type))
    b = lookup.by_day_expiry.get((as_of, back_expiry, option_type))
    if f is None or b is None:
        return {"signal": fv.INSUFFICIENT, "raw_signal": fv.INSUFFICIENT,
                "error": "one of those legs has no chain on this date"}, pd.DataFrame()

    fi = int(np.argmin(np.abs(f["strike"] - front_strike)))
    bi = int(np.argmin(np.abs(b["strike"] - back_strike)))

    spot = f["spot"]
    debit = float(b["px"][bi] - f["px"][fi])
    target = {
        "date": as_of, "option_type": option_type,
        "front_expiry": front_expiry, "back_expiry": back_expiry,
        "front_dte": f["dte"], "back_dte": b["dte"],
        "front_strike": float(f["strike"][fi]), "back_strike": float(b["strike"][bi]),
        "front_moneyness": float(f["moneyness"][fi]),
        "back_moneyness": float(b["moneyness"][bi]),
        "front_px": float(f["px"][fi]), "back_px": float(b["px"][bi]),
        "front_iv": float(f["iv"][fi]) * 100.0,
        "back_iv": float(b["iv"][bi]) * 100.0,
        "front_tradeable": bool(f["tradeable"][fi]),
        "back_tradeable": bool(b["tradeable"][bi]),
        "spot": spot, "debit": debit, "debit_pct": debit / spot * 100.0,
    }
    target["term_structure"] = target["front_iv"] - target["back_iv"]

    past = history(lookup, target, as_of, **history_kwargs)
    if len(past) < min_comparables:
        return {**target, **fv._blank(len(past))}, past

    stats = fv._weighted_stats(past["debit_pct"].to_numpy(),
                               past["weight"].to_numpy(), target["debit_pct"])
    stats["match_distance"] = float(past["state_distance"].mean())

    if (not np.isfinite(stats["z_score"])
            or abs(stats["fv_mean_pct"]) < 1e-12
            or stats["fv_std_pct"] / abs(stats["fv_mean_pct"])
            < C.MIN_RELATIVE_DISPERSION):
        return ({**target, **stats, "signal": fv.INSUFFICIENT,
                 "raw_signal": fv.INSUFFICIENT, "z_score": np.nan}, past)

    raw = fv._classify(stats["z_score"], z_entry)
    signal = fv._apply_filter(raw, target["term_structure"],
                              use_term_structure_filter)
    return {**target, **stats, "signal": signal, "raw_signal": raw}, past
