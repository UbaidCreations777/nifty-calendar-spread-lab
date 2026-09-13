"""How much of the result is the signal, and how much is the parameters?

Every matching rule in this project is a judgement call: how wide an IV bucket
still counts as "the same vol regime", how much slack the back leg gets, how
stretched a z-score has to be before it is worth trading. Any one setting can be
made to look good. The question worth answering is whether the edge survives the
neighbourhood around it.

This runs the whole signal-and-backtest chain across a grid of those settings and
reports what changes. A strategy that only works at one corner of the grid is a
strategy that was fitted to the sample.
"""
from __future__ import annotations

import itertools

import pandas as pd

from .. import config as C
from ..backtest import engine, metrics
from . import fair_value


def grid(spreads: pd.DataFrame, chain: pd.DataFrame,
         iv_widths=(0.25, 0.5, 1.0, 1.5),
         back_tolerances=(0, 1, 2, 3),
         min_comparables=(5, 8, 12),
         z_entries=(1.0, 1.5, 2.0),
         verbose: bool = True) -> pd.DataFrame:
    """Run the chain over a parameter grid. One row per combination."""
    combos = list(itertools.product(iv_widths, back_tolerances,
                                    min_comparables, z_entries))
    rows = []

    for i, (iv_w, back_tol, min_n, z_in) in enumerate(combos):
        signals = fair_value.evaluate(
            spreads, iv_half_width=iv_w, back_dte_tolerance=back_tol,
            min_comparables=min_n, z_entry=z_in)

        tradeable = signals[signals["signal"].isin([fair_value.BUY,
                                                    fair_value.SELL])]
        result = engine.run(signals, chain, verbose=False)
        stats = metrics.summarise(result["equity"], result["trades"],
                                  C.INITIAL_CAPITAL)

        rows.append({
            "iv_half_width": iv_w,
            "back_dte_tolerance": back_tol,
            "min_comparables": min_n,
            "z_entry": z_in,
            "pct_with_enough_data": float(
                (signals["signal"] != fair_value.INSUFFICIENT).mean() * 100),
            "median_comparables": float(signals["n_comparables"].median()),
            "n_signals": int(len(tradeable)),
            "n_trades": stats.get("n_trades", 0),
            "total_return_pct": stats.get("total_return_pct"),
            "cagr_pct": stats.get("cagr_pct"),
            "sharpe": stats.get("sharpe"),
            "max_drawdown_pct": stats.get("max_drawdown_pct"),
            "win_rate_pct": stats.get("win_rate_pct"),
            "profit_factor": stats.get("profit_factor"),
        })

        if verbose and (i + 1) % 10 == 0:
            print(f"  sensitivity: {i + 1}/{len(combos)}", flush=True)

    return pd.DataFrame(rows)


def summarise_stability(results: pd.DataFrame) -> pd.DataFrame:
    """Marginal effect of each parameter, holding nothing else fixed.

    A parameter whose setting swings the Sharpe wildly is one the result depends
    on; one that barely moves it is not where the edge lives.
    """
    out = []
    for param in ("iv_half_width", "back_dte_tolerance", "min_comparables",
                  "z_entry"):
        grp = (results[results["n_trades"] > 0]
               .groupby(param)
               .agg(n_settings=("sharpe", "size"),
                    median_sharpe=("sharpe", "median"),
                    median_trades=("n_trades", "median"),
                    median_return=("total_return_pct", "median"))
               .reset_index()
               .rename(columns={param: "value"}))
        grp.insert(0, "parameter", param)
        out.append(grp)
    return pd.concat(out, ignore_index=True).round(2)
