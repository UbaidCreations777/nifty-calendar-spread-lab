"""End-to-end: raw bhavcopy -> chain -> spreads -> signal -> backtest.

Each stage caches its output to data/processed so the expensive parts (the IV
solve across roughly a million contracts) run once rather than on every launch of
the dashboard.

CLI: python -m src.pipeline [--rebuild]
"""
from __future__ import annotations

import sys

import pandas as pd

from . import config as C
from .backtest import engine, metrics
from .data import nse_fo
from .signal import calendar, chain as chain_mod, fair_value

CHAIN_PATH = C.DATA_PROCESSED / "chain.parquet"
SPREADS_PATH = C.DATA_PROCESSED / "spreads.parquet"
ALL_PAIRS_PATH = C.DATA_PROCESSED / "spreads_all.parquet"
SIGNAL_PATH = C.DATA_PROCESSED / "signals.parquet"
TRADES_PATH = C.DATA_PROCESSED / "trades.parquet"
EQUITY_PATH = C.DATA_PROCESSED / "equity.parquet"


def build_chain(rebuild: bool = False) -> pd.DataFrame:
    if CHAIN_PATH.exists() and not rebuild:
        return pd.read_parquet(CHAIN_PATH)
    print("building option chain (forwards + implied vols)...", flush=True)
    raw = nse_fo.load_cached()
    ch = chain_mod.build_chain(raw)
    C.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    ch.to_parquet(CHAIN_PATH, index=False)
    return ch


def build_spreads(chain: pd.DataFrame, rebuild: bool = False) -> pd.DataFrame:
    """The daily spread series depends only on the chain, never on the signal
    settings, so it is cached once and reused across every parameter change."""
    if SPREADS_PATH.exists() and not rebuild:
        return pd.read_parquet(SPREADS_PATH)
    print("building calendar spreads...", flush=True)
    spreads = calendar.build_spreads(chain)
    C.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    spreads.to_parquet(SPREADS_PATH, index=False)
    return spreads


def build_all_pairs(rebuild: bool = False) -> pd.DataFrame:
    """Every front/back expiry combination per day, for the explorer view.

    The backtest deliberately does not use this: it trades one defined structure,
    and choosing the most stretched reading out of every pair on offer would be
    picking the winner from a much larger draw. It exists so a structure can be
    looked up and priced against its own history.
    """
    if ALL_PAIRS_PATH.exists() and not rebuild:
        out = pd.read_parquet(ALL_PAIRS_PATH)
    else:
        chain = build_chain()
        print("building every expiry pair...", flush=True)
        out = calendar.build_spreads(chain, all_pairs=True)
        C.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
        out.to_parquet(ALL_PAIRS_PATH, index=False)
    for col in ("date", "front_expiry", "back_expiry"):
        out[col] = pd.to_datetime(out[col]).dt.date
    return out


def build_signals(chain: pd.DataFrame, rebuild: bool = False,
                  **kwargs) -> tuple[pd.DataFrame, pd.DataFrame]:
    spreads = build_spreads(chain, rebuild)
    if SIGNAL_PATH.exists() and not rebuild and not kwargs:
        return spreads, pd.read_parquet(SIGNAL_PATH)
    print("evaluating fair value against history...", flush=True)
    signals = fair_value.evaluate(spreads, **kwargs)
    if not kwargs:
        signals.to_parquet(SIGNAL_PATH, index=False)
    return spreads, signals


def run_all(rebuild: bool = False, save: bool = True, **signal_kwargs) -> dict:
    chain = build_chain(rebuild)
    spreads, signals = build_signals(chain, rebuild, **signal_kwargs)

    print("running backtest...", flush=True)
    result = engine.run(signals, chain)
    stats = metrics.summarise(result["equity"], result["trades"],
                              C.INITIAL_CAPITAL)

    if save:
        result["trades"].to_parquet(TRADES_PATH, index=False)
        result["equity"].to_parquet(EQUITY_PATH, index=False)

    return {"chain": chain, "spreads": spreads, "signals": signals,
            "trades": result["trades"], "equity": result["equity"],
            "stats": stats}


def main() -> int:
    rebuild = "--rebuild" in sys.argv
    out = run_all(rebuild=rebuild)

    print(metrics.format_report(out["stats"]))

    sig = out["signals"]
    counts = sig["signal"].value_counts()
    print("\n  signal distribution")
    for name, n in counts.items():
        print(f"    {name:<20} {n:>6}  ({n / len(sig) * 100:.1f}%)")

    by_reason = metrics.by_exit_reason(out["trades"])
    if not by_reason.empty:
        print("\n  exits\n" + by_reason.to_string(index=False))

    latest = fair_value.latest_view(sig)
    if not latest.empty:
        print(f"\n  latest reading ({latest['date'].iloc[0]})")
        show = ["option_type", "front_dte", "back_dte", "front_iv", "debit",
                "debit_pct", "fv_mean_pct", "z_score", "percentile",
                "n_comparables", "signal"]
        print(latest[show].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
