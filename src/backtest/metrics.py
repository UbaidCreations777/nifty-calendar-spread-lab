"""Performance statistics for the backtest.

Returns are computed on the equity curve rather than on trade P&L, so idle days -
the ones where the signal said nothing and capital sat unused - dilute the Sharpe
exactly as they would in a real book. A ratio computed only over days in a
position flatters a strategy that trades rarely.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def summarise(equity: pd.DataFrame, trades: pd.DataFrame,
              initial_capital: float) -> dict:
    eq = equity["equity"].to_numpy(dtype=float)
    if eq.size < 2:
        return {"error": "not enough equity points"}

    rets = np.diff(eq) / eq[:-1]
    total_return = eq[-1] / initial_capital - 1.0
    years = len(eq) / TRADING_DAYS
    cagr = (eq[-1] / initial_capital) ** (1 / years) - 1.0 if years > 0 else np.nan

    vol = rets.std(ddof=1) * np.sqrt(TRADING_DAYS) if rets.size > 1 else np.nan
    sharpe = (rets.mean() * TRADING_DAYS) / vol if vol and vol > 1e-12 else np.nan

    downside = rets[rets < 0]
    dvol = (downside.std(ddof=1) * np.sqrt(TRADING_DAYS)
            if downside.size > 1 else np.nan)
    sortino = (rets.mean() * TRADING_DAYS) / dvol if dvol and dvol > 1e-12 else np.nan

    peak = np.maximum.accumulate(eq)
    drawdown = eq / peak - 1.0
    max_dd = float(drawdown.min())

    out = {
        "initial_capital": initial_capital,
        "final_equity": float(eq[-1]),
        "total_return_pct": total_return * 100.0,
        "cagr_pct": cagr * 100.0,
        "annual_vol_pct": float(vol * 100.0) if np.isfinite(vol) else np.nan,
        "sharpe": float(sharpe) if np.isfinite(sharpe) else np.nan,
        "sortino": float(sortino) if np.isfinite(sortino) else np.nan,
        "max_drawdown_pct": max_dd * 100.0,
        "calmar": float(cagr / abs(max_dd)) if max_dd < -1e-12 else np.nan,
        "days": int(len(eq)),
        "exposure_pct": float(equity["in_position"].mean() * 100.0),
    }

    if trades.empty:
        out.update({"n_trades": 0})
        return out

    net = trades["net_pnl"].to_numpy(dtype=float)
    wins, losses = net[net > 0], net[net <= 0]
    out.update({
        "n_trades": int(len(trades)),
        "win_rate_pct": float((net > 0).mean() * 100.0),
        "avg_win": float(wins.mean()) if wins.size else 0.0,
        "avg_loss": float(losses.mean()) if losses.size else 0.0,
        "profit_factor": (float(wins.sum() / abs(losses.sum()))
                          if losses.size and losses.sum() != 0 else np.nan),
        "gross_pnl": float(trades["gross_pnl"].sum()),
        "total_costs": float(trades["cost"].sum()),
        "net_pnl": float(net.sum()),
        # Meaningless when gross P&L is near zero - the ratio explodes and says
        # nothing. Costs relative to the capital at work is the honest reading
        # there, and it is reported alongside.
        "cost_drag_pct_of_gross": (
            float(trades["cost"].sum() / abs(trades["gross_pnl"].sum()) * 100.0)
            if abs(trades["gross_pnl"].sum()) > trades["cost"].sum() * 0.1
            else np.nan),
        "costs_pct_of_capital": float(trades["cost"].sum()
                                      / initial_capital * 100.0),
        "avg_days_held": float(trades["days_held"].mean()),
        "best_trade": float(net.max()),
        "worst_trade": float(net.min()),
    })
    return out


def by_exit_reason(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    return (trades.groupby("exit_reason")
            .agg(n=("net_pnl", "size"), total_pnl=("net_pnl", "sum"),
                 avg_pnl=("net_pnl", "mean"), win_rate=("net_pnl", lambda s: (s > 0).mean() * 100))
            .round(2).reset_index())


def format_report(stats: dict) -> str:
    lines = ["", "=" * 58, "  BACKTEST SUMMARY", "=" * 58]
    order = [
        ("Initial capital", "initial_capital", "Rs {:,.0f}"),
        ("Final equity", "final_equity", "Rs {:,.0f}"),
        ("Total return", "total_return_pct", "{:.2f}%"),
        ("CAGR", "cagr_pct", "{:.2f}%"),
        ("Annualised vol", "annual_vol_pct", "{:.2f}%"),
        ("Sharpe", "sharpe", "{:.2f}"),
        ("Sortino", "sortino", "{:.2f}"),
        ("Max drawdown", "max_drawdown_pct", "{:.2f}%"),
        ("Calmar", "calmar", "{:.2f}"),
        ("Days in sample", "days", "{:,}"),
        ("Time in position", "exposure_pct", "{:.1f}%"),
        ("", None, None),
        ("Trades", "n_trades", "{:,}"),
        ("Win rate", "win_rate_pct", "{:.1f}%"),
        ("Profit factor", "profit_factor", "{:.2f}"),
        ("Avg holding period", "avg_days_held", "{:.1f} days"),
        ("Gross P&L", "gross_pnl", "Rs {:,.0f}"),
        ("Total costs", "total_costs", "Rs {:,.0f}"),
        ("Costs as % of gross", "cost_drag_pct_of_gross", "{:.1f}%"),
        ("Costs as % of capital", "costs_pct_of_capital", "{:.2f}%"),
        ("Net P&L", "net_pnl", "Rs {:,.0f}"),
        ("Best trade", "best_trade", "Rs {:,.0f}"),
        ("Worst trade", "worst_trade", "Rs {:,.0f}"),
    ]
    for label, key, fmt in order:
        if key is None:
            lines.append("-" * 58)
            continue
        if key not in stats:
            continue
        val = stats[key]
        text = fmt.format(val) if isinstance(val, (int, float)) and np.isfinite(val) else "n/a"
        lines.append(f"  {label:<24} {text:>28}")
    lines.append("=" * 58)
    return "\n".join(lines)
