"""Streamlit dashboard for the NIFTY calendar spread lab.

Four views, in the order a desk would use them: what the signal says today, the
volatility surface it is derived from, how the rule performed historically, and
how much of that performance depends on where the parameters were set.

Run: streamlit run app.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import config as C
from src.backtest import metrics
from src.pipeline import run_all
from src.signal import chain as chain_mod, fair_value as fv

st.set_page_config(page_title="NIFTY Calendar Spread Lab",
                   page_icon="📐", layout="wide")

BUY_COLOUR = "#1a7f5a"
SELL_COLOUR = "#b3402f"
NEUTRAL = "#8a8f98"


@st.cache_data(show_spinner="Loading pipeline…")
def load(method: str, k: int, max_dist: float, iv_width: float, back_tol: int,
         min_n: int, z_entry: float, use_filter: bool, allow_short: bool):
    return run_all(rebuild=False, save=False, allow_short=allow_short,
                   method=method, k=k,
                   max_distance=max_dist, iv_half_width=iv_width,
                   back_dte_tolerance=back_tol, min_comparables=min_n,
                   z_entry=z_entry, use_term_structure_filter=use_filter)


# ------------------------------------------------------------------- sidebar
st.sidebar.title("Matching rule")
st.sidebar.caption(
    "How today is matched to the days in history that resemble it.")

method = st.sidebar.radio(
    "Method", [fv.KNN, fv.BUCKET],
    format_func=lambda m: ("Nearest neighbours" if m == fv.KNN
                           else "Fixed buckets"),
    help=("Fixed buckets need an exact match on the front leg's days to expiry. "
          "That rule reaches eight comparables on 0.1% of days here, because "
          "NSE moved NIFTY's weekly expiry from Thursday to Tuesday during 2025 "
          "and split the state space in two."))

if method == fv.KNN:
    k_neighbours = st.sidebar.slider("Neighbours (k)", 5, 40, int(C.KNN_K))
    max_dist = st.sidebar.slider("Max state distance", 0.5, 6.0,
                                 float(C.MAX_STATE_DISTANCE), 0.5,
                                 help="1.0 ≈ one day's difference on the front leg.")
    iv_width, back_tol = C.IV_BUCKET_HALF_WIDTH, C.BACK_DTE_TOLERANCE
else:
    iv_width = st.sidebar.slider("IV bucket (± vol points)", 0.25, 2.0,
                                 float(C.IV_BUCKET_HALF_WIDTH), 0.25)
    back_tol = st.sidebar.slider("Back leg DTE tolerance (± days)", 0, 5,
                                 int(C.BACK_DTE_TOLERANCE))
    k_neighbours, max_dist = C.KNN_K, C.MAX_STATE_DISTANCE

min_n = st.sidebar.slider("Minimum comparables", 3, 20,
                          int(C.MIN_COMPARABLES))
z_entry = st.sidebar.slider("Entry |z|", 0.5, 3.0, float(C.Z_ENTRY), 0.25)
use_filter = st.sidebar.checkbox("Term-structure confirmation", value=True)
allow_short = st.sidebar.checkbox(
    "Backtest the sell side too", value=C.ALLOW_SHORT,
    help=("Sell signals are always shown. This decides whether the backtest "
          "also trades them — off by default, because on this sample the sell "
          "side was right about a third of the time."))

st.sidebar.markdown("---")
st.sidebar.caption(
    f"Costs modelled: STT {C.STT_SELL_PREMIUM:.2%} on sell premium, exchange "
    f"{C.EXCHANGE_TXN_PCT:.4%}, GST {C.GST_PCT:.0%}, "
    f"{C.SLIPPAGE_TICKS} ticks slippage per leg.")
st.sidebar.markdown("---")
st.sidebar.caption(
    "**Not investment advice.** A personal research project. The buy and sell "
    "labels are a model's output being tested, not a recommendation — and the "
    "finding is that the rule does not clear its costs. No SEBI research "
    "analyst or investment adviser registration. Derivatives can lose "
    "substantial money.")

data = load(method, k_neighbours, max_dist, iv_width, back_tol, min_n,
            z_entry, use_filter, allow_short)
signals, chain, trades, equity = (data["signals"], data["chain"],
                                  data["trades"], data["equity"])
stats = metrics.summarise(equity, trades, C.INITIAL_CAPITAL)

st.title("NIFTY Calendar Spread Lab")
st.caption(
    "Is today's ATM calendar cheap or rich against the days in history that "
    "looked like today — same front and back days-to-expiry, same ATM vol "
    "bucket — with the debit measured as a percentage of spot so a rising index "
    "does not read as a richer spread.")

tab_today, tab_surface, tab_edge, tab_backtest, tab_robust = st.tabs(
    ["Today's signal", "Volatility surface", "Does the signal work?",
     "Backtest", "Robustness"])

@st.cache_data(show_spinner="Building every expiry pair…")
def load_all_pairs():
    """Each day's full grid of front/back combinations, for the explorer."""
    from src.pipeline import build_all_pairs
    return build_all_pairs()


VERDICT_WORDS = {
    fv.BUY: ("BUY THE SPREAD", "trading below its own history — pay the debit"),
    fv.SELL: ("SELL THE SPREAD", "trading above its own history — take the credit"),
    fv.FLAT: ("NO TRADE", "inside its normal range"),
    fv.INSUFFICIENT: ("NO CALL", "not enough comparable history"),
}


def render_banner(row: dict):
    """The verdict, stated before anything else on the page."""
    signal = row.get("signal", fv.INSUFFICIENT)
    raw = row.get("raw_signal", signal)
    word, why = VERDICT_WORDS[signal]
    colour = {fv.BUY: BUY_COLOUR, fv.SELL: SELL_COLOUR}.get(signal, NEUTRAL)

    z = row.get("z_score")
    detail = (f"z = {z:+.2f} &nbsp;·&nbsp; {row['percentile']:.0f}th percentile"
              if z is not None and np.isfinite(z) else why)

    st.markdown(
        f"<div style='padding:20px 24px;border-radius:10px;background:{colour};"
        f"color:white;margin-bottom:14px'>"
        f"<div style='font-size:2rem;font-weight:700;line-height:1.1'>{word}</div>"
        f"<div style='font-size:1rem;opacity:.92;margin-top:6px'>{why}"
        f" &nbsp;·&nbsp; {detail}</div></div>",
        unsafe_allow_html=True)

    # The term structure only ever vetoes. When it has, say so plainly rather
    # than showing a bare "no trade" the reader cannot account for.
    if raw != signal and raw in (fv.BUY, fv.SELL):
        side = "cheap" if raw == fv.BUY else "rich"
        wanted = "the near leg carrying more vol" if raw == fv.BUY \
            else "the far leg carrying more vol"
        st.info(
            f"The z-score alone says **{raw.upper()}** — the spread is {side} "
            f"against comparable days. The term-structure check overruled it: "
            f"that trade wants {wanted}, and right now the curve is the other "
            f"way round ({row['term_structure']:+.2f} vol points). Turn the "
            "confirmation off in the sidebar to trade the raw reading.")

    if fv.SELL in (signal, raw):
        st.warning(
            "**On the record, this side has been the weak one.** Measured over "
            "the sample, spreads that looked rich went on to widen further "
            "rather than narrow — the sell call was right about 29–35% of the "
            "time against 67–70% for the buy call. A calendar's debit drifts "
            "wider on its own as the front leg decays, so selling it fights "
            "that drift. The backtest therefore trades the long side only.")


def render_verdict(row: dict, source: pd.DataFrame, as_of):
    """One structure's card and its comparable-history chart."""
    render_banner(row)
    left, right = st.columns([1, 1.6])

    with left:
        st.markdown(f"### {row['option_type']} calendar")
        st.markdown(
            f"**Sell** {row['front_expiry']} ({row['front_dte']}d) · "
            f"**Buy** {row['back_expiry']} ({row['back_dte']}d)  \n"
            f"Strike {row['strike']:,.0f} · spot {row['spot']:,.0f}")

        c1, c2 = st.columns(2)
        c1.metric("Debit today", f"{row['debit']:,.1f} pts",
                  f"{row['debit_pct']:.3f}% of spot")
        if row["signal"] == fv.INSUFFICIENT:
            c2.metric("Verdict", "No call",
                      f"{int(row['n_comparables'])} comparables")
            st.info(
                f"Only {int(row['n_comparables'])} comparable days — below "
                f"the {min_n} needed. The honest answer is no signal, not a "
                "z-score computed from a handful of observations.")
            return

        c2.metric("Fair value", f"{row['fv_mean_pct']:.3f}%",
                  f"{row['cheapness_pct']:+.1f}% vs history")

        st.caption(
            f"Front ATM IV {row['front_iv']:.1f} · back {row['back_iv']:.1f} "
            f"· term structure {row['term_structure']:+.2f} vol points · "
            f"{int(row['n_comparables'])} comparable days")

    with right:
        hist = fv.comparables(source, row["option_type"], as_of,
                              method=method, k=k_neighbours,
                              max_distance=max_dist,
                              iv_half_width=iv_width,
                              back_dte_tolerance=back_tol,
                              front_dte=int(row["front_dte"]),
                              back_dte=int(row["back_dte"]))
        if hist.empty:
            st.info("No comparable history to plot.")
            return

        fig = px.histogram(hist, x="debit_pct", nbins=15,
                           title="What this structure cost on comparable days",
                           labels={"debit_pct": "Debit (% of spot)"})
        fig.update_traces(marker_color="#c9ced6")
        fig.add_vline(x=row["debit_pct"], line_color=BUY_COLOUR
                      if row["signal"] == fv.BUY else SELL_COLOUR,
                      line_width=3, annotation_text="today")
        # Mid grey rather than near-black: the page renders in whichever theme
        # the viewer has set, and a dark annotation disappears against a dark
        # background.
        fig.add_vline(x=row["fv_mean_pct"], line_dash="dash",
                      line_color="#8a8f98", annotation_text="mean",
                      annotation_font_color="#8a8f98")
        fig.update_layout(height=340, showlegend=False,
                          margin=dict(t=48, b=8, l=8, r=8))
        st.plotly_chart(fig, use_container_width=True)
        if method == fv.KNN:
            st.caption(
                f"The {len(hist)} nearest historical days in state space — "
                f"average distance {hist['state_distance'].mean():.2f} "
                "(1.0 ≈ one day's difference on the front leg).")
        else:
            st.caption(f"{len(hist)} historical days inside the bucket.")


# ---------------------------------------------------------------- today's tab
with tab_today:
    latest = fv.latest_view(signals)
    as_of = latest["date"].iloc[0]
    st.subheader(f"Reading for {as_of}")

    mode = st.radio(
        "Which structure?",
        ["The strategy's pair", "Pick the expiries yourself"],
        horizontal=True,
        help=("The strategy always sells the nearest expiry and buys the "
              "nearest monthly beyond it. Any other pair can still be priced "
              "against its own history — it is simply not what the backtest "
              "trades."))

    if mode == "The strategy's pair":
        for _, row in latest.iterrows():
            render_verdict(row.to_dict(), data["spreads"], as_of)
    else:
        grid = load_all_pairs()
        today_grid = grid[grid["date"] == as_of]

        if today_grid.empty:
            st.info(f"No expiry grid available for {as_of}.")
        else:
            c1, c2, c3 = st.columns(3)
            opt = c1.selectbox("Option type", ["PE", "CE"])
            pool = today_grid[today_grid["option_type"] == opt]

            fronts = sorted(pool["front_expiry"].unique())
            front = c2.selectbox(
                "Sell (front leg)", fronts,
                format_func=lambda e: f"{e}  ({(pool[pool.front_expiry == e]['front_dte'].iloc[0])}d)")

            backs = sorted(pool[pool["front_expiry"] == front]["back_expiry"].unique())
            if not backs:
                st.info("No back leg available beyond that expiry.")
            else:
                sub = pool[pool["front_expiry"] == front]
                back = c3.selectbox(
                    "Buy (back leg)", backs,
                    format_func=lambda e: f"{e}  ({(sub[sub.back_expiry == e]['back_dte'].iloc[0])}d)")

                chosen = sub[sub["back_expiry"] == back].iloc[0]
                verdict = fv.evaluate_one(
                    grid, opt, as_of,
                    front_dte=int(chosen["front_dte"]),
                    back_dte=int(chosen["back_dte"]),
                    z_entry=z_entry,
                    use_term_structure_filter=use_filter,
                    min_comparables=min_n,
                    method=method, k=k_neighbours, max_distance=max_dist,
                    iv_half_width=iv_width, back_dte_tolerance=back_tol)
                render_verdict(verdict, grid, as_of)

    st.markdown("---")
    st.subheader("Signal history")
    recent = signals[signals["signal"] != fv.INSUFFICIENT].tail(400)
    if recent.empty:
        st.info("No scored days at these settings.")
    else:
        fig = go.Figure()
        for opt, dash in (("CE", "solid"), ("PE", "dot")):
            part = recent[recent["option_type"] == opt]
            fig.add_trace(go.Scatter(x=part["date"], y=part["z_score"],
                                     name=opt, mode="lines",
                                     line=dict(dash=dash)))
        fig.add_hline(y=z_entry, line_dash="dash", line_color=SELL_COLOUR)
        fig.add_hline(y=-z_entry, line_dash="dash", line_color=BUY_COLOUR)
        fig.add_hline(y=0, line_color="#ccc")
        fig.update_layout(height=320, yaxis_title="z-score vs comparable history",
                          margin=dict(t=20, b=8, l=8, r=8))
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------- surface tab
with tab_surface:
    st.subheader("Implied volatility surface")
    st.caption(
        "Every IV here is solved from the settlement price against the forward "
        "implied by that expiry's own put-call parity, so each expiry carries "
        "its own carry rather than a shared dividend assumption.")

    dates = sorted(chain["date"].unique())
    pick = st.select_slider("Trading day", options=dates, value=dates[-1])
    day = chain[(chain["date"] == pick) & chain["iv"].notna()]

    c1, c2 = st.columns([1.4, 1])
    with c1:
        surf = day[day["option_type"] == st.radio(
            "Option type", ["CE", "PE"], horizontal=True)]
        if not surf.empty:
            fig = px.scatter(surf, x="strike", y="iv", color="dte",
                             labels={"iv": "Implied vol", "dte": "Days to expiry"},
                             color_continuous_scale="Viridis",
                             title=f"Skew by expiry — {pick}")
            fig.update_traces(marker=dict(size=6))
            fig.add_vline(x=float(surf["spot"].iloc[0]), line_dash="dash",
                          line_color="#666", annotation_text="spot")
            fig.update_layout(height=420, margin=dict(t=48, b=8, l=8, r=8))
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        atm = chain_mod.atm_iv(day)
        if not atm.empty:
            fig = px.line(atm.sort_values("dte"), x="dte", y="atm_iv",
                          color="option_type", markers=True,
                          labels={"dte": "Days to expiry", "atm_iv": "ATM IV"},
                          title="Term structure")
            fig.update_layout(height=420, margin=dict(t=48, b=8, l=8, r=8))
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "A downward slope means the near expiry carries more vol than "
                "the far one, which is the condition a long calendar wants.")

# ------------------------------------------------------------------- edge tab
with tab_edge:
    st.subheader("Does the signal predict anything?")
    st.caption(
        "A backtest bundles the signal together with the structure's carry, the "
        "exit rule, the sizing and the costs — so a bad result does not say "
        "which of the five was at fault. This measures the signal on its own: "
        "what the same legs were worth a few days later, grouped by how "
        "stretched the reading was at the time.")

    @st.cache_data(show_spinner="Measuring forward moves…")
    def _moves(_sig, _chain):
        from src.signal import edge_study
        return edge_study.forward_moves(_sig, _chain)

    moves = _moves(signals, chain)
    if moves.empty:
        st.info("No scored days to measure.")
    else:
        from src.signal import edge_study

        horizon = st.radio("Horizon (trading days)", [1, 2, 3, 5],
                           index=2, horizontal=True)
        buckets = edge_study.by_z_bucket(moves, horizon)

        fig = px.bar(buckets, x="bucket", y="hit_rate",
                     labels={"bucket": "z-score at the time",
                             "hit_rate": "Signal was right (%)"},
                     title=f"Hit rate by signal strength — {horizon} days later")
        fig.add_hline(y=50, line_dash="dash", line_color="#666",
                      annotation_text="coin flip")
        fig.update_traces(marker_color=[
            BUY_COLOUR if "-" in str(b) else SELL_COLOUR
            for b in buckets["bucket"]])
        fig.update_layout(height=360, margin=dict(t=48, b=8, l=8, r=8))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            "**The asymmetry is the finding.** Cheap spreads (left, green) "
            "subsequently widen, which is what a long calendar wants, and the "
            "signal is right around two thirds of the time. Rich spreads "
            "(right, red) also widen — the debit drifts wider on its own as the "
            "front leg decays faster than the back — so selling a rich calendar "
            "fights the structure's own carry and is right around a third of "
            "the time. That is why the strategy trades the long side only.")

        st.dataframe(buckets, hide_index=True, use_container_width=True)

        st.markdown("**Signal strength against everything else**")
        st.dataframe(edge_study.decompose(moves, horizon, z_entry),
                     hide_index=True, use_container_width=True)


# --------------------------------------------------------------- backtest tab
with tab_backtest:
    st.subheader("Backtest")
    if trades.empty:
        st.warning(
            "No trades at these settings. Widen the IV bucket or lower the "
            "minimum comparables in the sidebar — with a strict rule there are "
            "simply not enough matching days to form a distribution.")
    else:
        cols = st.columns(6)
        cols[0].metric("Net P&L", f"₹{stats['net_pnl']:,.0f}")
        cols[1].metric("Total return", f"{stats['total_return_pct']:.1f}%")
        cols[2].metric("Sharpe", f"{stats['sharpe']:.2f}")
        cols[3].metric("Max drawdown", f"{stats['max_drawdown_pct']:.1f}%")
        cols[4].metric("Win rate", f"{stats['win_rate_pct']:.0f}%",
                       f"{stats['n_trades']} trades")
        cols[5].metric("Costs", f"₹{stats['total_costs']:,.0f}",
                       f"{stats['costs_pct_of_capital']:.2f}% of capital")

        if stats["gross_pnl"] <= stats["total_costs"]:
            st.warning(
                f"**The honest result.** Gross P&L is ₹{stats['gross_pnl']:,.0f} "
                f"and costs are ₹{stats['total_costs']:,.0f}, so the strategy "
                "does not clear its own transaction costs. The signal carries "
                "direction — see the previous tab — but not enough magnitude to "
                "pay for four orders a round trip.")

        with st.expander(
                "Holding it intraday instead — and why the version that worked "
                "was not real"):
            st.markdown(
                "A calendar is short gamma, so the overnight gaps were the "
                "obvious suspect. Closing before the bell **did** turn the "
                "result positive — but only while the entry price was assumed "
                "rather than observed.\n\n"
                "| | Overnight | Intraday, *assumed* fills | Intraday, **real minute** fills |\n"
                "|---|---|---|---|\n"
                "| Return | −0.35% | +1.17% | **−0.26%** |\n"
                "| Sharpe | −0.21 | 0.94 | **−0.34** |\n"
                "| Win rate | 55% | 74% | **37%** |\n"
                "| Gross P&L | ₹2 | ₹15,123 | **₹837** |\n\n"
                "The middle column priced each leg at its own first trade of the "
                "day. Those prints are minutes apart, so that spread never "
                "existed as a quote — it was favourable by a median of 4.35 "
                "points against a median real move of 6.85. Re-pricing both legs "
                "at the *same instant* removed the entire edge.\n\n"
                "It is not a question of picking the right clock either: across "
                "28 entry/exit time combinations, 9 are positive and the median "
                "is −0.17%.\n\n"
                "The minute data was checked against NSE before being trusted — "
                "open, high and low match bhavcopy on 99.7% of contract-days, "
                "and close matches once compared on NSE's own last-half-hour "
                "VWAP convention rather than the last trade.")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=equity["date"], y=equity["equity"],
                                 name="Equity", line=dict(color="#1f4e79")))
        fig.add_hline(y=C.INITIAL_CAPITAL, line_dash="dash", line_color="#999")
        fig.update_layout(height=360, yaxis_title="Equity (₹)",
                          margin=dict(t=20, b=8, l=8, r=8))
        st.plotly_chart(fig, use_container_width=True)

        left, right = st.columns(2)
        with left:
            st.markdown("**P&L by exit reason**")
            st.dataframe(metrics.by_exit_reason(trades), hide_index=True,
                         use_container_width=True)
        with right:
            st.markdown("**Gross vs net**")
            bar = pd.DataFrame({
                "component": ["Gross P&L", "Costs", "Net P&L"],
                "value": [stats["gross_pnl"], -stats["total_costs"],
                          stats["net_pnl"]]})
            fig = px.bar(bar, x="component", y="value", text_auto=".2s")
            fig.update_layout(height=280, showlegend=False,
                              margin=dict(t=20, b=8, l=8, r=8))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Trades**")
        show = ["entry_date", "exit_date", "option_type", "strike", "is_long",
                "entry_z", "entry_debit", "exit_debit", "n_lots", "days_held",
                "gross_pnl", "cost", "net_pnl", "exit_reason"]
        st.dataframe(trades[show].sort_values("entry_date", ascending=False),
                     hide_index=True, use_container_width=True, height=320)

# ------------------------------------------------------------- robustness tab
with tab_robust:
    st.subheader("Does the edge survive its own parameters?")
    st.caption(
        "Any single setting can be made to look good. What matters is whether "
        "the result holds in the neighbourhood around it.")

    if st.button("Run the parameter grid (takes a minute)"):
        from src.signal import sensitivity
        with st.spinner("Running…"):
            grid = sensitivity.grid(data["spreads"], chain, verbose=False)
        st.session_state["grid"] = grid

    grid = st.session_state.get("grid")
    if grid is None:
        st.info("Run the grid to populate this view.")
    else:
        traded = grid[grid["n_trades"] > 0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Settings tested", len(grid))
        c2.metric("Settings that traded", len(traded))
        if not traded.empty:
            c3.metric("Median Sharpe", f"{traded['sharpe'].median():.2f}")

        if not traded.empty:
            fig = px.box(traded, x="z_entry", y="sharpe", points="all",
                         labels={"z_entry": "Entry |z|", "sharpe": "Sharpe"},
                         title="Sharpe across every other setting")
            fig.update_layout(height=340, margin=dict(t=48, b=8, l=8, r=8))
            st.plotly_chart(fig, use_container_width=True)

            from src.signal import sensitivity
            st.markdown("**Marginal effect of each parameter**")
            st.dataframe(sensitivity.summarise_stability(grid),
                         hide_index=True, use_container_width=True)

        st.dataframe(grid.round(2), hide_index=True, use_container_width=True,
                     height=320)
