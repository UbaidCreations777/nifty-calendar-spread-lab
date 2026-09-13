# NIFTY Calendar Spread Lab

An empirical fair-value model for NIFTY calendar spreads, and a backtest of
trading the deviations from it.

The question it answers is not "what does a model say this spread is worth". It
is the question a desk actually asks:

> The last time the front leg had 4 days to expiry, the back leg had 11, and ATM
> vol was 16 — what did this structure cost? It cost 0.53% of spot. It costs
> 0.37% today. So it is cheap.

Everything here is built around making that comparison honest.

---

## The signal

A calendar spread is a bet on the shape of the volatility term structure: sell
the near expiry, buy the further-out monthly, same strike, struck at the money.
What it costs — the net debit — depends on where vol is and how much time each
leg has left.

So a day is described by the state it was priced in:

| Component | Why it is in the state |
|---|---|
| Front leg days to expiry | The sold leg, where theta and gamma move fastest |
| Back leg days to expiry | The bought leg, within a ±2 day tolerance |
| Front ATM implied vol | The vol regime, bucketed to ±0.5 vol points |
| Option type | CE and PE carry different skew, so they get separate histories |

Today's debit is compared against every earlier day in the same state. If it sits
far below that distribution the spread is cheap and the signal is a buy; far
above and it is rich, so a sell. Distance is measured as a z-score, with the
percentile shown alongside it.

### Three things that keep it honest

**The debit is a percentage of spot, not points.** NIFTY moved from roughly
19,000 to 26,000 over the sample. Option premium scales with the level of the
index, so 100 points in 2023 and 100 points in 2026 are not the same trade.
Comparing raw points would read the index rally as a steady enrichment of the
spread and turn a drift into a signal.

**Comparables come only from the past.** For any day *t*, the distribution is
built from days strictly before *t*. Using the whole history would let a 2026
trade be judged against 2026 data it could not have seen — the most common way a
mean-reversion backtest comes out profitable and is untradeable.

**A minimum sample, and a minimum spread.** Below 8 matching days there is no
distribution, and the signal says so instead of producing a z-score from three
observations. If those comparables barely vary, the z-score is not trusted
either: dividing by a near-zero standard deviation turns rounding into conviction.

### The confirmation filter

Cheapness alone does not make a trade. A long calendar wants the leg it sells to
carry more implied vol than the leg it buys; when the term structure says the
opposite, the discount is more likely to be deserved than mispriced. Term
structure is a veto, never a trigger.

---

## What the data says

**The signal carries direction. It does not carry enough magnitude to pay for
trading it.** That is the result, and the project is built to show why rather
than to bury it.

### The signal, measured on its own

A backtest bundles the signal together with the structure's carry, the exit rule,
the sizing and the costs — so a bad number does not say which of the five was at
fault. `edge_study.py` separates the first one out: for every scored day it
measures what the *same legs* were worth three days later, grouped by how
stretched the reading was at the time.

| z-score at the time | What the spread did next | Signal was right |
|---|---|---|
| < −2 (very cheap) | widened | **70%** |
| −1.5 to −1 | widened | **67%** |
| −0.5 to 0.5 | flat | 52% |
| 1 to 1.5 | widened | 42% |
| 1.5 to 2 | widened | 35% |
| > 2 (very rich) | widened more | **29%** |

**The asymmetry is the finding.** Cheap spreads widen afterwards, which is what a
long calendar wants, and the call is right about two thirds of the time. Rich
spreads *also* widen — the debit drifts wider on its own as the front leg decays
faster than the back — so selling a rich calendar is betting against the
structure's own carry, and it is right about a third of the time.

That is a mechanical reason, not a statistical artefact, and it is why the
strategy trades the long side only.

### The backtest

Three years, NIFTY, ₹10 lakh, long side only, all costs charged:

| | |
|---|---|
| Trades | 20 |
| Win rate | 55% |
| Gross P&L | **₹2** |
| Costs | **₹3,521** |
| Net P&L | −₹3,519 (−0.35%) |
| Max drawdown | −1.09% |
| Sharpe | −0.21 |

Gross P&L is within rounding distance of zero. The edge exists in the direction
of the call and vanishes in its size: four orders a round trip, on a structure
whose whole move is a few tenths of a percent of spot, costs more than the
mispricing is worth.

Trading both directions is materially worse (−2.51%, Sharpe −0.93), which is the
asymmetry above showing up in P&L.

### What was tried and did not help

Capping the holding period at two, three and five days — motivated by the signal's
information decaying to nothing by day five — made the result slightly *worse* at
every setting. It is left off. A parameter that does not improve anything is a
parameter fitted to noise.

---

## What is in the box

```
src/
  config.py              every assumption, in one file
  data/nse_fo.py         NSE F&O bhavcopy download - both archive formats
  pricing/
    black_scholes.py     Black-76 price and Greeks, written out
    implied_vol.py       Newton-Raphson with a bisection fallback
    forward.py           per-expiry forward from put-call parity
  signal/
    chain.py             raw bhavcopy -> forwards -> implied vols
    calendar.py          the daily ATM calendar spread series
    fair_value.py        the signal
    edge_study.py        does the signal predict anything, before P&L
    sensitivity.py       the same result across a parameter grid
  backtest/
    engine.py            event loop, sizing, exits
    costs.py             STT, exchange, SEBI, stamp, GST, slippage
    metrics.py           Sharpe, Sortino, drawdown, cost drag
app.py                   Streamlit dashboard
tests/                   pricing validated against py_vollib; look-ahead tests
```

---

## Modelling choices worth defending

**Black-76 on a forward, not Black-Scholes on spot.** NIFTY's dividend stream is
irregular, so a spot-based model needs a dividend-yield assumption that is wrong
most days. The forward is taken from the option chain itself via put-call parity,
`F = K + e^{rT}(C − P)`, evaluated where `|C − P|` is smallest. Each expiry gets
its own forward — which matters for a structure spanning two of them.

**Settlement prices, not closes.** NSE marks every contract at end of day
including ones that never traded. A close on an untraded strike can be days
stale, and backtesting off stale closes manufactures profit that was never
available. Entries additionally require that the strike actually traded.

**Comparability is a distance, not a bucket.** The obvious rule — match the front
leg's days-to-expiry exactly — reaches eight comparable days on **0.1%** of the
sample. The reason is a market-structure change: NSE moved NIFTY's weekly expiry
from Thursday to Tuesday during 2025, which splits the days-to-expiry state space
into two eras and leaves the current one matching almost nothing in its own
history.

So neighbours are taken in a state space scaled so that one unit means "about as
different as one day on the front leg" — three days on a thirty-day back leg, or
half a vol point, are the rough equivalents. The nearest fifteen are used,
weighted by closeness, and a distance cap drops any day that is not really the
same trade even when that leaves too few to speak. Every verdict reports the
average distance behind it, so a weak match is visible rather than implied.

**Exit on reversion, or a day before front expiry.** The trade is opened against
a mispricing, so it closes when that mispricing is gone — whatever the position
is worth at the time. The time exit keeps the book clear of expiry-day pin risk
and of the higher STT charged on exercised contracts.

**Costs are charged on all four orders.** A calendar is a four-order round trip.
On a structure whose edge is a few tenths of a percent of spot, leaving costs out
does not shrink the profit, it usually reverses its sign.

---

## Known limitations

- **End-of-day data.** Bhavcopy is one price per contract per day, so intraday
  entry timing, the real bid-ask, and any intraday exit cannot be tested. Entries
  are assumed at the settlement price plus a slippage allowance.
- **Margin is approximated.** A calendar is margined as a spread, and the exact
  number needs the exchange's SPAN risk parameter file. A flat percentage of
  notional stands in for it, which will be wrong in a vol shock — exactly when it
  matters most.
- **One structure, one underlying.** ATM only, NIFTY only, monthly back leg only.
- **Roughly three years.** Long enough to cross several vol regimes; not long
  enough to call anything here a stable edge.

---

## Running it

```bash
pip install -r requirements.txt
python -m src.data.download          # NSE F&O bhavcopy, cached locally
python -m src.pipeline               # chain -> signal -> backtest
streamlit run app.py                 # dashboard
pytest -q                            # pricing validation + look-ahead tests
```

The data step downloads from NSE's public archive and caches each day to
`data/raw/`, so it only pays the download cost once.

---

Built by [Ubaid Shaikh](https://www.linkedin.com/in/ubaidshaikhwork) — derivatives
trader, proprietary desk. NISM Series VIII certified.
