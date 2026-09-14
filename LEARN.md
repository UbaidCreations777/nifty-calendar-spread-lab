# LEARN.md — defending this project in an interview

Every design decision in this repo, the reasoning behind it, and the question an
interviewer is likely to ask about it. Read this before the interview; it is
written so you can answer from your own understanding rather than from memory.

The theme that runs through all of it: **an interviewer is not testing whether
you can write Python. They are testing whether you know what would make the
number wrong.** Almost every answer below is a version of that.

---

## 1. The one-line pitch

> I built an empirical fair-value model for NIFTY calendar spreads. Instead of
> asking what a pricing model says the spread is worth, it asks what this exact
> structure actually cost on every historical day that looked like today — same
> front and back days-to-expiry, same ATM vol, same gap between the two legs'
> forwards — and trades the deviation. It backtests on three years of NSE F&O
> bhavcopy with the full Indian cost stack, and the signal only ever sees data
> from before the day it is trading.

If they only remember one thing, make it the last clause. Point-in-time
discipline is the thing that separates a real backtest from a plausible-looking
one.

---

## 2. The maths you must be able to derive

### Black-76

Pricing an option on a forward `F` rather than a spot `S`:

```
d1 = [ln(F/K) + (σ²/2)T] / (σ√T)
d2 = d1 − σ√T

Call = e^{−rT} [F·N(d1) − K·N(d2)]
Put  = e^{−rT} [K·N(−d2) − F·N(−d1)]
```

**Why the forward and not spot?** NIFTY pays an irregular dividend stream. A
spot-based Black-Scholes needs a dividend yield `q`, and any number you pick is
wrong on most days. Pricing off the forward removes that free parameter — the
market tells you the carry instead of you assuming it.

**Where does the forward come from?** Put-call parity, from the chain itself:

```
C − P = e^{−rT}(F − K)    ⟹    F = K + e^{rT}(C − P)
```

Evaluated at the strike where `|C − P|` is smallest — i.e. nearest the money,
where both quotes are real rather than stale.

> **Q: Why not just use the futures price?**
> NIFTY has monthly futures but weekly options. There is no matching future for a
> weekly expiry, so parity gives a per-expiry forward where the futures market
> gives none.

### The Greeks, and what each one means for a calendar

| Greek | Formula (Black-76) | What it does to a long calendar |
|---|---|---|
| Delta | `e^{−rT}N(d1)` (call) | ≈ 0 at entry — same strike, both legs |
| Gamma | `e^{−rT}n(d1) / (Fσ√T)` | **Negative.** Front leg's gamma dominates; a big spot move hurts |
| Vega | `e^{−rT}F·n(d1)√T` | **Positive.** Back leg has more time, so more vega |
| Theta | `−F e^{−rT}n(d1)σ/(2√T) + r·e^{−rT}[…]` | **Positive.** Front decays faster than back |

**The single most important sentence about calendar spreads:** a long calendar is
*short gamma and long vega*. You earn the difference in time decay and you want
volatility to rise — but a sharp move in the underlying works against you,
because the leg you sold is the one with the most gamma.

> **Q: Your calendar is delta-neutral at entry. Is it still neutral tomorrow?**
> No. Both legs are struck at the same price but have different gammas, so as
> spot moves the front leg's delta changes faster. The position picks up delta in
> the direction that hurts — that is what being short gamma means.

> **Q: Vega is positive, so you want vol up. But you sold the front leg, which is
> the highest-vol leg. Contradiction?**
> No — those are different exposures. Vega is about the *level* of vol, and the
> back leg's longer maturity gives it more. The term structure view is about the
> *shape*: you want the vol you sold to fall relative to the vol you bought. You
> can be long vol and still want the curve to steepen.

### Implied volatility, and why Newton alone is not enough

Newton-Raphson on `f(σ) = BS(σ) − market`, with `f'(σ) = vega`:

```
σ_{n+1} = σ_n − [BS(σ_n) − market] / vega(σ_n)
```

Seeded with the Brenner-Subrahmanyam approximation `σ ≈ √(2π/T) · price/F`, which
is exact at the money and close enough elsewhere to converge in a few steps.

**Why the bisection fallback?** Vega collapses toward zero for deep in- and
out-of-the-money options. Dividing by a near-zero derivative throws the next
iterate far from the root, and Newton diverges. Bisection cannot diverge — it
only ever halves a bracket already known to contain the root — so it catches what
Newton walks away from. Newton for speed, bisection for guarantees.

> **Q: What if there is no solution?**
> Then there is no implied vol, and the code returns NaN rather than a number.
> Three cases: the price is below intrinsic (arbitrage violation, usually stale
> data), above the forward (same), or below NSE's 0.05 tick — at which point any
> vol over a wide range reproduces it and the "solution" is noise. See
> `test_rejects_arbitrage_violating_quotes` and `test_rejects_sub_tick_prices`.

---

## 3. The data decisions — where interviewers actually probe

### Settlement price, not close

NSE marks every contract at end of day, including ones that never traded. On an
untraded strike, `close` is whatever it last printed, which can be days old.

**Why it matters:** backtesting a spread off stale closes manufactures profit
that was never available. You "buy" at a price nobody was showing.

Entries carry a second condition: the strike must have actually traded that day.
Settlement is fine for *valuing* a position; it is not evidence you could have
*entered* one.

> **Q: How do you know your fills are realistic?**
> I do not, fully — that is a stated limitation. Bhavcopy is one price per
> contract per day, so there is no bid-ask in the data. I charge two ticks of
> slippage per leg, on all four orders, as a stand-in. With real quote data the
> honest test would be entering at the offer and exiting at the bid.

### Two archive formats

NSE moved the F&O bhavcopy to the UDiFF format on 2024-07-08. Anything earlier
only exists in the legacy layout. The downloader tries both per date and reports
which answered, so a moved path shows up as a named failure rather than a
silently empty folder.

The UDiFF files carry two fields the legacy ones do not: the underlying price and
the contract lot size. **Lot size matters** — NSE has revised NIFTY's several
times over this sample. Reading it from the file rather than hardcoding it is why
the position sizing is right across the whole period.

> **Q: What is the lot size of NIFTY?**
> It has changed repeatedly — which is exactly why the code reads it per contract
> from the exchange file instead of assuming one. Hardcoding it would silently
> misprice every trade on one side of the revision date.

---

## 4. The three failure modes this project is built to avoid

These are the answers that will separate you from a candidate who ran a backtest
and got a nice curve.

### Look-ahead bias

For any day *t*, comparables are drawn from days strictly before *t*. Nothing
else. There is a test that proves it: two synthetic series share their first 20
days and differ wildly afterwards, and every verdict in the shared prefix must be
identical. If the signal could see forward, they would not be.

> **Q: Why does that matter so much for this strategy in particular?**
> Because the signal is a distance from a historical mean. If that mean is
> computed over the full sample, then on any given day it already contains the
> future — and "the spread is below its mean" becomes partly a statement about
> where the spread is *going*. You would get a beautiful equity curve that cannot
> be traded.

### Normalisation for the level of the index

NIFTY went from roughly 19,000 to 26,000 across the sample. Premium scales with
the level of the underlying, so the same trade costs more points at a higher
index. Storing the debit as a percentage of spot is what makes 2023 and 2026
comparable.

> **Q: What would have happened without it?**
> Raw points drift upward as the index rises. Every late-sample spread reads as
> expensive against an early-sample history, so the model generates a stream of
> sell signals that reflect the index rally rather than anything about the
> spread. You would be backtesting the direction of NIFTY and calling it vol
> relative value.

### A distribution that is not a distribution

Two guards. Below 8 comparable days the signal reports insufficient data rather
than computing a z-score from a handful of points. And if those comparables
barely vary, the z-score is not trusted at all: `z = (x − μ)/σ` with σ near zero
turns a rounding-sized difference into enormous conviction.

> **Q: Isn't refusing to trade a wasted opportunity?**
> Refusing to trade is a position. The alternative is a signal that fires most
> confidently exactly where it has the least evidence, which is worse than no
> signal.

---

## 5. Questions about the strategy itself

> **Q: Why does a calendar's debit drift wider on its own?**
> Because the front leg decays faster than the back. The debit is
> back minus front, both legs lose time value, and the near-dated one loses it
> much more quickly — so the difference widens with nothing happening in the
> market at all. That is the same thing as saying a long calendar is positive
> theta, viewed from the price rather than the Greek. It matters here because a
> z-score computed on the debit *level* does not know about that drift, which is
> why the rich side of the signal is inverted rather than merely weak.

> **Q: Your edge is that the spread is cheap relative to history. What if it is
> cheap for a reason?**
> Often it is — and that is what the term-structure filter is for. If the spread
> is cheap *and* the front leg carries more vol than the back, the structure is
> discounted while the condition it wants is present. If it is cheap while the
> curve is the wrong way round, the discount is probably deserved and the trade
> is stood down. The filter is a veto, never a trigger.

> **Q: What is the real risk in this trade?**
> Short gamma. The position is close to delta-neutral and earns the decay
> differential, but a sharp move in NIFTY — a gap, an event day — hurts the leg I
> sold most. The second risk is an IV crush in the back leg: I am long vega
> there, so if longer-dated vol falls the structure loses even when the front leg
> behaves.

> **Q: Why exit on signal reversion instead of holding to expiry?**
> The trade was opened against a mispricing. When the mispricing is gone the
> reason to hold is gone, whatever the position is worth. Holding to expiry turns
> a relative-value trade into a directional bet on the front leg pinning.

> **Q: Three years and only seventeen trades. Is that significant?**
> No, and I would not claim it is. Seventeen trades cannot establish an edge, in
> either direction — which is also why I do not read the P&L difference between
> model variants as evidence about them. What the sample does support is the
> direction finding, which rests on **867 scored days** rather than 17 trades:
> the edge study measures every day the signal spoke, not only the ones that
> cleared the entry threshold. The backtest tells me the economics do not work;
> the edge study tells me why. Matching the claim to the sample that can carry it
> is the discipline being tested by this question.

> **Q: Why did the exact-DTE matching fail?**
> A market-structure change. NSE moved NIFTY's weekly expiry from Thursday to
> Tuesday during 2025, so days-to-expiry combinations from before the change
> barely occur after it. The state space splits into two eras and the current one
> has almost no history to match against — eight comparables were reachable on
> 0.1% of days. That is why matching is a distance with a cap rather than a
> bucket. It is also the kind of thing you only catch by looking at why the
> output was empty instead of loosening the threshold until something appeared.

> **Q: What would you do next?**
> Proper SPAN margin from the exchange risk file, because the flat approximation
> will be most wrong in a vol shock, which is when the strategy is most exposed.
> A walk-forward test where the parameters themselves are chosen only on prior
> data — right now they are fixed across the sample, which is a mild form of the
> bias the rest of the project avoids. And switching the entry threshold from a
> z-score to a percentile, for the reason in §5c.

---

## 5b. The basis — the state variable that was missing

This is the best thing to talk about, because it is a correction to the model
rather than a feature of it, and it came from arguing with the model rather than
from running it.

**The problem.** A calendar's two legs are struck at the same price but priced
off *two different forwards* — each expiry has its own. The gap between those
forwards is the basis, and it moves the debit on its own. At a fixed strike, a
wider basis pushes the call leg relatively into the money and the put leg
relatively out of it. So two days that agree on vol and on both legs'
days-to-expiry can still be pricing materially different trades. The model had no
way to see that: its state was front DTE, back DTE and front IV, and nothing
else.

**The evidence.** Within matched states, the correlation between the basis and
the debit is **+0.356 for calls and −0.178 for puts**. The opposite signs are the
whole argument — that is exactly what put-call parity predicts, and noise does
not produce a sign flip that lines up with theory. One standard deviation of
basis is worth about **13.7 points of debit** on the call side, against a
fair-value dispersion of roughly 32 points.

**The calibration.** The scale is measured, not chosen. A front-leg day moves the
debit about 10.4 points; a basis point moves it about 0.37. So ~28 points of
basis is the same size of step as one day on the front leg — rounded to 30, which
keeps the convention that one unit of distance means "about as different as one
day on the front leg".

> **Q: Did adding it improve the strategy?**
> No, and that is the honest answer. The fair-value estimate tightened by 6.2%
> and the buy-side hit rate went from 63.2% to 67.7%. The backtest got *worse* —
> −0.35% to −0.56% — and fewer days now reach eight comparables, 68% down to 60%,
> because a fourth dimension makes neighbours harder to find.

> **Q: Then why keep it?**
> Because of which measurement has the sample to support it. The dispersion
> improvement is measured across ~900 scored days. The P&L difference is measured
> across 17 trades, where a swing of that size is noise. Judging a specification
> change on the seventeen-trade number would be reading signal into a sample that
> cannot carry it. And the change is not a tuned parameter — it corrects a
> mis-specification with a known mechanism. The hit-rate improvement, for
> completeness, is *not* statistically significant either (p = 0.60).

> **Q: Where does the model already handle this correctly?**
> The diagonal explorer. It matches each leg on its own moneyness relative to its
> own expiry's forward, so the basis is controlled for implicitly. The calendar
> path did not, because both legs share a strike and ATM was chosen off the front
> forward alone. The fix was to make the calendar do explicitly what the diagonal
> was already doing implicitly.

---

## 5c. Two biases found by asking "would this work live?"

**The time-of-day bias.** The comparable history is built from end-of-day
settlement prices. Read a live quote at 10:00 against that history and the
comparison is not like for like: measured across 534 sessions, the ATM calendar
debit sits about **9 points below** its own end-of-day settlement in the morning,
and converges to it by 15:20. In z-score terms that is a systematic **−0.27
shift** — every morning reads cheaper than it is.

Rebuilding the whole series at 10:00 from minute data and matching it against
10:00 history removes it exactly:

| | Mean z |
|---|---|
| 10:00 reading vs EOD history (the mistake) | −0.497 |
| 10:00 reading vs 10:00 history (matched) | −0.231 |
| Control: EOD vs EOD, same sessions | −0.252 |

The analogy that makes it land: weighing yourself every morning but comparing
against a diary of evening weights. You will conclude you are losing weight every
single day.

**The z-score is not symmetric.** The debit distribution is right-skewed (skew
+1.01), so the z-scores are too (+0.93). With symmetric ±1.5 thresholds that
produces **6.9% of days above +1.5 and only 3.1% below −1.5** — the model fires
more than twice as many sell signals as buys, purely from distribution shape. And
the sell side is the side the edge study says is wrong. The percentile is
rank-based and immune to this (mean 52.7, near the unbiased 50), which is why
switching the threshold to a percentile is on the list.

> **Q: Is the tool live?**
> No. It runs on bhavcopy, which NSE publishes after the close, so the dashboard
> shows the last settled session. NSE's live option-chain endpoint works and
> carries bid and ask, so a live reading is buildable — but it would need the
> time-of-day matched history above, or it would print a buy most mornings.

---

## 5d. The diagonal explorer, and why it is not backtested

The dashboard will price any two legs — different strikes as well as different
expiries — against that structure's own history. Two things are worth being able
to say about it.

**Why it is computed on demand.** Precomputing every strike against every strike
across the sample is about 98 million rows. The history for one chosen structure
is a few hundred lookups, so the chain is indexed once (~4 seconds) and each
query returns in under a tenth of a second. The right answer to "isn't that
expensive?" is that the expensive version was never necessary.

**Why past days are matched on moneyness, not strike number.** NIFTY ran from
roughly 19,000 to 23,400 across the sample. A 24,000 call was 26% out of the
money at one end of that and 2.5% out at the other — not the same trade. Matching
by strike number would put those side by side.

**Why it carries a warning.** A calendar is described by three state variables;
a diagonal by five. The neighbourhood thins fast, so the average match distance
is reported with every verdict and the page flags it once it passes 1.5 units.

> **Q: Why not backtest it and find the best structure?**
> Because scanning every strike and expiry for the most stretched reading is
> choosing a winner from an enormous draw, and the winner would be selection
> rather than edge. The backtest deliberately trades one defined structure. The
> explorer is a lookup for a trade you are already considering, and the tab says
> so in as many words.

---

## 6. If asked how you built it

Answer plainly. You directed the build, you made the modelling calls, and you can
defend every one of them — which is what this document is. The domain judgement
in here is the part that is hard to acquire: knowing that settlement prices beat
closes on untraded strikes, that the index level has to be normalised out, that a
calendar is margined as a spread and not as a naked short, that the front leg's
DTE matters more than the back leg's. That knowledge came off the desk.

What to be careful about: do not claim to have hand-typed every line, and do not
under-claim either. The honest and strongest version is that you specified it,
made the decisions, validated it, and know what would break it.

If they push on a specific piece of code, the safe move is to explain *what it
does and why it is there* rather than recite syntax. That is what they are
actually testing.

---

## 7. Numbers to have at your fingertips

From the final run — know these cold:

From the current run — the basis is in the state, long side only. Know these cold:

| | |
|---|---|
| Sample | Sep 2023 – Sep 2026, 746 trading days, 724 in the backtest |
| Days the signal could speak | 60% (the rest had too few comparables) |
| Comparables behind a signal | 15, median state distance 2.03, cap 3.0 |
| Trades | 17 (long side only) |
| Win rate | 52.9% |
| Average holding period | 3.6 days |
| Gross P&L | **−₹2,596** |
| Costs | **₹3,020** (0.30% of capital) |
| Net P&L | **−₹5,616 (−0.56%)** |
| Max drawdown | −0.83% |
| Sharpe | −0.40 |
| Signal distribution | 1.8% buy, 2.6% sell, 55.6% flat, 40.1% no call |
| Without the basis, for contrast | −0.35%, 20 trades, gross ₹2 |
| Both directions | −2.51%, Sharpe −0.93 |
| Intraday, real minute fills | **−0.26%, Sharpe −0.34, 37% win rate** |
| Intraday, daily-proxy fills (rejected) | +1.17% — an artefact, see 7b |
| Minute data validated | open/high/low match NSE on 99.7% of contract-days |

Hit rates from the edge study, three-day horizon: **89%** at z < −2 (n=9),
**69%** at −2 to −1.5, **64%** at −1.5 to −1, against **17%** at 1.5 to 2 and
**40%** at z > 2. Quote the n alongside the 89% — nine observations is a number
to be honest about, not one to lead with.

**The line to lead with:** on seventeen trades the P&L is not the finding. The
finding is the asymmetry in the edge study, which rests on 867 scored days: the
cheap side is right about two thirds of the time and the rich side is inverted,
for a mechanical reason — the debit drifts wider on its own as the front leg
decays. The strategy does not clear its costs, and the project is built to show
why rather than to bury it.

---

## 7b. The strongest thing in this project

The sequence, in order — this is what to walk an interviewer through:

1. The overnight backtest lost money, and the edge study said why: the signal has
   direction but not magnitude.
2. A calendar is short gamma, so the obvious suspect was the overnight gaps. I
   tested closing before the bell using bhavcopy's daily open and close, and it
   turned positive — +1.10%, Sharpe 0.66, win rate up from 55% to 64%.
3. **I did not believe it**, and I wrote down exactly why before testing further:
   each leg's "open" in bhavcopy is its own first trade at an unknown moment, so
   the entry spread I was pricing never existed as a quote. I quantified what
   would have to be true for it to survive — capturing 29% of the session's move.
4. I got minute data, validated it against NSE first (open/high/low match on
   99.7% of contract-days; close matches on NSE's last-half-hour VWAP convention;
   volume differs by exactly the lot size), then re-ran the identical trades with
   both legs priced at the same instant.
5. The edge vanished: +1.17% became **−0.26%**, win rate 74% became 37%, gross
   ₹15,123 became ₹837. The assumed entry had been favourable by a median of 4.35
   points against a median real move of 6.85 — the phantom advantage was two
   thirds the size of the thing being traded.
6. A sweep of 28 entry/exit times confirmed it is not a clock problem: 9 positive,
   median −0.17%, and the best result is the best of 28 on 19 trades.

> **The line:** I found a result I wanted to be true, named the assumption it
> depended on, went and got the data that could kill it, and it did.

**Q: Why not just publish the +1.10% version?**
Because it was not a strategy, it was an artefact of how the data was recorded.
Someone would have traded it.

**Q: How did you know to be suspicious?**
Because of what the number was made of. The edge was concentrated in the entry
price, not in the exit, and the entry price was the one thing bhavcopy could not
actually tell me. When a result depends most on the weakest part of your data,
that is the thing to go and test.

---

## 8. How to present a strategy that lost money

This is the part most candidates get wrong, so it is worth rehearsing.

Do not open with the backtest. Open with the question, then the method, then the
finding — and present the finding as a finding, because it is one:

> I built a fair-value model for calendar spreads and tested whether the
> deviations were tradeable. They are not, and I can show you exactly why. The
> signal has real direction — cheap spreads widen afterwards about two thirds of
> the time — but the move is smaller than the cost of four orders. The part I
> did not expect was the asymmetry: the rich side isn't just weaker, it's
> inverted, because the debit drifts wider on its own as the front leg decays.
> Selling a rich calendar is betting against the structure's own carry.

Then the three things that make it credible:

1. **You separated the signal from the strategy.** The edge study measures
   prediction before any question of P&L, so "it lost money" can be attributed to
   the right cause instead of being one undifferentiated failure.
2. **You did not keep tuning until it was green.** You tested a holding-period
   cap, it made things worse, and you left it out and said so. Interviewers are
   specifically listening for whether you know the difference between a finding
   and a fit.
3. **You know what would change the answer.** Intraday data would cut the
   slippage assumption. A wider structure — a longer front leg, or further from
   the money — would have a larger move to cover the same fixed cost. Those are
   the next tests, and you can say why each one is the right next test.

**If they ask "so was the project a failure?"** — no. The question was whether
the deviations were tradeable. They are not, at this size and this cost base, and
that is an answer. The alternative outcome, had I not measured it this carefully,
would have been to trade it.

**What not to do:** do not apologise for it, do not reach for a version of the
numbers that looks better, and do not claim a Sharpe you cannot reproduce in
front of them. The repo is on GitHub and it runs.
