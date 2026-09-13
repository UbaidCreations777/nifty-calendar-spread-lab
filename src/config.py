"""Central configuration for the NIFTY calendar spread lab."""
from __future__ import annotations

from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"

# ---------------------------------------------------------------- data source
NSE_BASE = "https://www.nseindia.com"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)
TIMEOUT = 30
REQUEST_DELAY = 0.6

# NSE moved the F&O bhavcopy to the UDiFF format on 2024-07-08. Older dates only
# exist in the legacy layout, so both are tried and whichever answers wins.
SOURCES = [
    {
        "name": "udiff",
        "url": ("https://nsearchives.nseindia.com/content/fo/"
                "BhavCopy_NSE_FO_0_0_0_{YYYYMMDD}_F_0000.csv.zip"),
    },
    {
        "name": "legacy",
        "url": ("https://nsearchives.nseindia.com/content/historical/DERIVATIVES/"
                "{yyyy}/{MON}/fo{dd}{MON}{yyyy}bhav.csv.zip"),
    },
]

BACKTEST_START = date(2023, 9, 1)
BACKTEST_END = date(2026, 9, 12)

# ------------------------------------------------------------------ contract
UNDERLYING = "NIFTY"

# NSE has revised the NIFTY lot size several times. The UDiFF files carry the lot
# size per contract so it is read from the data where available; this schedule is
# the fallback for legacy-format dates. Keyed by the date the size took effect.
LOT_SIZE_SCHEDULE = {
    date(2021, 10, 29): 50,
    date(2024, 11, 20): 25,
    date(2025, 12, 26): 75,
}
LOT_SIZE_DEFAULT = 75

# ------------------------------------------------------------------- pricing
RISK_FREE_RATE = 0.065          # ~India 91-day T-bill through the sample
IV_SOLVER_TOL = 1e-6
IV_SOLVER_MAX_ITER = 100
IV_BOUNDS = (0.01, 3.0)         # 1% to 300% annualised
# Below one tick an option cannot be quoted, so a "price" under it carries no
# volatility information and is rejected rather than solved to a junk root.
MIN_OPTION_PRICE = 0.05

# --------------------------------------------------------------- fair value
# The state a calendar spread is matched on. Two historical days are comparable
# when they share the option type, both legs' days-to-expiry, and sit in the same
# front-leg ATM IV bucket.
IV_BUCKET_HALF_WIDTH = 0.5      # IV 16.0 matches history in [15.5, 16.5]
# The front leg's DTE must match exactly; the back leg gets this much room. Two
# days out of thirty changes a monthly leg very little, and requiring an exact
# match on both legs leaves too few comparable days to form a distribution.
BACK_DTE_TOLERANCE = 2

# --- nearest-neighbour matching (the default)
# Fixed buckets fail on this data: NSE moved NIFTY's weekly expiry from Thursday
# to Tuesday during 2025, which splits the days-to-expiry state space into two
# eras and leaves the current one with almost no history to match against. Only
# 0.1% of days reach eight comparables under an exact-DTE rule.
#
# Neighbours are taken in a state space scaled so that one unit means "about as
# different as one day on the front leg". A day on the front leg changes the
# structure materially; three days on a thirty-day back leg is the rough
# equivalent; half a vol point is the comparable move in IV. These scales are the
# domain judgement in the model, written down explicitly rather than buried in a
# standardisation.
KNN_K = 15
STATE_SCALE_FRONT_DTE = 1.0
STATE_SCALE_BACK_DTE = 3.0
STATE_SCALE_IV = 0.5
# Beyond this distance a historical day is not the same trade, and is dropped
# even if that leaves too few neighbours to speak. Without the cap, kNN always
# returns an answer - including on days whose nearest neighbour is nothing like
# them.
MAX_STATE_DISTANCE = 3.0
MIN_COMPARABLES = 8             # below this the signal reports "insufficient data"
# A z-score divides by the spread of the comparables, so a set that barely varies
# produces enormous scores from rounding-sized differences. Below this ratio of
# standard deviation to mean the distribution is treated as having no usable
# width rather than as a very strong signal.
MIN_RELATIVE_DISPERSION = 0.01
ATM_TOLERANCE_PCT = 0.005       # strike within 0.5% of spot counts as ATM

Z_ENTRY = 1.5                   # |z| beyond this is a trade
Z_EXIT = 0.5                    # mean reversion back inside this closes it

# --------------------------------------------------------------- backtesting
INITIAL_CAPITAL = 1_000_000     # Rs 10 lakh
MAX_MARGIN_UTILISATION = 0.60   # never commit more than 60% of capital to margin
# A calendar is margined as a spread, not as a naked short: the long back leg
# offsets the short front leg, so SPAN charges a fraction of what the short alone
# would cost. This is an approximation of that netted number as a percentage of
# notional - the exact figure needs the exchange's SPAN risk parameter file.
MARGIN_PER_LOT_PCT = 0.025

# Margin availability is a broker's constraint, not a risk limit. Sizing purely
# on it puts on whatever the deposit allows, which on this structure reached 22
# lots and an 8% single-trade loss. So position size is also capped by what an
# adverse move would cost: measured over this sample, a 99th-percentile daily
# move in the spread is about 1% of spot, and no single trade is allowed to lose
# more than 2% of capital on a move that size.
RISK_PER_TRADE_PCT = 0.02
STRESS_MOVE_PCT_OF_SPOT = 1.0

# Front leg is the near weekly (sold), back leg the monthly (bought).
FRONT_DTE_MIN, FRONT_DTE_MAX = 2, 9
BACK_DTE_MIN, BACK_DTE_MAX = 10, 45

# ------------------------------------------------------------------ costs
# Rates as applicable to NSE index options, FY26.
BROKERAGE_PER_ORDER = 20.0      # flat discount-broker rate, per leg per side
STT_SELL_PREMIUM = 0.001        # 0.1% on sell-side premium
EXCHANGE_TXN_PCT = 0.0003503    # NSE options, on premium turnover
SEBI_TURNOVER_PCT = 0.000001
STAMP_DUTY_BUY_PCT = 0.00003    # 0.003% on buy side
GST_PCT = 0.18                  # on brokerage + exchange + SEBI charges
SLIPPAGE_TICKS = 2              # ticks crossed per leg
TICK_SIZE = 0.05

# --------------------------------------------------------------- direction
# The edge study finds the signal's two sides are not symmetric. Cheap spreads
# subsequently widen, which is what a long calendar wants, and the hit rate runs
# around two thirds. Rich spreads also widen - the debit drifts wider on its own
# as the front leg decays faster than the back - so selling a rich calendar
# fights the structure's own carry, and hits around a third of the time. Trading
# only the long side is the conclusion that evidence supports.
ALLOW_SHORT = False
# The same study shows the signal's information gone by the fifth day, which
# suggests capping how long a position is carried. Tested at two, three and five
# days, it made the result slightly worse rather than better, so it is left off:
# a parameter that does not improve anything is a parameter fitted to noise. The
# exit rule remains signal reversion, or the day before the front leg expires.
MAX_HOLDING_DAYS = None
