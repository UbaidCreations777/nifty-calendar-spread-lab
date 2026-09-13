"""Indian index-option transaction costs, charged per leg.

A calendar is a four-order round trip - sell the front and buy the back on the
way in, reverse both on the way out - and every one of those orders pays. On a
structure whose entire edge is a few tenths of a percent of spot, costs are not a
rounding error; they are most of the question. A backtest that leaves them out
does not show a smaller profit than reality, it usually shows a profit where
reality has a loss.

Rates are the FY26 schedule for NSE index options. STT is charged on the sell
side of premium only, stamp duty on the buy side only, and GST applies to
brokerage and the regulatory fees rather than to the premium.
"""
from __future__ import annotations

from dataclasses import dataclass

from .. import config as C


@dataclass(frozen=True)
class LegCost:
    brokerage: float
    stt: float
    exchange: float
    sebi: float
    stamp: float
    gst: float
    slippage: float

    @property
    def total(self) -> float:
        return (self.brokerage + self.stt + self.exchange + self.sebi
                + self.stamp + self.gst + self.slippage)


def leg_cost(premium: float, quantity: int, is_buy: bool) -> LegCost:
    """Cost of transacting one leg at `premium` for `quantity` units."""
    turnover = abs(premium) * quantity

    brokerage = C.BROKERAGE_PER_ORDER
    stt = 0.0 if is_buy else turnover * C.STT_SELL_PREMIUM
    exchange = turnover * C.EXCHANGE_TXN_PCT
    sebi = turnover * C.SEBI_TURNOVER_PCT
    stamp = turnover * C.STAMP_DUTY_BUY_PCT if is_buy else 0.0
    gst = (brokerage + exchange + sebi) * C.GST_PCT

    # Crossing the spread is a real cost and it is paid on entry and on exit, so
    # it is charged per leg rather than netted once.
    slippage = C.SLIPPAGE_TICKS * C.TICK_SIZE * quantity

    return LegCost(brokerage, stt, exchange, sebi, stamp, gst, slippage)


def round_trip_cost(front_px_in: float, back_px_in: float,
                    front_px_out: float, back_px_out: float,
                    quantity: int, is_long_calendar: bool) -> dict:
    """Total cost of entering and exiting one calendar spread.

    A long calendar sells the front and buys the back; a short calendar is the
    mirror. Each leg is transacted twice.
    """
    front_is_buy_on_entry = not is_long_calendar
    back_is_buy_on_entry = is_long_calendar

    legs = [
        ("front_entry", leg_cost(front_px_in, quantity, front_is_buy_on_entry)),
        ("back_entry", leg_cost(back_px_in, quantity, back_is_buy_on_entry)),
        ("front_exit", leg_cost(front_px_out, quantity, not front_is_buy_on_entry)),
        ("back_exit", leg_cost(back_px_out, quantity, not back_is_buy_on_entry)),
    ]

    breakdown = {name: cost.total for name, cost in legs}
    breakdown["total"] = sum(breakdown.values())
    breakdown["stt"] = sum(cost.stt for _, cost in legs)
    breakdown["brokerage"] = sum(cost.brokerage for _, cost in legs)
    breakdown["slippage"] = sum(cost.slippage for _, cost in legs)
    return breakdown
