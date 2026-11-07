"""
Autoquoter: a class-based port of dxy_autoquoter.py's quoting logic,
generalized so one Autoquoter instance can run per ticker/instrument.

WHY A CLASS: dxy_autoquoter.py hardcoded a single global TICKER and a
DXYPricer instance at module level, so only one contract could be quoted
per process. This class takes ticker + a pluggable `price_fn` in its
constructor instead, so the SAME class serves any instrument -- DXY,
Treasury-yield binaries, whatever -- as long as price_fn() returns:
    {"fair_yes_cents": float, "fair_no_cents": float,
     "spot": float, "strike": float}
That's the concrete first step toward the stated long-term goal: running
several Autoquoter instances side by side, one per ticker, each with its
own price_fn. This file does NOT build that multi-instance orchestrator
yet (would need threading/asyncio/multiprocessing since run_forever() is
a blocking loop) -- that's the deliberately-deferred next step.

WHAT WAS PRESERVED EXACTLY FROM dxy_autoquoter.py:
  - Dynamic edge sizing (edge shrinks linearly away from the 50c center)
  - ITM/OTM single-sided quoting (only the ITM side is quoted; OTM is
    pulled entirely)
  - "Place new -> confirm resting -> cancel old" repricing pattern in
    maybe_requote_side()

A DISCREPANCY WORTH FLAGGING, FOUND WHILE PORTING THIS: the ORIGINAL
dxy_autoquoter.py's maybe_requote_side() docstring describes using
kc.amend_order() for "TRUE ZERO-DOWNTIME, NO-OVERLAP" repricing (same
order_id throughout, never two orders resting at once). The actual
original implementation never calls amend_order() at all -- it does
place-then-cancel instead, which avoids a zero-coverage gap but DOES
have a brief window with two orders resting simultaneously, contradicting
what the docstring claims. This class preserves that same place-then-
cancel BEHAVIOR unchanged (not silently rewriting live trading logic
during a refactor), but if you want it to actually match its own
docstring, TrueAmendRequote below is a drop-in alternative -- see its
docstring for exactly what would change.

SAFETY: dry_run defaults to True (same convention used everywhere else in
this project). In dry-run mode, run_cycle() computes and prints every
action it WOULD take without calling place_limit_order/cancel_order/
amend_order/cancel_orders_batch.
"""
from __future__ import annotations

import os

import time
from typing import Callable, Optional

from KalshiClient import KalshiClient


def resting_price_and_size(orders: list[dict], side: str) -> tuple[int | None, float]:
    """PURE function, no API call: derives (price_cents, total_remaining_size)
    for `side` from an ALREADY-FETCHED orders list (e.g. from a single
    kc.get_open_orders(ticker) call). Mirrors KalshiClient.get_my_resting_price()'s
    logic exactly, but operates on data you already have instead of making a
    fresh HTTP request every time -- get_my_resting_price() itself calls
    get_open_orders() internally with ZERO caching, so calling it twice (once
    for "yes", once for "no") on the same ticker means two separate HTTP
    round-trips for data that's identical between the two calls except for
    which side gets filtered. With many tracked tickers, calling this instead
    of get_my_resting_price() twice per ticker, and sharing ONE
    get_open_orders() fetch across every consumer that needs it (run_cycle,
    maybe_requote_side's existing-order lookup, and the orchestrator's
    stray-order scan), cuts what were 100+ redundant HTTP calls per cycle
    down to one per unique ticker."""
    by_price: dict[int, float] = {}
    for o in orders:
        if KalshiClient.economic_side(o) != side or not KalshiClient._is_actually_resting(o):
            continue
        pd_ = o.get(KalshiClient.economic_price_field(o))
        if not pd_:
            continue
        pc = round(float(pd_) * 100)
        by_price[pc] = by_price.get(pc, 0.0) + float(o.get("remaining_count_fp", 0))
    if not by_price:
        return None, 0.0
    best_price = max(by_price, key=lambda k: by_price[k])
    return best_price, by_price[best_price]


class Autoquoter:
    def __init__(
        self,
        kc: KalshiClient,
        ticker: str,
        price_fn: Callable[[], dict],
        *,
        quote_notional_dollars: float = 30.0,
        total_budget_dollars: float = 50.0,
        edge_center_cents: int = 50,
        edge_at_center_cents: int = 5,
        edge_slope: float = 0.08,
        edge_min_cents: int = 1,
        requote_threshold_cents: int = 3,
        dry_run: bool = True,
        verbose: bool | None = None,
        label: Optional[str] = None,
    ):
        self.kc = kc
        self.ticker = ticker
        self.price_fn = price_fn

        self.quote_notional_dollars = quote_notional_dollars
        self.total_budget_dollars = total_budget_dollars
        self.edge_center_cents = edge_center_cents
        self.edge_at_center_cents = edge_at_center_cents
        self.edge_slope = edge_slope
        self.edge_min_cents = edge_min_cents
        self.requote_threshold_cents = requote_threshold_cents
        self.dry_run = dry_run
        self.verbose = dry_run if verbose is None else verbose  # backward-compatible default
        self.label = label or ticker

        self.closing_quote_state = {"order_id": None, "price_cents": None, "side": None}

    def _log(self, msg: str) -> None:
        prefix = f"[{self.label}]" + (" DRY_RUN" if self.dry_run else "")
        print(f"{prefix} {msg}")

    # -----------------------------------------------------------------
    # PURE HELPERS (ported 1:1 from dxy_autoquoter.py)
    # -----------------------------------------------------------------
    @staticmethod
    def clamp_price(c: int) -> int:
        return max(1, min(99, c))

    def dynamic_edge_cents(self, fair_price_cents: int) -> int:
        distance_from_center = abs(fair_price_cents - self.edge_center_cents)
        edge = self.edge_at_center_cents - self.edge_slope * distance_from_center
        return max(self.edge_min_cents, round(edge))

    def compute_target_price(self, fair_price_cents: int, best_bid: int | None, best_ask: int | None) -> int | None:
        """Base implementation always returns an int. Subclasses may
        return None to signal "don't quote this side at all" -- see
        maybe_requote_side()'s docstring for exactly what that does."""
        edge = self.dynamic_edge_cents(fair_price_cents)
        candidate = fair_price_cents - edge
        if best_ask is not None:
            candidate = min(candidate, best_ask - 1)
        return self.clamp_price(candidate)

    @staticmethod
    def determine_itm_side(spot: float, strike: float) -> str | None:
        if spot > strike:
            return "yes"
        elif spot < strike:
            return "no"
        return None

    # -----------------------------------------------------------------
    # PORTFOLIO / BUDGET (ported 1:1, scoped to self.ticker)
    # -----------------------------------------------------------------
    def get_filled_exposure_dollars(self, positions: list[dict] | None = None) -> float:
        if positions is None:
            positions = self.kc.get_positions()
        for p in positions:
            if p.get("ticker") == self.ticker:
                try:
                    return abs(float(p.get("market_exposure_dollars", 0)))
                except (TypeError, ValueError):
                    return 0.0
        return 0.0

    def get_resting_notional_dollars(self) -> float:
        total = 0.0
        for o in self.kc.get_open_orders(self.ticker):
            if not KalshiClient._is_actually_resting(o):
                continue
            price_dollars = o.get(KalshiClient.economic_price_field(o))
            remaining = float(o.get("remaining_count_fp", 0))
            if price_dollars is not None:
                total += float(price_dollars) * remaining
        return total

    def get_position_info(self, positions: list[dict] | None = None) -> tuple[str | None, float, float]:
        if positions is None:
            positions = self.kc.get_positions()
        for p in positions:
            if p.get("ticker") != self.ticker:
                continue
            position_fp = float(p.get("position_fp", 0))
            if position_fp == 0:
                return None, 0.0, 0.0
            held_side = "yes" if position_fp > 0 else "no"
            qty = abs(position_fp)
            exposure = abs(float(p.get("market_exposure_dollars", 0)))
            avg_cost_cents = round((exposure / qty) * 100) if qty > 0 else 0
            return held_side, qty, avg_cost_cents
        return None, 0.0, 0.0

    # -----------------------------------------------------------------
    # QUOTING (ported behavior-identical from dxy_autoquoter.py)
    # -----------------------------------------------------------------
    _UNSET = object()  # sentinel: distinguishes "caller didn't pass this" from "caller passed None"

    def maybe_requote_side(self, side: str, target_price: int | None,
                            filled_exposure: float, other_side_resting_notional: float,
                            current_price=_UNSET, current_size=_UNSET,
                            fair_price_cents: float | None = None,
                            spot: float | None = None, strike: float | None = None,
                            hours_remaining: float | None = None,
                            priced_maturity_hours: float | None = None,
                            best_bid: int | None = None, best_ask: int | None = None,
                            open_orders: list[dict] | None = None) -> None:
        """
        BUDGET MODEL: self.total_budget_dollars is treated as a CEILING that
        may be refreshed by the caller every cycle (e.g. an orchestrator
        recomputing it from real, ACCOUNT-WIDE FILLED exposure only -- see
        kalshi_treasury_quoter.py). Resting/unfilled orders do NOT consume
        this budget (Kalshi doesn't lock capital for merely resting orders),
        so `committed` here is filled exposure only, not resting notional.
        Every line independently gets to size up to the SAME ceiling -- this
        is not a shared pool divided across tickers, since there's no actual
        contention: Kalshi allows many simultaneously-resting orders whose
        combined notional exceeds the account balance, as long as each
        individual one is itself affordable and nothing has filled yet.

        Because the ceiling can shrink between cycles (as fills accumulate
        anywhere on the account), this checks SIZE as well as price: an
        existing order that's now oversized relative to the current ceiling
        gets shrunk (or pulled entirely if the ceiling no longer supports
        any size) -- not just left alone because its price hasn't moved.

        target_price=None is a DELIBERATE "don't quote this side at all"
        signal from compute_target_price() -- e.g. BestBidOrFairAutoquoter
        uses this when the best bid is already at 99c, since Kalshi's
        liquidity-reward policy requires liquidity on BOTH sides and a
        single-sided strategy matching an already-maximal best bid doesn't
        help meet that -- there's no reward benefit to holding capital
        there. This PULLS any existing resting order on that side (not
        just skips placing a new one), since leaving a stale order there
        would contradict the "shouldn't be quoted here" decision.

        open_orders: pass a pre-fetched kc.get_open_orders(ticker) list to
        avoid a fresh HTTP call here for existing-order-id lookup -- see
        run_cycle's docstring for why this matters at scale.
        """
        if current_price is self._UNSET:
            # Backward-compatible fallback for anyone calling this directly
            # without already having the resting price on hand.
            if open_orders is not None:
                current_price, current_size = resting_price_and_size(open_orders, side)
            else:
                current_price, current_size = self.kc.get_my_resting_price(self.ticker, side)

        maturity_str = f", maturity={hours_remaining:.1f}h" if hours_remaining is not None else ""
        # Only show the priced (buffered) value when it actually differs
        # from the true value -- keeps the log uncluttered when
        # maturity_buffer_minutes is 0, and makes the buffer's real effect
        # directly visible (not just asserted in a comment) whenever it's on.
        if (priced_maturity_hours is not None and hours_remaining is not None
                and abs(priced_maturity_hours - hours_remaining) > 1e-6):
            maturity_str += f" (priced at {priced_maturity_hours:.1f}h w/ buffer)"
        spread_str = ""
        if best_bid is not None or best_ask is not None:
            bid_str = f"{best_bid}c" if best_bid is not None else "none"
            ask_str = f"{best_ask}c" if best_ask is not None else "none"
            spread_str = f", bid/ask={bid_str}/{ask_str}"
        context = (f"spot={spot:.5f}, strike={strike:.5f}, fair_{side}={fair_price_cents:.2f}c"
                   f"{spread_str}{maturity_str}"
                   if fair_price_cents is not None else "")

        if target_price is None:
            # Deliberate "don't quote this side at all" signal -- see
            # docstring above. Pull any existing resting order rather than
            # just skipping placement, since leaving a stale order here
            # would contradict the decision not to quote this side.
            if current_price is None:
                if self.verbose:
                    self._log(f"{side.upper()}: no target (compute_target_price declined to "
                               f"quote this side), nothing resting -- no action. {context}")
                return
            if open_orders is None:
                open_orders = self.kc.get_open_orders(self.ticker)
            existing_orders = [
                o for o in open_orders
                if self.kc.economic_side(o) == side and self.kc._is_actually_resting(o)
            ]
            was = f"{current_price}c ({current_size:.0f} contracts)"
            if self.dry_run:
                self._log(f"WOULD PULL {side.upper()} entirely: compute_target_price declined "
                           f"to quote this side (was {was}). {context}")
                return
            old_order_ids = [o["order_id"] for o in existing_orders]
            if old_order_ids:
                self.kc.cancel_orders_batch(old_order_ids)
            self._log(f"PULLED {side.upper()} entirely: compute_target_price declined to "
                       f"quote this side (was {was}). {context}")
            return

        price_dollars = target_price / 100.0
        if price_dollars <= 0:
            return

        # --- Compute the size the current ceiling actually supports ---
        desired_size = max(1, int(self.quote_notional_dollars // price_dollars))
        committed = filled_exposure + other_side_resting_notional
        remaining_budget = self.total_budget_dollars - committed
        max_affordable = max(0, int(remaining_budget // price_dollars)) if remaining_budget > 0 else 0
        size = min(desired_size, max_affordable)

        price_close = current_price is not None and abs(target_price - current_price) < self.requote_threshold_cents
        size_ok = current_price is not None and current_size is not None and size >= current_size
        is_shrink = current_price is not None and current_size is not None and size < current_size

        if price_close and size_ok:
            if self.verbose:
                self._log(f"{side.upper()}: already resting at {current_price}c "
                           f"({current_size:.0f} contracts), {context} — no action.")
            return

        if open_orders is None:
            open_orders = self.kc.get_open_orders(self.ticker)
        existing_orders = [
            o for o in open_orders
            if self.kc.economic_side(o) == side and self.kc._is_actually_resting(o)
        ]
        was = f"{current_price}c ({current_size:.0f} contracts)" if current_price is not None else "none"

        if size <= 0:
            if current_price is None:
                # Nothing resting and nothing affordable -- genuinely nothing to do.
                self._log(f"SKIPPING {side.upper()} requote @ {target_price}c: would exceed "
                           f"${self.total_budget_dollars:.2f} budget (committed=${committed:.2f}, "
                           f"remaining=${remaining_budget:.2f})")
                return

            # Something IS resting but the ceiling no longer supports ANY size on
            # this side -- pull it entirely rather than leaving an unaffordable
            # order sitting there untouched forever.
            if self.dry_run:
                self._log(f"WOULD PULL {side.upper()} entirely: budget no longer supports any size "
                           f"(was {was}), {context}")
                return
            old_order_ids = [o["order_id"] for o in existing_orders]
            if old_order_ids:
                self.kc.cancel_orders_batch(old_order_ids)
            self._log(f"PULLED {side.upper()} entirely: budget no longer supports any size "
                       f"(was {was}), {context}")
            return

        action_word = "SHRINK" if is_shrink else "requote"

        if self.dry_run:
            self._log(f"WOULD {action_word} {side.upper()}: {size} @ {target_price}c "
                       f"(${size*price_dollars:.2f} notional, was {was}), {context}")
            return

        old_order_ids = [o["order_id"] for o in existing_orders]

        if is_shrink:
            # SHRINKING: cancel the old (larger) order FIRST, then place the
            # smaller one. This is the opposite of the usual place-then-
            # cancel pattern, and deliberately so: place-then-cancel would
            # momentarily need capital for BOTH the old (larger) and new
            # (smaller) order at once -- exactly backwards when the entire
            # reason we're shrinking is that capital is tight. Confirmed in
            # practice: a 515->508 contract shrink (a 7-contract reduction)
            # failed with insufficient_balance under place-then-cancel,
            # because for a moment it needed capital for 515+508=1023
            # contracts, not the 508 the new order actually needs. Trade-off:
            # this accepts a brief coverage gap (no resting order on this
            # side between the cancel and the new placement) in exchange for
            # actually being able to shrink when capital is constrained.
            if old_order_ids:
                self._log(f"Cancelling old {side.upper()} order(s) FIRST (shrink -- avoids momentarily "
                           f"needing capital for both old and new size at once)")
                self.kc.cancel_orders_batch(old_order_ids)

            self._log(f"ATTEMPTING {action_word} {side.upper()}: {size} @ {target_price}c "
                       f"(${size*price_dollars:.2f} notional, was {was})")
            try:
                result = self.kc.place_limit_order(self.ticker, side=side, price_cents=target_price, count=size)
            except Exception as e:  # noqa: BLE001
                self._log(f"{action_word.upper()} PLACEMENT FAILED (old order already cancelled -- "
                           f"this side now has NO resting order until the next successful cycle): "
                           f"{type(e).__name__}: {e} -- was attempting {side.upper()} {size} @ "
                           f"{target_price}c (${size*price_dollars:.2f} notional)")
                raise
            new_is_resting = (result.get("fill_count", "0.00") != "0.00"
                               or result.get("remaining_count", "0.00") != "0.00")
            if not new_is_resting:
                self._log(f"New (shrunk) {side.upper()} quote @ {target_price}c was rejected (likely "
                           f"post_only cross) -- this side now has NO resting order until next cycle.")
                return

            self._log(f"Shrunk {side.upper()}: {size} @ {target_price}c "
                       f"(${size*price_dollars:.2f} notional, was {was}), {context}")
            return

        # --- NOT a shrink (fresh quote, reprice, or grow): PLACE NEW QUOTE
        # FIRST (old quote, if any, is still live right now) -- the usual
        # zero-coverage-gap-safe pattern. ---
        self._log(f"ATTEMPTING {action_word} {side.upper()}: {size} @ {target_price}c "
                   f"(${size*price_dollars:.2f} notional, was {was})")
        try:
            result = self.kc.place_limit_order(self.ticker, side=side, price_cents=target_price, count=size)
        except Exception as e:  # noqa: BLE001
            self._log(f"{action_word.upper()} PLACEMENT FAILED: {type(e).__name__}: {e} -- "
                       f"was attempting {side.upper()} {size} @ {target_price}c "
                       f"(${size*price_dollars:.2f} notional)")
            raise
        new_is_resting = (result.get("fill_count", "0.00") != "0.00"
                           or result.get("remaining_count", "0.00") != "0.00")

        if not new_is_resting:
            self._log(f"New {side.upper()} quote @ {target_price}c was rejected (likely post_only "
                       f"cross) — keeping old quote at {was} in place rather than leaving zero coverage.")
            return

        # --- New quote confirmed resting — NOW safe to pull the old one(s) ---
        if old_order_ids:
            self.kc.cancel_orders_batch(old_order_ids)

        self._log(f"Requoted {side.upper()}: {size} @ {target_price}c "
                   f"(${size*price_dollars:.2f} notional, was {was}), {context}")

    def manage_closing_quote(self, best_yes_bid: int | None, best_no_bid: int | None,
                              positions: list[dict] | None = None,
                              best_yes_ask: int | None = None, best_no_ask: int | None = None) -> None:
        held_side, qty_held, avg_cost_cents = self.get_position_info(positions)

        if held_side is None or qty_held <= 0:
            if self.closing_quote_state.get("order_id"):
                if not self.dry_run:
                    try:
                        self.kc.cancel_order(self.closing_quote_state["order_id"])
                    except Exception:  # noqa: BLE001
                        pass
                self._log("Closing quote cleared: no more inventory to close.")
            self.closing_quote_state["order_id"] = None
            self.closing_quote_state["price_cents"] = None
            self.closing_quote_state["side"] = None
            return

        # Sell AT the current best ask for the side we're actually holding
        # (not the bid) -- this joins the current best offer directly
        # rather than resting just above the highest bid, which can sit
        # well below where sellers are actually asking in a wide-spread
        # market. Still floored at avg_cost+1c so a fill never locks in a
        # guaranteed loss, even if the best ask happens to be below cost.
        best_ask_for_held = best_yes_ask if held_side == "yes" else best_no_ask
        cost_based_price = avg_cost_cents + 1
        ask_based_price = best_ask_for_held if best_ask_for_held is not None else cost_based_price
        sell_price_for_held = min(99, max(cost_based_price, ask_based_price))

        closing_side = "no" if held_side == "yes" else "yes"
        mechanical_price = self.clamp_price(100 - sell_price_for_held)
        count = int(qty_held)
        if count <= 0:
            return

        if (self.closing_quote_state.get("price_cents") is not None and self.closing_quote_state.get("order_id")
                and abs(mechanical_price - self.closing_quote_state["price_cents"]) < self.requote_threshold_cents):
            return

        profit_per_contract = sell_price_for_held - avg_cost_cents
        winner = "cost+1c" if cost_based_price >= ask_based_price else "best_ask"

        if self.dry_run:
            self._log(f"WOULD post closing quote: holding {qty_held:.0f} {held_side.upper()} "
                       f"@ avg_cost={avg_cost_cents}c -> {closing_side.upper()} @ {mechanical_price}c "
                       f"[{winner}], locks in >= {profit_per_contract}c/contract if filled")
            return

        old_id = self.closing_quote_state.get("order_id")

        # Cancel the OLD closing quote FIRST, then place the new one -- same
        # reasoning as the shrink case in maybe_requote_side: place-then-
        # cancel would momentarily need capital for both the old and new
        # closing order at once, which this account has repeatedly failed
        # on (insufficient_balance) even for small changes. Trade-off:
        # accepts a brief gap with no resting closing quote, in exchange for
        # actually being able to place the replacement when capital is tight.
        if old_id:
            try:
                self.kc.cancel_order(old_id)
            except Exception:  # noqa: BLE001
                pass

        self._log(f"ATTEMPTING closing quote: {closing_side.upper()} {count} @ {mechanical_price}c "
                   f"(closing {qty_held:.0f} held {held_side.upper()} @ avg_cost={avg_cost_cents}c, "
                   f"[{winner}], target profit >= {profit_per_contract}c/contract)")
        try:
            result = self.kc.place_limit_order(self.ticker, side=closing_side, price_cents=mechanical_price, count=count)
        except Exception as e:  # noqa: BLE001
            self._log(f"CLOSING QUOTE PLACEMENT FAILED (old closing quote already cancelled -- "
                       f"NO closing quote resting until next successful cycle): {type(e).__name__}: {e} -- "
                       f"was attempting {closing_side.upper()} {count} @ {mechanical_price}c "
                       f"(held {qty_held:.0f} {held_side.upper()}, position data: "
                       f"held_side={held_side}, qty_held={qty_held}, avg_cost_cents={avg_cost_cents})")
            self.closing_quote_state["order_id"] = None
            self.closing_quote_state["price_cents"] = None
            self.closing_quote_state["side"] = None
            raise
        new_is_resting = (result.get("fill_count", "0.00") != "0.00"
                           or result.get("remaining_count", "0.00") != "0.00")

        if not new_is_resting:
            self._log(f"Closing quote @ {mechanical_price}c ({closing_side.upper()}) rejected "
                       f"(post_only cross) -- NO closing quote resting until next cycle.")
            self.closing_quote_state["order_id"] = None
            self.closing_quote_state["price_cents"] = None
            self.closing_quote_state["side"] = None
            return

        self.closing_quote_state["order_id"] = result.get("order_id")
        self.closing_quote_state["price_cents"] = mechanical_price
        self.closing_quote_state["side"] = closing_side
        self._log(f"CLOSING QUOTE: holding {qty_held:.0f} {held_side.upper()} @ avg_cost={avg_cost_cents}c "
                   f"-> {closing_side.upper()} @ {mechanical_price}c [{winner}], "
                   f">= {profit_per_contract}c/contract if filled")

    # -----------------------------------------------------------------
    # MAIN CYCLE
    # -----------------------------------------------------------------
    def run_cycle(self, positions: list[dict] | None = None,
                  open_orders: list[dict] | None = None) -> dict:
        """One full iteration. Returns a summary dict (useful for logging
        when multiple Autoquoters run side by side later).

        positions: pass a pre-fetched list from kc.get_positions() when
        running many Autoquoters together (e.g. from an orchestrator loop)
        to avoid each instance independently re-fetching your WHOLE
        portfolio -- get_positions() returns everything, not just this
        ticker, so with N bots that's N redundant identical fetches per
        cycle if each one calls it itself. Leave as None for standalone/
        single-bot use, which fetches its own copy as before.

        open_orders: pass a pre-fetched list from kc.get_open_orders(ticker)
        for THIS ticker when running many bots together, to avoid the
        redundancy of get_my_resting_price() -- which makes its OWN fresh
        get_open_orders() HTTP call internally, with zero caching -- being
        called twice per bot (once for yes, once for no) AND again by an
        orchestrator-level stray-order scan on the same ticker in the same
        cycle. Sharing one fetch across all of that turned 100+ redundant
        HTTP calls per cycle into one per unique ticker (see
        resting_price_and_size()'s docstring). Leave as None for standalone
        use, which fetches its own copy as before.

        BUDGET: self.total_budget_dollars can be refreshed by the caller
        every cycle (see kalshi_treasury_quoter.py, which recomputes it
        from real account-wide FILLED exposure only, and applies the SAME
        ceiling to every tracked ticker -- not a divided/shared pool, since
        Kalshi doesn't lock capital for merely resting orders). See
        maybe_requote_side()'s docstring for the full reasoning.

        TWO-PASS ARCHITECTURE: this method handles market-making ONLY
        (the target/ITM side's quote, and pulling the OTM side). It does
        NOT call manage_closing_quote() -- that's now a deliberately
        SEPARATE pass, called by the orchestrator AFTER every bot's
        run_cycle() has completed for the cycle (see kalshi_treasury_
        quoter.py's run_one_cycle). Why: run_cycle's own OTM-pull logic
        used to conflict with manage_closing_quote() -- if a legitimate
        closing quote happened to be resting on the OTM side (e.g. a NO
        order closing out a held YES position while itm_side=="yes"),
        run_cycle would treat it as OTM garbage and cancel it, then
        manage_closing_quote() -- checking only its own stale internal
        state, not the live orderbook -- would think nothing changed and
        silently fail to re-place it. Confirmed in practice: a closing
        quote got destroyed exactly this way and never came back. Two
        fixes together close this: (1) the OTM-pull logic here explicitly
        skips any order matching the tracked closing_quote_state (see
        _is_tracked_closing_quote), so it can never cancel a real closing
        quote even within this same pass; (2) closing quotes are now
        managed in their own separate pass entirely, so they're never
        touched mid-market-making-cycle regardless.
        """
        if positions is None:
            positions = self.kc.get_positions()

        fair = self.price_fn()

        market = self.kc.best_market_prices(self.kc.get_orderbook(self.ticker))
        if market is not None:
            best_yes_bid, best_yes_ask = market
            best_no_bid, best_no_ask = 100 - best_yes_ask, 100 - best_yes_bid
        else:
            best_yes_bid = best_yes_ask = best_no_bid = best_no_ask = None

        target_yes = self.compute_target_price(fair["fair_yes_cents"], best_yes_bid, best_yes_ask)
        target_no = self.compute_target_price(fair["fair_no_cents"], best_no_bid, best_no_ask)
        itm_side = self.determine_itm_side(fair["spot"], fair["strike"])

        filled_exposure = self.get_filled_exposure_dollars(positions)

        if filled_exposure >= self.total_budget_dollars:
            self._log(f"Filled exposure (${filled_exposure:.2f}) at/over budget "
                       f"(${self.total_budget_dollars:.2f}) — pulling all quotes, no new ones.")
            if not self.dry_run:
                self.kc.cancel_all_orders(self.ticker)
            self.closing_quote_state["order_id"] = None
            self.closing_quote_state["price_cents"] = None
            self.closing_quote_state["side"] = None
            return {"ticker": self.ticker, "action": "budget_exhausted"}

        # ONE get_open_orders() call covers both yes AND no sides -- replaces
        # what used to be two separate get_my_resting_price() calls, each of
        # which made its OWN fresh get_open_orders() HTTP request internally
        # (that method has zero caching). Reuse a pre-fetched list from the
        # orchestrator if given (shared across this bot's own resting-price
        # check, maybe_requote_side's existing-order lookup below, AND the
        # orchestrator's stray-order scan -- see kalshi_treasury_quoter.py).
        if open_orders is None:
            open_orders = self.kc.get_open_orders(self.ticker)
        yes_price, yes_size = resting_price_and_size(open_orders, "yes")
        no_price, no_size = resting_price_and_size(open_orders, "no")
        yes_notional = (yes_price / 100.0 * yes_size) if yes_price else 0.0
        no_notional = (no_price / 100.0 * no_size) if no_price else 0.0

        maturity_hours = fair.get("hours_remaining")
        priced_maturity_hours = fair.get("priced_maturity_hours")
        maturity_str = f", maturity={maturity_hours:.1f}h" if maturity_hours is not None else ""

        if itm_side in (None, "yes"):
            self.maybe_requote_side("yes", target_yes, filled_exposure, no_notional,
                                     current_price=yes_price, current_size=yes_size,
                                     fair_price_cents=fair["fair_yes_cents"],
                                     spot=fair["spot"], strike=fair["strike"],
                                     hours_remaining=maturity_hours,
                                     priced_maturity_hours=priced_maturity_hours,
                                     best_bid=best_yes_bid, best_ask=best_yes_ask,
                                     open_orders=open_orders)
        elif yes_price is not None:
            if self._is_tracked_closing_quote("yes"):
                pass  # this IS the closing quote for a held NO position -- never OTM-pull it here
            elif self.dry_run:
                self._log(f"WOULD pull YES entirely (OTM). spot={fair['spot']:.5f}, "
                           f"strike={fair['strike']:.5f}{maturity_str}")
            else:
                self.kc.pull_side(self.ticker, "yes")
                self._log(f"Pulled YES entirely (OTM). spot={fair['spot']:.5f}, "
                           f"strike={fair['strike']:.5f}{maturity_str}")

        if itm_side in (None, "no"):
            self.maybe_requote_side("no", target_no, filled_exposure, yes_notional,
                                     current_price=no_price, current_size=no_size,
                                     fair_price_cents=fair["fair_no_cents"],
                                     spot=fair["spot"], strike=fair["strike"],
                                     hours_remaining=maturity_hours,
                                     priced_maturity_hours=priced_maturity_hours,
                                     best_bid=best_no_bid, best_ask=best_no_ask,
                                     open_orders=open_orders)
        elif no_price is not None:
            if self._is_tracked_closing_quote("no"):
                pass  # this IS the closing quote for a held YES position -- never OTM-pull it here.
                      # manage_closing_quote() (called SEPARATELY by the orchestrator, in its own
                      # pass, after every bot's market-making cycle has run) is the only thing that
                      # manages this order -- see run_cycle's docstring "TWO-PASS ARCHITECTURE" note.
            elif self.dry_run:
                self._log(f"WOULD pull NO entirely (OTM). spot={fair['spot']:.5f}, "
                           f"strike={fair['strike']:.5f}{maturity_str}")
            else:
                self.kc.pull_side(self.ticker, "no")
                self._log(f"Pulled NO entirely (OTM). spot={fair['spot']:.5f}, "
                           f"strike={fair['strike']:.5f}{maturity_str}")

        return {
            "ticker": self.ticker, "fair_yes_cents": fair["fair_yes_cents"],
            "fair_no_cents": fair["fair_no_cents"], "target_yes": target_yes,
            "target_no": target_no, "itm_side": itm_side,
        }

    def _is_tracked_closing_quote(self, side: str) -> bool:
        """True if `side` is exactly the side of the currently-tracked
        closing quote (manage_closing_quote() populates closing_quote_state
        with the side it placed on). Used by run_cycle's OTM-pull logic to
        avoid ever cancelling the closing quote -- that order is
        exclusively managed by manage_closing_quote(), called separately
        by the orchestrator (see run_cycle's docstring)."""
        return (self.closing_quote_state.get("order_id") is not None
                and self.closing_quote_state.get("side") == side)

    def run_forever(self, poll_interval_sec: int = 2, cancel_all_on_start: bool = True) -> None:
        """Blocking loop -- for running several Autoquoters CONCURRENTLY
        (the stated long-term goal), each instance needs its own thread/
        process; this method alone only serves one ticker per call."""
        if cancel_all_on_start and not self.dry_run:
            self.kc.cancel_all_orders(self.ticker)

        while True:
            try:
                self.run_cycle()
            except Exception as e:  # noqa: BLE001
                self._log(f"ERROR during cycle: {type(e).__name__}: {e}")
                self._log("FAIL-SAFE: pulling all quotes on this ticker...")
                if not self.dry_run:
                    try:
                        self.kc.cancel_all_orders(self.ticker)
                        self.closing_quote_state["order_id"] = None
                        self.closing_quote_state["price_cents"] = None
                        self.closing_quote_state["side"] = None
                    except Exception as cancel_err:  # noqa: BLE001
                        self._log(f"WARNING: fail-safe pull itself failed: {cancel_err}")
            time.sleep(poll_interval_sec)

    def shutdown(self) -> None:
        if not self.dry_run:
            self.kc.cancel_all_orders(self.ticker)
        self.closing_quote_state["order_id"] = None
        self.closing_quote_state["price_cents"] = None
        self.closing_quote_state["side"] = None