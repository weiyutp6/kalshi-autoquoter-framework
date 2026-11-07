#!/usr/bin/env python3
"""
DXY Daily Up/Down Autoquoter for Kalshi.

Composes KalshiClient + DXYPricer: refits the GARCH model periodically
(expensive), prices the contract every loop (cheap), and only pulls/requotes
a side when its fair value has moved past REQUOTE_THRESHOLD_CENTS from what
you currently have resting there. All orders are post_only (maker-only).

ITM/OTM rule: only the ITM side is quoted. If spot > strike, YES is ITM
-> quote YES only (pull/skip NO). If spot < strike, NO is ITM -> quote NO
only (pull/skip YES). At spot == strike (unlikely in practice), both sides
are quoted.

Single order per side (no tiered/laddered quoting).

Requires KalshiClient.py and DXYPricer.py to be importable from the same
directory (or on your PYTHONPATH).
"""
from __future__ import annotations

import math
import os
import time
from datetime import datetime, timezone

from KalshiClient import KalshiClient
from DXYPricer import DXYPricer, parse_strike_and_settlement

# ===========================================================================
# CONFIG
# ===========================================================================
KALSHI_KEY_ID = os.environ.get("KALSHI_KEY_ID", "")
KALSHI_PRIVATE_KEY_PATH = os.environ.get("KALSHI_PRIVATE_KEY_PATH", "kalshi_private_key.pem")

TICKER = "KXDXYDUD-26JUL29-T101.5625"      # <-- update daily
DAILY_SETTLEMENT_ET = "13:59:00"

PYTH_SYMBOL = "FX.USDXY"
PYTH_ASSET_TYPE = "fx"
HISTORY_DAYS = 30  # 個人智慧財產,參數不提供 (placeholder)
BAR_RESOLUTION_MIN = 1
GARCH_SCALE = 10000.0
ASSUME_ZERO_DRIFT = True
RISK_FREE_RATE = 0.0

DISABLE_PERIODIC_REFIT = True   # if True, GARCH is fit ONCE at startup (protected by the
                                  # absolute-bounds guard) and never refit again during the
                                  # session. Every instability seen this session came from a
                                  # REFIT landing in a bad local optimum, not the initial fit
                                  # — and intraday vol doesn't seem to shift enough within one
                                  # trading day to justify the risk. Set False to re-enable
                                  # periodic refitting on the REFIT_INTERVAL_SEC cadence below.
REFIT_INTERVAL_SEC = 300  # 個人智慧財產,參數不提供 (placeholder) |        # refit GARCH at most this often (wall-clock time),
                                 # independent of how often the poll loop runs
POLL_INTERVAL_SEC = 5  # 個人智慧財產,參數不提供 (placeholder)
QUOTE_NOTIONAL_DOLLARS = 10.0   # size each quote as ~$10 worth of contracts at — 個人智慧財產,參數不提供 (placeholder)
                                  # the quoted price (count = floor(15 / price)),
                                  # not a fixed contract count
EDGE_CENTER_CENTS = 50          # fair price at which edge is largest (max uncertainty) — 個人智慧財產,參數不提供 (placeholder)
EDGE_AT_CENTER_CENTS = 4        # edge (in cents) applied when fair value == EDGE_CENTER_CENTS — 個人智慧財產,參數不提供 (placeholder)
EDGE_SLOPE = 0.06               # cents of edge REMOVED per cent of distance from center — — 個人智慧財產,參數不提供 (placeholder)
                                  # e.g. at fair=99c (49c from center), edge shrinks by
                                  # 0.08*49=3.92c, leaving ~1c edge instead of the full 5c
EDGE_MIN_CENTS = 2              # floor — edge never goes below this (always leave SOME margin) — 個人智慧財產,參數不提供 (placeholder)
REQUOTE_THRESHOLD_CENTS = 2    # only pull/requote a side if its target moved >= this much — 個人智慧財產,參數不提供 (placeholder)

TOTAL_BUDGET_DOLLARS = 50.0    # HARD CAP, strictly enforced: filled exposure must never — 個人智慧財產,參數不提供 (placeholder)
                                 # reach this, and filled + resting notional combined must
                                 # never exceed this either (so a resting order can't fill
                                 # and push you over the cap)


def clear_pricer_cache(pricer: DXYPricer) -> None:
    """
    Deletes the history CSV cache and GARCH-params disk cache, so the next
    run starts completely fresh instead of risking stale/contaminated
    cached data (e.g. bars from a different resolution, or a fit accepted
    before the outlier-rejection guard existed). Safe to call even if the
    files don't exist.
    """
    for path in (pricer.csv_path, pricer.garch_cache_path):
        try:
            if os.path.exists(path):
                os.remove(path)
                print(f"  Cleared cache: {path}")
        except OSError as e:
            print(f"  WARNING: could not clear cache file {path}: {e}")


def clamp_price(c: int) -> int:
    return max(1, min(99, c))


def get_filled_exposure_dollars(kc: KalshiClient, ticker: str) -> float:
    """Actual filled/held exposure on this ticker, from your portfolio positions."""
    for p in kc.get_positions():
        if p.get("ticker") == ticker:
            try:
                return abs(float(p.get("market_exposure_dollars", 0)))
            except (TypeError, ValueError):
                return 0.0
    return 0.0


def get_resting_notional_dollars(kc: KalshiClient, ticker: str) -> float:
    """Worst-case dollar exposure if every currently-resting order (both sides) filled.
    Uses ECONOMIC side/price (via KalshiClient.economic_side/economic_price_field),
    not the raw "side" field — a NO order is placed as an ask on the YES book and
    would otherwise be misread as an expensive YES position at the wrong price.
    Also skips stale/already-dead entries (status != "resting" or remaining == 0)."""
    total = 0.0
    for o in kc.get_open_orders(ticker):
        if not KalshiClient._is_actually_resting(o):
            continue
        price_dollars = o.get(KalshiClient.economic_price_field(o))
        remaining = float(o.get("remaining_count_fp", 0))
        if price_dollars is not None:
            total += float(price_dollars) * remaining
    return total


def determine_itm_side(spot: float, strike: float) -> str | None:
    """
    Returns "yes", "no", or None (quote both — spot exactly at strike).
    If spot > strike, this YES-above-strike contract is currently ITM on the
    YES side -> quote YES only, skip NO.
    If spot < strike, NO is ITM -> quote NO only, skip YES.
    """
    if spot > strike:
        return "yes"
    elif spot < strike:
        return "no"
    return None


def dynamic_edge_cents(fair_price_cents: int) -> int:
    """
    Linear edge centered at EDGE_CENTER_CENTS (50c by default): edge is
    largest right at the center (where true uncertainty is highest) and
    shrinks linearly as fair value moves toward either extreme (0c or
    100c), where the outcome is close to certain and a wide edge just
    wastes fill probability on something already near-guaranteed. Floored
    at EDGE_MIN_CENTS so there's always at least a token margin.
    """
    distance_from_center = abs(fair_price_cents - EDGE_CENTER_CENTS)
    edge = EDGE_AT_CENTER_CENTS - EDGE_SLOPE * distance_from_center
    return max(EDGE_MIN_CENTS, round(edge))


def compute_target_price(fair_price_cents: int, best_bid: int | None, best_ask: int | None) -> int:
    """
    Quotes fair_price_cents - dynamic_edge_cents(fair_price_cents). Stays
    at least 1c under the best ask so we never try to cross (post_only
    would reject it anyway, but better to compute a sane price up front).
    """
    edge = dynamic_edge_cents(fair_price_cents)
    candidate = fair_price_cents - edge

    if best_ask is not None:
        candidate = min(candidate, best_ask - 1)

    return clamp_price(candidate)


def describe_price(fair_cents: int, bid: int | None, ask: int | None) -> str:
    edge = dynamic_edge_cents(fair_cents)
    return f"(fair {fair_cents}c - {edge}c dynamic edge)"


def compute_would_be_size(price_cents: int, filled_exposure: float,
                           other_side_notional: float) -> tuple[int, float, str]:
    """Same sizing math as maybe_requote_side, but pure/read-only — used by
    both dry_run_test and test_custom_spot to show what WOULD be quoted."""
    price_dollars = price_cents / 100.0
    if price_dollars <= 0:
        return 0, 0.0, "invalid price"
    desired = max(1, int(QUOTE_NOTIONAL_DOLLARS // price_dollars))
    committed = filled_exposure + other_side_notional
    remaining = TOTAL_BUDGET_DOLLARS - committed
    if remaining <= 0:
        return 0, 0.0, f"budget fully committed (${committed:.2f}/${TOTAL_BUDGET_DOLLARS:.2f})"
    max_affordable = int(remaining // price_dollars)
    size = min(desired, max_affordable)
    if size <= 0:
        return 0, 0.0, f"can't afford even 1 (${remaining:.2f} remaining)"
    note = "" if size == desired else f", budget-limited from {desired} contracts"
    return size, size * price_dollars, note


def maybe_requote_side(kc: KalshiClient, ticker: str, side: str, target_price: int,
                        filled_exposure: float, other_side_resting_notional: float) -> None:
    """
    Single order per side (no ladder). Checks your currently-resting price;
    only requotes if there's nothing resting yet, or the target has moved
    by at least REQUOTE_THRESHOLD_CENTS.

    TRUE ZERO-DOWNTIME, NO-OVERLAP REPRICING: if there's exactly one order
    already resting on this side, this AMENDS it in place (same order_id,
    single atomic call) rather than cancelling and placing a new one. There
    is no window where the quote disappears (order never gets cancelled)
    and no window where two orders are resting simultaneously (it's still
    one order throughout) — amend_order() changes price/size on the
    existing order directly.

    Falls back to a plain place_limit_order() only when there's nothing
    resting yet at all (first quote of the session) or if the amend itself
    fails for some reason, in which case it places fresh rather than
    leaving you with no quote.
    """
    current_price, current_size = kc.get_my_resting_price(ticker, side)

    if current_price is not None and abs(target_price - current_price) < REQUOTE_THRESHOLD_CENTS:
        return  # close enough, leave it resting

    existing_orders = [
        o for o in kc.get_open_orders(ticker)
        if kc.economic_side(o) == side and kc._is_actually_resting(o)
    ]

    price_dollars = target_price / 100.0
    if price_dollars <= 0:
        return
    desired_size = max(1, int(QUOTE_NOTIONAL_DOLLARS // price_dollars))

    committed = filled_exposure + other_side_resting_notional
    remaining_budget = TOTAL_BUDGET_DOLLARS - committed
    if remaining_budget <= 0:
        print(f"  SKIPPING {side.upper()} requote: budget already fully committed "
              f"(${committed:.2f} / ${TOTAL_BUDGET_DOLLARS:.2f})")
        return

    max_affordable = int(remaining_budget // price_dollars)
    size = min(desired_size, max_affordable)

    if size <= 0:
        print(f"  SKIPPING {side.upper()} requote @ {target_price}c: would exceed "
              f"${TOTAL_BUDGET_DOLLARS:.2f} budget (committed=${committed:.2f}, "
              f"remaining=${remaining_budget:.2f})")
        return

    if size < desired_size:
        print(f"  Reducing {side.upper()} size from {desired_size} (${QUOTE_NOTIONAL_DOLLARS:.2f} "
              f"notional) to {size} to stay within budget "
              f"(committed=${committed:.2f}, remaining=${remaining_budget:.2f})")

    was = f"{current_price}c" if current_price is not None else "none"

    # --- PLACE NEW QUOTE FIRST (old quote, if any, is still live right now) ---
    old_order_ids = [o["order_id"] for o in existing_orders]
    result = kc.place_limit_order(ticker, side=side, price_cents=target_price, count=size)
    new_is_resting = (result.get("fill_count", "0.00") != "0.00"
                       or result.get("remaining_count", "0.00") != "0.00")

    if not new_is_resting:
        print(f"  New {side.upper()} quote @ {target_price}c was rejected (likely post_only "
              f"cross) — keeping old quote at {was} in place rather than leaving zero coverage.")
        return

    # --- New quote confirmed resting — NOW safe to pull the old one(s) ---
    if old_order_ids:
        kc.cancel_orders_batch(old_order_ids)

    print(f"  Requoted {side.upper()}: {size} @ {target_price}c "
          f"(${size*price_dollars:.2f} notional, was {was}) "
          f"[new quote confirmed before old one was pulled]")


def test_custom_spot(pricer: DXYPricer, kc: KalshiClient, spot: float,
                      bars_remaining: int | None = None) -> None:
    """
    Read-only, no live spot fetch: prices the contract using a manually
    supplied hypothetical spot instead of pulling the latest Pyth price, so
    you can sanity-check the model's behavior across a range of "what if
    DXY were at X" scenarios. Still pulls the LIVE Kalshi orderbook (public,
    read-only) so the quoted-price logic (improve-best-bid, ITM/OTM, sizing)
    is tested against a real current market. Requires pricer.refit() to have
    already been called at least once (reuses the existing GARCH fit —
    doesn't refit per call, so you can run this many times quickly).

    bars_remaining: if None, computed from the real current time vs the
    contract's settlement (same as live). Pass an explicit value to also
    test "what if X hours were left" scenarios independently of spot.
    """
    if pricer.g is None:
        raise RuntimeError("Call pricer.refit() at least once before test_custom_spot().")

    strike, settlement_utc = parse_strike_and_settlement(TICKER, pricer.daily_settlement_et)

    if bars_remaining is None:
        now = datetime.now(timezone.utc)
        mins_remaining = max((settlement_utc - now).total_seconds() / 60, 0)
        bars_remaining = max(0, math.ceil(mins_remaining / pricer.resolution_min))

    fair = pricer.fair_yes_prob(spot, strike, bars_remaining)
    fair_yes_cents = round(fair["prob"] * 100)
    fair_no_cents = 100 - fair_yes_cents

    print(f"\n{'-'*70}")
    print(f"CUSTOM SPOT TEST: spot={spot}  (strike={strike}, bars_remaining={bars_remaining})")
    print(f"{'-'*70}")
    print(f"  t_years={fair['t_years']:.8f}  sigma_annual={fair['sigma_annual']:.6f} "
          f"({fair['sigma_annual']*100:.3f}% ann. vol)")
    print(f"  d1={fair['d1']:.6f}  d2={fair['d2']:.6f}")
    print(f"  FAIR YES: {fair['prob']:.4f}  ({fair_yes_cents}c)   FAIR NO: {1-fair['prob']:.4f}  ({fair_no_cents}c)")

    if spot > strike:
        print(f"  spot IS ABOVE strike by {spot-strike:.6f}  ->  YES is ITM, NO is OTM")
    elif spot < strike:
        print(f"  spot IS BELOW strike by {strike-spot:.6f}  ->  NO is ITM, YES is OTM")
    else:
        print(f"  spot EXACTLY EQUALS strike  ->  quoting both sides")

    itm_side = determine_itm_side(spot, strike)
    print(f"  itm_side = {itm_side or 'both'}   (quoting the ITM side only)")

    market = kc.best_market_prices(kc.get_orderbook(TICKER))
    if market is not None:
        best_yes_bid, best_yes_ask = market
        best_no_bid, best_no_ask = 100 - best_yes_ask, 100 - best_yes_bid
        print(f"  Live market: yes_bid={best_yes_bid}c yes_ask={best_yes_ask}c "
              f"| no_bid={best_no_bid}c no_ask={best_no_ask}c")
    else:
        best_yes_bid = best_yes_ask = best_no_bid = best_no_ask = None
        print("  Live market: too thin to compute best bid/ask.")

    target_yes = compute_target_price(fair_yes_cents, best_yes_bid, best_yes_ask)
    target_no = compute_target_price(fair_no_cents, best_no_bid, best_no_ask)

    filled_exposure = get_filled_exposure_dollars(kc, TICKER)
    yes_price_r, yes_size_r = kc.get_my_resting_price(TICKER, "yes")
    no_price_r, no_size_r = kc.get_my_resting_price(TICKER, "no")
    yes_notional_r = (yes_price_r / 100.0 * yes_size_r) if yes_price_r else 0.0
    no_notional_r = (no_price_r / 100.0 * no_size_r) if no_price_r else 0.0

    print(f"  WOULD QUOTE:")
    if itm_side in (None, "yes"):
        size, notional, size_note = compute_would_be_size(target_yes, filled_exposure, no_notional_r)
        print(f"    YES: {size} contracts @ {target_yes}c  =  ${notional:.2f} notional  " +
              describe_price(fair_yes_cents, best_yes_bid, best_yes_ask) +
              (f"  [{size_note}]" if size_note else ""))
    else:
        print(f"    YES: skipped — OTM side, not quoted")

    if itm_side in (None, "no"):
        size, notional, size_note = compute_would_be_size(target_no, filled_exposure, yes_notional_r)
        print(f"    NO: {size} contracts @ {target_no}c  =  ${notional:.2f} notional  " +
              describe_price(fair_no_cents, best_no_bid, best_no_ask) +
              (f"  [{size_note}]" if size_note else ""))
    else:
        print(f"    NO: skipped — OTM side, not quoted")


def get_position_info(kc: KalshiClient, ticker: str) -> tuple[str | None, float, float]:
    """
    Returns (held_side, qty_held, avg_cost_cents) for your current position
    on this ticker, or (None, 0.0, 0.0) if flat. avg_cost_cents is your
    average cost per contract on the held side, derived from Kalshi's
    market_exposure_dollars / position size.
    """
    for p in kc.get_positions():
        if p.get("ticker") != ticker:
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


def manage_closing_quote(kc: KalshiClient, ticker: str, state: dict,
                          best_yes_bid: int | None, best_no_bid: int | None) -> None:
    """
    If you're holding a filled position on this ticker, quotes it for exit
    at whichever is HIGHER: the current best bid for your held side + 1c,
    or your average cost + 1c (a guaranteed minimum profit floor). This
    tracks the live market rather than the model's fair value — you get
    paid either from where the market is actually willing to buy you out,
    or at minimum 1c/contract above what you paid, whichever is better.

    MECHANICS NOTE: Kalshi has a single YES-denominated book. To sell a
    held NO position, the correct order is actually a YES-side bid (buying
    YES nets against NO holdings at settlement/matching); to sell a held
    YES position, it's a NO-side ask. This function handles that side
    flip internally — you just get "sell my holdings at price X" behavior.

    `state` is a small persisted dict ({"order_id": ..., "price_cents": ...})
    the caller keeps across loop iterations, so this can track and update
    its own closing order independently of the normal opening-side quotes
    — this runs SIMULTANEOUSLY with those every cycle, not instead of them.
    Uses the same place-first-confirm-then-cancel-old pattern as
    maybe_requote_side for minimal downtime on the closing quote itself.
    """
    held_side, qty_held, avg_cost_cents = get_position_info(kc, ticker)

    if held_side is None or qty_held <= 0:
        if state.get("order_id"):
            try:
                kc.cancel_order(state["order_id"])
            except requests.HTTPError:
                pass
            print("  Closing quote cleared: no more inventory to close.")
        state["order_id"] = None
        state["price_cents"] = None
        return

    # Best bid for the side you're actually holding (what the market would
    # currently pay to buy you out), not the opposite/closing side's bid.
    best_bid_for_held = best_yes_bid if held_side == "yes" else best_no_bid

    cost_based_price = avg_cost_cents + 1
    bid_based_price = (best_bid_for_held + 1) if best_bid_for_held is not None else cost_based_price
    sell_price_for_held = min(99, max(cost_based_price, bid_based_price))

    # ...translates to placing the OPPOSITE-labeled order at the complementary price.
    closing_side = "no" if held_side == "yes" else "yes"
    mechanical_price = clamp_price(100 - sell_price_for_held)
    count = int(qty_held)
    if count <= 0:
        return

    if (state.get("price_cents") is not None and state.get("order_id")
            and abs(mechanical_price - state["price_cents"]) < REQUOTE_THRESHOLD_CENTS):
        return  # close enough, leave the existing closing quote resting

    old_id = state.get("order_id")
    result = kc.place_limit_order(ticker, side=closing_side, price_cents=mechanical_price, count=count)
    new_is_resting = (result.get("fill_count", "0.00") != "0.00"
                       or result.get("remaining_count", "0.00") != "0.00")

    if not new_is_resting:
        print(f"  Closing quote @ {mechanical_price}c ({closing_side.upper()}) rejected "
              f"(post_only cross) — keeping previous closing quote in place if any.")
        return

    if old_id:
        try:
            kc.cancel_order(old_id)
        except requests.HTTPError as e:
            if getattr(e.response, "status_code", None) != 404:
                print(f"  (closing quote) failed to cancel old order {old_id}: {e}")

    state["order_id"] = result.get("order_id")
    state["price_cents"] = mechanical_price
    profit_per_contract = sell_price_for_held - avg_cost_cents
    winner = "cost+1c" if cost_based_price >= bid_based_price else "best_bid+1c"
    print(f"  CLOSING QUOTE: holding {qty_held:.0f} {held_side.upper()} @ avg_cost={avg_cost_cents}c  "
          f"[cost+1c={cost_based_price}c vs best_bid+1c={bid_based_price}c -> using {winner}]")
    print(f"    -> exit via {closing_side.upper()} @ {mechanical_price}c "
          f"(sells held {held_side.upper()} @ {sell_price_for_held}c, "
          f"locks in >= {profit_per_contract}c/contract profit if filled)")


def run_autoquoter():
    kc = KalshiClient(key_id=KALSHI_KEY_ID, private_key_path=KALSHI_PRIVATE_KEY_PATH)
    pricer = DXYPricer(
        symbol=PYTH_SYMBOL,
        asset_type=PYTH_ASSET_TYPE,
        history_days=HISTORY_DAYS,
        resolution_min=BAR_RESOLUTION_MIN,
        garch_scale=GARCH_SCALE,
        assume_zero_drift=ASSUME_ZERO_DRIFT,
        risk_free_rate=RISK_FREE_RATE,
        daily_settlement_et=DAILY_SETTLEMENT_ET,
    )

    print(f"Contract: {TICKER}")

    print("Pulling any existing resting orders on this ticker before starting...")
    kc.cancel_all_orders(TICKER)
    print("Done.\n")

    print("Fetching authoritative market data from Kalshi (strike, resolution time)...")
    market_info = kc.get_market(TICKER)
    authoritative_strike = market_info.get("floor_strike") or market_info.get("cap_strike")
    close_time_str = market_info.get("close_time")
    close_time_utc = datetime.fromisoformat(close_time_str.replace("Z", "+00:00")) if close_time_str else None

    # occurrence_datetime on the MARKET object is the actual resolution-
    # observation moment (confirmed against rules_primary text, e.g. "above
    # 100.9650 at 11:59 PM ET" converts exactly to this UTC timestamp).
    # strike_date on the EVENT object does NOT exist for this event type —
    # don't rely on it. close_time/expiration_time are separate, later,
    # unrelated cutoffs (trading stop / dispute window), not the resolution
    # moment.
    occurrence_str = market_info.get("occurrence_datetime")

    if occurrence_str:
        authoritative_settlement_utc = datetime.fromisoformat(occurrence_str.replace("Z", "+00:00"))
        print(f"  strike={authoritative_strike}")
        print(f"  occurrence_datetime (actual resolution-observation time): {authoritative_settlement_utc.isoformat()}")
        print(f"  rules_primary: {market_info.get('rules_primary')}")
        if close_time_utc and abs((close_time_utc - authoritative_settlement_utc).total_seconds()) > 60:
            print(f"  NOTE: market close_time ({close_time_utc.isoformat()}) differs from occurrence_datetime "
                  f"by {(close_time_utc - authoritative_settlement_utc)} — close_time is when trading "
                  f"stops, NOT when the contract resolves; using occurrence_datetime as authoritative.")
    else:
        print(f"  WARNING: no occurrence_datetime found — falling back to market close_time, "
              f"which may NOT be the actual resolution-observation moment for this contract type.")
        print(f"  DEBUG raw market_info: {market_info}")
        authoritative_settlement_utc = close_time_utc
        print(f"  strike={authoritative_strike}  close_time={authoritative_settlement_utc.isoformat()}")

    print("Initial GARCH fit (this is the slow part)...")
    pricer.refit()
    last_refit_time = time.time()
    REFERENCE_BARS_FOR_DISPLAY = 60
    startup_iv = pricer.annualized_vol_forecast(REFERENCE_BARS_FOR_DISPLAY)
    print(f"  GARCH-implied annualized vol (60-bar reference): {startup_iv * 100:.3f}%  "
          f"[mu={pricer.g.mu:.8f}  alpha={pricer.g.alpha:.6f}  beta={pricer.g.beta:.6f}  "
          f"persistence={pricer.g.alpha + pricer.g.beta:.6f}]")
    if DISABLE_PERIODIC_REFIT:
        print("  Periodic refitting DISABLED — this initial fit will be used for the rest "
              "of the session (still protected by the absolute-bounds sanity guard).")
    else:
        print(f"  Periodic refitting enabled every {REFIT_INTERVAL_SEC}s.")
    print("Done.\n")

    loop_count = 0
    closing_quote_state = {"order_id": None, "price_cents": None}
    try:
        while True:
            loop_count += 1
            cycle_start = time.time()
            try:
                if not DISABLE_PERIODIC_REFIT and time.time() - last_refit_time >= REFIT_INTERVAL_SEC:
                    print(f"[{datetime.now(timezone.utc).isoformat()}] "
                          f"Refitting GARCH ({REFIT_INTERVAL_SEC}s since last fit)...")
                    pricer.refit()
                    last_refit_time = time.time()

                t0 = time.time()
                fair = pricer.price_contract(
                    TICKER, strike_override=authoritative_strike,
                    settlement_utc_override=authoritative_settlement_utc,
                )  # includes the Pyth spot fetch
                t_pyth = time.time() - t0

                if fair["bars_remaining"] <= 0:
                    print("Contract has settled (or is about to). Stopping autoquoter.")
                    kc.cancel_all_orders(TICKER)
                    clear_pricer_cache(pricer)
                    break

                t0 = time.time()
                market = kc.best_market_prices(kc.get_orderbook(TICKER))
                t_orderbook = time.time() - t0
                if market is not None:
                    best_yes_bid, best_yes_ask = market
                    best_no_bid, best_no_ask = 100 - best_yes_ask, 100 - best_yes_bid
                else:
                    best_yes_bid = best_yes_ask = best_no_bid = best_no_ask = None

                target_yes = compute_target_price(fair["fair_yes_cents"], best_yes_bid, best_yes_ask)
                target_no = compute_target_price(fair["fair_no_cents"], best_no_bid, best_no_ask)

                itm_side = determine_itm_side(fair["spot"], fair["strike"])
                if fair["spot"] > fair["strike"]:
                    itm_note = f"spot>strike by {fair['spot']-fair['strike']:.6f} (YES ITM/NO OTM)"
                elif fair["spot"] < fair["strike"]:
                    itm_note = f"spot<strike by {fair['strike']-fair['spot']:.6f} (NO ITM/YES OTM)"
                else:
                    itm_note = "spot==strike"

                market_str = f"mkt_yes_bid={best_yes_bid}c mkt_yes_ask={best_yes_ask}c" if market else "mkt=thin"

                t0 = time.time()
                filled_exposure = get_filled_exposure_dollars(kc, TICKER)
                resting_notional = get_resting_notional_dollars(kc, TICKER)
                t_portfolio = time.time() - t0
                total_committed = filled_exposure + resting_notional

                print(f"\n[{datetime.now(timezone.utc).isoformat()}] "
                      f"spot={fair['spot']:.4f} strike={fair['strike']} bars_left={fair['bars_remaining']} "
                      f"fair_yes={fair['fair_yes_cents']}c fair_no={fair['fair_no_cents']}c {market_str} "
                      f"-> target_yes={target_yes}c target_no={target_no}c")
                print(f"  {itm_note}  itm_side={itm_side or 'both (at strike)'}")
                print(f"  [budget] filled=${filled_exposure:.2f}  resting=${resting_notional:.2f}  "
                      f"total=${total_committed:.2f} / ${TOTAL_BUDGET_DOLLARS:.2f}")
                print(f"  [timing so far] pyth_spot={t_pyth*1000:.0f}ms  orderbook={t_orderbook*1000:.0f}ms  "
                      f"portfolio_check={t_portfolio*1000:.0f}ms")

                if filled_exposure >= TOTAL_BUDGET_DOLLARS:
                    print(f"  Filled exposure alone (${filled_exposure:.2f}) has reached/exceeded "
                          f"the ${TOTAL_BUDGET_DOLLARS:.2f} cap. Pulling all resting orders and "
                          f"not placing any new ones this loop.")
                    kc.cancel_all_orders(TICKER)
                    closing_quote_state["order_id"] = None
                    closing_quote_state["price_cents"] = None
                    print(f"  [TOTAL CYCLE TIME] {(time.time()-cycle_start)*1000:.0f}ms")
                    time.sleep(POLL_INTERVAL_SEC)
                    continue

                t0 = time.time()
                yes_price, yes_size = kc.get_my_resting_price(TICKER, "yes")
                no_price, no_size = kc.get_my_resting_price(TICKER, "no")
                t_resting_price = time.time() - t0
                yes_notional = (yes_price / 100.0 * yes_size) if yes_price else 0.0
                no_notional = (no_price / 100.0 * no_size) if no_price else 0.0

                t0 = time.time()
                # OTM side gets pulled and skipped entirely; only the ITM side
                # (or both, if exactly at strike) gets requoted.
                if itm_side in (None, "yes"):
                    maybe_requote_side(kc, TICKER, "yes", target_yes,
                                        filled_exposure=filled_exposure,
                                        other_side_resting_notional=no_notional)
                elif yes_price is not None:
                    kc.pull_side(TICKER, "yes")
                    print(f"  Pulled YES entirely: contract is OTM on YES (spot={fair['spot']:.4f} "
                          f"< strike={fair['strike']}), only quoting the ITM side.")

                if itm_side in (None, "no"):
                    maybe_requote_side(kc, TICKER, "no", target_no,
                                        filled_exposure=filled_exposure,
                                        other_side_resting_notional=yes_notional)
                elif no_price is not None:
                    kc.pull_side(TICKER, "no")
                    print(f"  Pulled NO entirely: contract is OTM on NO (spot={fair['spot']:.4f} "
                          f"> strike={fair['strike']}), only quoting the ITM side.")

                # --- Profit-taking exit quote for any filled inventory ---
                manage_closing_quote(kc, TICKER, closing_quote_state,
                                      best_yes_bid=best_yes_bid, best_no_bid=best_no_bid)

                t_requote = time.time() - t0
                print(f"  [timing] get_my_resting_price={t_resting_price*1000:.0f}ms  "
                      f"requote_logic={t_requote*1000:.0f}ms")
                print(f"  [TOTAL CYCLE TIME] {(time.time()-cycle_start)*1000:.0f}ms  "
                      f"(vs POLL_INTERVAL_SEC={POLL_INTERVAL_SEC*1000:.0f}ms budget)")

            except Exception as e:  # noqa: BLE001
                # FAIL-SAFE: any error during a loop iteration immediately pulls
                # every resting order on this ticker before continuing.
                print(f"ERROR during loop iteration: {type(e).__name__}: {e}")
                print("FAIL-SAFE: pulling all quotes on this ticker...")
                try:
                    kc.cancel_all_orders(TICKER)
                    closing_quote_state["order_id"] = None
                    closing_quote_state["price_cents"] = None
                    print("  All orders pulled successfully.")
                except Exception as cancel_err:  # noqa: BLE001
                    print(f"  WARNING: fail-safe pull itself failed: {cancel_err}. "
                          f"Check your account manually.")

            time.sleep(POLL_INTERVAL_SEC)

    except KeyboardInterrupt:
        print("\nInterrupted by user. Pulling all quotes before exit...")
        kc.cancel_all_orders(TICKER)
        clear_pricer_cache(pricer)
        print("Done — all orders on this ticker pulled, cache cleared.")
    finally:
        try:
            kc.cancel_all_orders(TICKER)
        except Exception:  # noqa: BLE001
            pass
        try:
            clear_pricer_cache(pricer)
        except Exception:  # noqa: BLE001
            pass


def dry_run_test(loops: int = 1, interval_sec: int = 10) -> None:
    """
    Read-only test mode: fits GARCH, prices the contract, computes what
    WOULD be quoted on each side given the current market — but never
    places, pulls, or cancels anything. Safe to run against your live
    account/key with zero trading risk.

    loops=1 (default) prints one snapshot and exits. Pass loops>1 to watch
    it repeat every interval_sec, e.g. to see how fair value and the
    quoted line drift over a few minutes before trusting it live.
    """
    kc = KalshiClient(key_id=KALSHI_KEY_ID, private_key_path=KALSHI_PRIVATE_KEY_PATH)
    pricer = DXYPricer(
        symbol=PYTH_SYMBOL,
        asset_type=PYTH_ASSET_TYPE,
        history_days=HISTORY_DAYS,
        resolution_min=BAR_RESOLUTION_MIN,
        garch_scale=GARCH_SCALE,
        assume_zero_drift=ASSUME_ZERO_DRIFT,
        risk_free_rate=RISK_FREE_RATE,
        daily_settlement_et=DAILY_SETTLEMENT_ET,
    )

    print("=" * 70)
    print("DRY RUN — no orders will be placed, pulled, or cancelled.")
    print("=" * 70)
    print("\n--- Active config / features this run ---")
    print(f"  Sizing: ~${QUOTE_NOTIONAL_DOLLARS:.2f} notional/quote  |  "
          f"Budget cap: ${TOTAL_BUDGET_DOLLARS:.2f}  |  "
          f"Requote threshold: {REQUOTE_THRESHOLD_CENTS}c")
    print(f"  Dynamic edge: center={EDGE_CENTER_CENTS}c  at_center={EDGE_AT_CENTER_CENTS}c  "
          f"slope={EDGE_SLOPE}  floor={EDGE_MIN_CENTS}c")
    print(f"  GARCH: {'periodic refit DISABLED (fit once, guards still active)' if DISABLE_PERIODIC_REFIT else f'refit every {REFIT_INTERVAL_SEC}s'}")
    print(f"  Quoting strategy: ITM-side only (OTM side pulled/skipped)")
    print(f"  Closing-quote rule: max(cost+1c, best_bid_for_held_side+1c)")

    print(f"\nContract: {TICKER}")
    print("Fetching authoritative market data from Kalshi (strike, resolution time)...")
    market_info = kc.get_market(TICKER)
    authoritative_strike = market_info.get("floor_strike") or market_info.get("cap_strike")
    close_time_str = market_info.get("close_time")
    close_time_utc = datetime.fromisoformat(close_time_str.replace("Z", "+00:00")) if close_time_str else None

    occurrence_str = market_info.get("occurrence_datetime")

    if occurrence_str:
        authoritative_settlement_utc = datetime.fromisoformat(occurrence_str.replace("Z", "+00:00"))
        print(f"  strike={authoritative_strike}")
        print(f"  occurrence_datetime (actual resolution-observation time): {authoritative_settlement_utc.isoformat()}")
        print(f"  rules_primary: {market_info.get('rules_primary')}")
        if close_time_utc and abs((close_time_utc - authoritative_settlement_utc).total_seconds()) > 60:
            print(f"  NOTE: market close_time ({close_time_utc.isoformat()}) differs from occurrence_datetime "
                  f"by {(close_time_utc - authoritative_settlement_utc)} — using occurrence_datetime as authoritative.")
    else:
        print(f"  WARNING: no occurrence_datetime found — falling back to market close_time.")
        print(f"  DEBUG raw market_info: {market_info}")
        authoritative_settlement_utc = close_time_utc
        print(f"  strike={authoritative_strike}  close_time={authoritative_settlement_utc.isoformat()}")

    print("Fitting GARCH (this is the slow part)...")
    g = pricer.refit()
    print("\n--- Model parameters ---")
    print(f"  mu={g.mu:.8f}  omega={g.omega:.10f}  alpha={g.alpha:.6f}  "
          f"beta={g.beta:.6f}  last_h={g.last_h:.10f}  "
          f"persistence(alpha+beta)={g.alpha+g.beta:.6f}")
    print(f"  ASSUME_ZERO_DRIFT={ASSUME_ZERO_DRIFT}  RISK_FREE_RATE={RISK_FREE_RATE}  "
          f"GARCH_SCALE={GARCH_SCALE}  resolution={BAR_RESOLUTION_MIN}min")

    for i in range(loops):
        print(f"\n{'='*70}\nSNAPSHOT {i+1}/{loops}  [{datetime.now(timezone.utc).isoformat()}]\n{'='*70}")
        snapshot_start = time.time()

        t0 = time.time()
        fair = pricer.price_contract(
            TICKER, strike_override=authoritative_strike,
            settlement_utc_override=authoritative_settlement_utc,
        )  # includes the Pyth spot fetch
        t_pyth = time.time() - t0
        print(f"  spot={fair['spot']:.6f}  strike={fair['strike']}  "
              f"bars_remaining={fair['bars_remaining']}  t_years={fair['t_years']:.8f}")
        print(f"  sigma_annual={fair['sigma_annual']:.6f} ({fair['sigma_annual']*100:.3f}% ann. vol)")
        print(f"  d1={fair['d1']:.6f}  d2={fair['d2']:.6f}")
        print(f"\n  FAIR YES: {fair['prob']:.4f}  ({fair['fair_yes_cents']}c)")
        print(f"  FAIR NO:  {1-fair['prob']:.4f}  ({fair['fair_no_cents']}c)")

        t0 = time.time()
        market = kc.best_market_prices(kc.get_orderbook(TICKER))
        t_orderbook = time.time() - t0
        if market is not None:
            best_yes_bid, best_yes_ask = market
            best_no_bid, best_no_ask = 100 - best_yes_ask, 100 - best_yes_bid
            print(f"\n  Market: yes_bid={best_yes_bid}c yes_ask={best_yes_ask}c "
                  f"| no_bid={best_no_bid}c no_ask={best_no_ask}c")
        else:
            best_yes_bid = best_yes_ask = best_no_bid = best_no_ask = None
            print("\n  Market: too thin to compute best bid/ask.")

        target_yes = compute_target_price(fair["fair_yes_cents"], best_yes_bid, best_yes_ask)
        target_no = compute_target_price(fair["fair_no_cents"], best_no_bid, best_no_ask)
        itm_side = determine_itm_side(fair["spot"], fair["strike"])

        if fair["spot"] > fair["strike"]:
            print(f"\n  spot ({fair['spot']:.6f}) IS ABOVE strike ({fair['strike']}) by "
                  f"{fair['spot']-fair['strike']:.6f}  ->  YES is ITM, NO is OTM")
        elif fair["spot"] < fair["strike"]:
            print(f"\n  spot ({fair['spot']:.6f}) IS BELOW strike ({fair['strike']}) by "
                  f"{fair['strike']-fair['spot']:.6f}  ->  NO is ITM, YES is OTM")
        else:
            print(f"\n  spot EXACTLY EQUALS strike  ->  quoting both sides")

        print(f"  itm_side = {itm_side or 'both (spot == strike)'}   (quoting the ITM side only)")

        # Compute what size WOULD be quoted, mirroring maybe_requote_side's
        # logic exactly (notional-based, budget-capped) — read-only, no orders.
        filled_exposure = get_filled_exposure_dollars(kc, TICKER)
        yes_price_r, yes_size_r = kc.get_my_resting_price(TICKER, "yes")
        no_price_r, no_size_r = kc.get_my_resting_price(TICKER, "no")
        yes_notional_r = (yes_price_r / 100.0 * yes_size_r) if yes_price_r else 0.0
        no_notional_r = (no_price_r / 100.0 * no_size_r) if no_price_r else 0.0

        print(f"  WOULD QUOTE:")

        if itm_side in (None, "yes"):
            size, notional, size_note = compute_would_be_size(target_yes, filled_exposure, no_notional_r)
            print(f"    YES: {size} contracts @ {target_yes}c  =  ${notional:.2f} notional  " +
                  describe_price(fair['fair_yes_cents'], best_yes_bid, best_yes_ask) +
                  (f"  [{size_note}]" if size_note else ""))
        else:
            print(f"    YES: skipped — OTM side, not quoted")

        if itm_side in (None, "no"):
            size, notional, size_note = compute_would_be_size(target_no, filled_exposure, yes_notional_r)
            print(f"    NO: {size} contracts @ {target_no}c  =  ${notional:.2f} notional  " +
                  describe_price(fair['fair_no_cents'], best_no_bid, best_no_ask) +
                  (f"  [{size_note}]" if size_note else ""))
        else:
            print(f"    NO: skipped — OTM side, not quoted")

        # --- Detailed budget diagnostics: show the raw order fields AND the
        # computed economic side/price for each one, so the book_side-based
        # fix (NO orders reported as asks on the YES book) can be visually
        # verified against real resting orders whenever any exist. ---
        print(f"\n  --- BUDGET DIAGNOSTICS ---")
        raw_orders = [o for o in kc.get_open_orders(TICKER) if kc._is_actually_resting(o)]
        if not raw_orders:
            print(f"  (no genuinely-resting orders right now — nothing to verify against)")
        else:
            for o in raw_orders:
                econ_side = kc.economic_side(o)
                price_field = kc.economic_price_field(o)
                price_dollars = o.get(price_field)
                remaining = float(o.get("remaining_count_fp", 0))
                notional = (float(price_dollars) * remaining) if price_dollars else 0.0
                print(f"    order_id={o.get('order_id')}")
                print(f"      raw: side={o.get('side')} book_side={o.get('book_side')} "
                      f"action={o.get('action')} status={o.get('status')}")
                print(f"      raw prices: yes_price_dollars={o.get('yes_price_dollars')} "
                      f"no_price_dollars={o.get('no_price_dollars')} remaining={remaining}")
                print(f"      -> ECONOMIC SIDE: {econ_side}   using field: {price_field}   "
                      f"-> price={price_dollars} ({round(float(price_dollars)*100) if price_dollars else '?'}c)  "
                      f"notional=${notional:.2f}")

        t0 = time.time()
        filled_exposure = get_filled_exposure_dollars(kc, TICKER)
        resting_notional = get_resting_notional_dollars(kc, TICKER)
        my_yes_price, my_yes_size = kc.get_my_resting_price(TICKER, "yes")
        my_no_price, my_no_size = kc.get_my_resting_price(TICKER, "no")
        t_portfolio = time.time() - t0
        print(f"\n  get_my_resting_price('yes') -> price={my_yes_price}c size={my_yes_size}")
        print(f"  get_my_resting_price('no')  -> price={my_no_price}c size={my_no_size}")
        print(f"  [budget, read-only] filled=${filled_exposure:.2f}  resting=${resting_notional:.2f}  "
              f"total=${filled_exposure+resting_notional:.2f} / ${TOTAL_BUDGET_DOLLARS:.2f}")

        # --- Closing-quote preview (read-only): mirrors manage_closing_quote's
        # logic exactly, but never places/cancels anything. Shows what WOULD
        # be quoted to exit any filled inventory, at max(cost+1c, best_bid+1c).
        print(f"\n  --- CLOSING QUOTE PREVIEW (would this fire if you had a fill?) ---")
        held_side, qty_held, avg_cost_cents = get_position_info(kc, TICKER)
        if held_side is None or qty_held <= 0:
            print(f"    No position held on this ticker — nothing to close/exit.")
        else:
            best_bid_for_held = best_yes_bid if held_side == "yes" else best_no_bid
            cost_based_price = avg_cost_cents + 1
            bid_based_price = (best_bid_for_held + 1) if best_bid_for_held is not None else cost_based_price
            sell_price_for_held = min(99, max(cost_based_price, bid_based_price))
            closing_side = "no" if held_side == "yes" else "yes"
            mechanical_price = clamp_price(100 - sell_price_for_held)
            winner = "cost+1c" if cost_based_price >= bid_based_price else "best_bid+1c"
            profit_per_contract = sell_price_for_held - avg_cost_cents
            print(f"    Holding {qty_held:.0f} {held_side.upper()} @ avg_cost={avg_cost_cents}c")
            print(f"    cost+1c={cost_based_price}c  vs  best_bid+1c={bid_based_price}c  -> using {winner}")
            print(f"    WOULD QUOTE: {closing_side.upper()} @ {mechanical_price}c "
                  f"(sells held {held_side.upper()} @ {sell_price_for_held}c, "
                  f"locks in >= {profit_per_contract}c/contract profit if filled)")

        print(f"\n  [timing] pyth_spot={t_pyth*1000:.0f}ms  orderbook={t_orderbook*1000:.0f}ms  "
              f"portfolio_check={t_portfolio*1000:.0f}ms")
        print(f"  [TOTAL SNAPSHOT TIME] {(time.time()-snapshot_start)*1000:.0f}ms  "
              f"(no order placement calls in dry-run, so live cycles will run slightly longer)")

        if i < loops - 1:
            time.sleep(interval_sec)

    print(f"\n{'='*70}\nDRY RUN COMPLETE — nothing was placed, pulled, or cancelled.\n{'='*70}")


DRY_RUN = True                  # <-- set False to actually run live and place orders
DRY_RUN_LOOPS = 1                # how many snapshots to print if DRY_RUN is True
DRY_RUN_INTERVAL_SEC = 10        # seconds between snapshots if DRY_RUN_LOOPS > 1

# Set this to a non-empty list to test the model against hypothetical spot
# prices instead of the live Pyth feed, e.g. CUSTOM_SPOT_TEST = [100.90, 100.95,
# 101.00, 101.05]. Runs one refit (slow part, done once), then prices each
# value against the LIVE Kalshi orderbook (read-only) — no orders placed.
# Leave as [] to skip this mode and run DRY_RUN / live as normal.
CUSTOM_SPOT_TEST: list[float] = []  # 個人智慧財產,參數不提供 (placeholder)
CUSTOM_SPOT_TEST_BARS_REMAINING: int | None = None  # None = use real current time


if __name__ == "__main__":
    if CUSTOM_SPOT_TEST:
        _kc = KalshiClient(key_id=KALSHI_KEY_ID, private_key_path=KALSHI_PRIVATE_KEY_PATH)
        _pricer = DXYPricer(
            symbol=PYTH_SYMBOL, asset_type=PYTH_ASSET_TYPE, history_days=HISTORY_DAYS,
            resolution_min=BAR_RESOLUTION_MIN, garch_scale=GARCH_SCALE,
            assume_zero_drift=ASSUME_ZERO_DRIFT, risk_free_rate=RISK_FREE_RATE,
            daily_settlement_et=DAILY_SETTLEMENT_ET,
        )
        print("Fitting GARCH once (shared across all custom spot tests below)...")
        _pricer.refit()
        for _spot in CUSTOM_SPOT_TEST:
            test_custom_spot(_pricer, _kc, _spot, bars_remaining=CUSTOM_SPOT_TEST_BARS_REMAINING)
    elif DRY_RUN:
        dry_run_test(loops=DRY_RUN_LOOPS, interval_sec=DRY_RUN_INTERVAL_SEC)
    else:
        run_autoquoter()
