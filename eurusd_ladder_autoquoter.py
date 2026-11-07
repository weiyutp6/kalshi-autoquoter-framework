#!/usr/bin/env python3
"""
EUR/USD Strike-Ladder Autoquoter for Kalshi.

Structural difference from dxy_autoquoter.py: DXY quoted ONE ticker/ONE
strike. EUR/USD has a full array of strikes under a
single event (e.g. KXEURUSDAW-26JUL31: 1.117, 1.119, 1.121, ... 1.133) all
sharing the same underlying and settlement — so this is priced the way an
interest-rate desk prices a strip: ONE shared spot fetch + ONE shared GARCH
vol fit per loop, then Black-Scholes evaluated independently at EACH strike
in the ladder. No per-strike refitting.

Budget is POOLED across the entire ladder (one TOTAL_BUDGET_DOLLARS shared
by every strike's filled + resting exposure combined), not per-strike —
chosen as the default; change
BUDGET_MODE below if you actually want per-strike caps instead.

ITM/OTM and edge/sizing/requote logic is otherwise copied from
dxy_autoquoter.py unchanged, just applied per-strike in a loop each cycle.

Spot feed is Pyth HERMES (feed auto-discovered by searching "EUR/USD",
same pattern as DXYPricer/dxy_autoquoter — not Lazer).

Settlement time is pulled AUTHORITATIVELY from Kalshi via
get_authoritative_settlement() (occurrence_datetime, falling back to
close_time), same as dxy_autoquoter.run_autoquoter — no longer guessed
from the ticker string or a hardcoded ET time.

REMAINING UNVERIFIED ITEMS:
- KalshiClient.py's base_url is api.elections.kalshi.com; Kalshi's own API
  docs reference external-api.kalshi.com. If get_market() 404s, try
  swapping the base_url.
- PYTH_HISTORY_SYMBOL against the benchmarks history-shim endpoint —
  check load_or_update_history() actually returns bars before trusting
  the GARCH fit.

Requires KalshiClient.py  and EURUSDPricer.py
importable from the same directory.
"""
from __future__ import annotations

import math
import os
import time
from datetime import datetime, timezone

from KalshiClient import KalshiClient
from EURUSDPricer import EURUSDPricer, parse_strike_and_settlement, parse_series_and_event

# ===========================================================================
# CONFIG
# ===========================================================================
KALSHI_KEY_ID = os.environ.get("KALSHI_KEY_ID", "")
KALSHI_PRIVATE_KEY_PATH = os.environ.get("KALSHI_PRIVATE_KEY_PATH", "kalshi_private_key.pem")

# One representative ticker from the ladder — used only to derive the
# series/event ticker and the shared settlement time. The actual list of
# strikes quoted is discovered live via KalshiClient.find_markets(), not
# hardcoded, so new/removed strikes in the ladder are picked up automatically.
SAMPLE_TICKER = "KXEURUSDAW-26JUL31-1.1170"      # <-- update daily/weekly
DAILY_SETTLEMENT_ET = "17:00:00"                  # NO LONGER used to determine settlement —
                                                     # kept only because parse_strike_and_settlement's
                                                     # signature requires it when parsing strikes out
                                                     # of ticker strings; the actual settlement time
                                                     # now comes from get_authoritative_settlement()
                                                     # (Kalshi's own occurrence_datetime/close_time),
                                                     # not from this constant.

PYTH_SYMBOL = "FX.EURUSD"           # search query used for Hermes feed discovery
PYTH_HISTORY_SYMBOL = "FX.EUR/USD"   # benchmarks history-shim symbol — verify this resolves;
                                       # if it 404s/empty, try "EUR/USD" (with slash) instead
HISTORY_DAYS = 30  # 個人智慧財產,參數不提供 (placeholder)
BAR_RESOLUTION_MIN = 1
GARCH_SCALE = 10000.0
ASSUME_ZERO_DRIFT = True
RISK_FREE_RATE = 0.0

DISABLE_PERIODIC_REFIT = True
REFIT_INTERVAL_SEC = 300  # 個人智慧財產,參數不提供 (placeholder)
POLL_INTERVAL_SEC = 5  # 個人智慧財產,參數不提供 (placeholder)
QUOTE_NOTIONAL_DOLLARS = 10.0  # 個人智慧財產,參數不提供 (placeholder)
EDGE_CENTER_CENTS = 50  # 個人智慧財產,參數不提供 (placeholder)
EDGE_AT_CENTER_CENTS = 4  # 個人智慧財產,參數不提供 (placeholder)
EDGE_SLOPE = 0.06  # 個人智慧財產,參數不提供 (placeholder)
EDGE_MIN_CENTS = 2  # 個人智慧財產,參數不提供 (placeholder)
REQUOTE_THRESHOLD_CENTS = 2  # 個人智慧財產,參數不提供 (placeholder)

# --- Interest-rate-strip style near-certainty rule ---
# Once fair value is this confident (99c+ / 1c-), the contract is
# essentially decided — trying to extract edge here just gets you skipped
# by anyone willing to buy at the true price. Instead of quoting
# fair - dynamic_edge, JOIN the best bid outright (no edge at all) so the
# quote actually has a chance of getting filled/collecting the free cents
# still on the table between "certain" and "100c". This mirrors how a
# rates desk treats deep ITM/OTM strikes on a strip: far-from-the-money
# legs aren't priced for edge, they're priced to just get done.
HIGH_CONFIDENCE_THRESHOLD_CENTS = 97  # 個人智慧財產,參數不提供 (placeholder)

# --- BUDGET: pooled across the WHOLE ladder, not per-strike ---
TOTAL_BUDGET_DOLLARS = 50.0  # 個人智慧財產,參數不提供 (placeholder)


def clamp_price(c: int) -> int:
    return max(1, min(99, c))


def dynamic_edge_cents(fair_price_cents: int) -> int:
    distance_from_center = abs(fair_price_cents - EDGE_CENTER_CENTS)
    edge = EDGE_AT_CENTER_CENTS - EDGE_SLOPE * distance_from_center
    return max(EDGE_MIN_CENTS, round(edge))


def compute_target_price(fair_price_cents: int, best_bid: int | None, best_ask: int | None) -> int:
    """
    Normal case: fair_price_cents - dynamic_edge_cents(fair).

    NEAR-CERTAINTY OVERRIDE: if fair_price_cents >= HIGH_CONFIDENCE_THRESHOLD_CENTS
    (i.e. this side is basically a lock), skip the edge calc entirely and
    just MATCH the best bid — join the top of book instead of trying to sit
    behind it for margin that isn't really there anymore. Falls back to a
    minimal-edge price if the book is too thin to have a best bid at all.
    """
    if fair_price_cents >= HIGH_CONFIDENCE_THRESHOLD_CENTS:
        candidate = best_bid if best_bid is not None else fair_price_cents - EDGE_MIN_CENTS
    else:
        edge = dynamic_edge_cents(fair_price_cents)
        candidate = fair_price_cents - edge

    if best_ask is not None:
        candidate = min(candidate, best_ask - 1)
    return clamp_price(candidate)


def get_authoritative_settlement(kc: KalshiClient, ticker: str) -> tuple[datetime, str]:
    """
    Pulls the REAL settlement/resolution-observation time for this ticker
    directly from Kalshi — same approach as dxy_autoquoter.run_autoquoter:
    prefer market_info["occurrence_datetime"] (the actual resolution
    moment) over close_time (when trading stops, which can be later and is
    NOT necessarily the same instant). Falls back to close_time only if
    occurrence_datetime isn't present, and says so loudly either way.

    Returns (settlement_utc, source_note) — source_note describes which
    field was actually used, for the caller to print/log.
    """
    market_info = kc.get_market(ticker)
    occurrence_str = market_info.get("occurrence_datetime")
    close_time_str = market_info.get("close_time")
    close_time_utc = datetime.fromisoformat(close_time_str.replace("Z", "+00:00")) if close_time_str else None

    if occurrence_str:
        settlement_utc = datetime.fromisoformat(occurrence_str.replace("Z", "+00:00"))
        note = f"occurrence_datetime={settlement_utc.isoformat()} (authoritative resolution time)"
        if close_time_utc and abs((close_time_utc - settlement_utc).total_seconds()) > 60:
            note += (f"  [NOTE: close_time ({close_time_utc.isoformat()}) differs from "
                      f"occurrence_datetime — close_time is when trading stops, NOT when "
                      f"the contract resolves; using occurrence_datetime]")
        return settlement_utc, note

    if close_time_utc is None:
        raise RuntimeError(f"get_market({ticker}) returned neither occurrence_datetime nor "
                            f"close_time — cannot determine settlement. Raw response: {market_info}")
    return close_time_utc, (f"WARNING: no occurrence_datetime on {ticker} — falling back to "
                             f"close_time={close_time_utc.isoformat()}, which may NOT be the "
                             f"actual resolution-observation moment for this contract type.")


def print_garch_summary(pricer: EURUSDPricer) -> None:
    g = pricer.g
    if g is None:
        print("  (no GARCH fit yet)")
        return
    print(f"  mu={g.mu:.8f}  omega={g.omega:.10f}  alpha={g.alpha:.6f}  "
          f"beta={g.beta:.6f}  last_h={g.last_h:.10f}  "
          f"persistence(alpha+beta)={g.alpha+g.beta:.6f}")
    vol_60 = pricer.annualized_vol_forecast(60)
    long_run = pricer.long_run_unconditional_vol()
    long_run_str = f"{long_run*100:.3f}%" if not math.isnan(long_run) else "n/a (persistence~1)"
    print(f"  sigma_annual (60-bar fwd)={vol_60*100:.3f}%   long-run unconditional={long_run_str}")


def determine_itm_side(spot: float, strike: float) -> str | None:
    if spot > strike:
        return "yes"
    elif spot < strike:
        return "no"
    return None


def get_filled_exposure_dollars(kc: KalshiClient, ticker: str) -> float:
    for p in kc.get_positions():
        if p.get("ticker") == ticker:
            try:
                return abs(float(p.get("market_exposure_dollars", 0)))
            except (TypeError, ValueError):
                return 0.0
    return 0.0


def get_resting_notional_dollars(kc: KalshiClient, ticker: str) -> float:
    total = 0.0
    for o in kc.get_open_orders(ticker):
        if not KalshiClient._is_actually_resting(o):
            continue
        price_dollars = o.get(KalshiClient.economic_price_field(o))
        remaining = float(o.get("remaining_count_fp", 0))
        if price_dollars is not None:
            total += float(price_dollars) * remaining
    return total


def compute_would_be_size(price_cents: int, remaining_budget: float) -> tuple[int, float, str]:
    """Same math as DXY's compute_would_be_size, but takes the ALREADY-POOLED
    remaining budget directly (pool is shared across every strike in the
    ladder, so the caller computes remaining once per loop, not per-strike)."""
    price_dollars = price_cents / 100.0
    if price_dollars <= 0:
        return 0, 0.0, "invalid price"
    desired = max(1, int(QUOTE_NOTIONAL_DOLLARS // price_dollars))
    if remaining_budget <= 0:
        return 0, 0.0, "ladder budget fully committed"
    max_affordable = int(remaining_budget // price_dollars)
    size = min(desired, max_affordable)
    if size <= 0:
        return 0, 0.0, f"can't afford even 1 (${remaining_budget:.2f} pooled remaining)"
    note = "" if size == desired else f", budget-limited from {desired} contracts"
    return size, size * price_dollars, note


def maybe_requote_side(kc: KalshiClient, ticker: str, side: str, target_price: int,
                        remaining_budget_before_this_order: float,
                        requote_threshold_cents: int = REQUOTE_THRESHOLD_CENTS) -> float:
    """Same place-first-confirm-then-cancel-old pattern as dxy_autoquoter.

    remaining_budget_before_this_order is the FILLED-exposure-only budget
    remaining this cycle (resting orders don't lock capital, so they're not
    subtracted from it — every strike/side is sized against the same
    filled-only figure, not a pool that shrinks as more resting quotes go
    out). sizing is capped by QUOTE_NOTIONAL_DOLLARS per quote regardless.

    requote_threshold_cents: how far target_price must move from the
    currently-resting price before bothering to requote. Pass a tighter
    value (e.g. 1) when in HIGH_CONFIDENCE match-the-bid mode — that mode's
    whole point is to track the best bid tick-for-tick, so gating it behind
    the normal 3c fair-value-noise threshold would leave stale quotes
    sitting behind a bid that already moved."""
    current_price, current_size = kc.get_my_resting_price(ticker, side)

    if current_price is not None and abs(target_price - current_price) < requote_threshold_cents:
        prev_notional = (current_price / 100.0) * current_size
        return prev_notional  # unchanged, nothing to do

    existing_orders = [
        o for o in kc.get_open_orders(ticker)
        if kc.economic_side(o) == side and kc._is_actually_resting(o)
    ]

    price_dollars = target_price / 100.0
    if price_dollars <= 0:
        return 0.0

    size, notional, note = compute_would_be_size(target_price, remaining_budget_before_this_order)
    if size <= 0:
        print(f"  SKIPPING {ticker} {side.upper()} requote @ {target_price}c: {note}")
        return 0.0

    was = f"{current_price}c" if current_price is not None else "none"
    old_order_ids = [o["order_id"] for o in existing_orders]
    result = kc.place_limit_order(ticker, side=side, price_cents=target_price, count=size)
    new_is_resting = (result.get("fill_count", "0.00") != "0.00"
                       or result.get("remaining_count", "0.00") != "0.00")

    if not new_is_resting:
        print(f"  {ticker} new {side.upper()} quote @ {target_price}c rejected (post_only cross) "
              f"— keeping old quote at {was}.")
        prev_notional = (current_price / 100.0 * current_size) if current_price else 0.0
        return prev_notional

    if old_order_ids:
        kc.cancel_orders_batch(old_order_ids)

    print(f"  {ticker} requoted {side.upper()}: {size} @ {target_price}c "
          f"(${notional:.2f} notional, was {was}){note and ' [' + note + ']'}")
    return notional


def run_ladder_dry_run(loops: int = 1, interval_sec: int = 10) -> None:
    """
    Read-only preview: fits GARCH ONCE, fetches spot ONCE per snapshot,
    discovers every open strike in the ladder's event, and prints what
    WOULD be quoted on each — no orders placed/pulled/cancelled.
    """
    kc = KalshiClient(key_id=KALSHI_KEY_ID, private_key_path=KALSHI_PRIVATE_KEY_PATH)
    pricer = EURUSDPricer(
        symbol=PYTH_SYMBOL, history_symbol=PYTH_HISTORY_SYMBOL,
        history_days=HISTORY_DAYS, resolution_min=BAR_RESOLUTION_MIN,
        garch_scale=GARCH_SCALE, assume_zero_drift=ASSUME_ZERO_DRIFT,
        risk_free_rate=RISK_FREE_RATE, daily_settlement_et=DAILY_SETTLEMENT_ET,
    )

    series_ticker, event_ticker = parse_series_and_event(SAMPLE_TICKER)
    settlement_utc, settlement_note = get_authoritative_settlement(kc, SAMPLE_TICKER)

    print("=" * 70)
    print("DRY RUN (LADDER) — no orders will be placed, pulled, or cancelled.")
    print("=" * 70)
    print(f"series_ticker={series_ticker}  event_ticker={event_ticker}")
    print(f"Shared settlement (UTC): {settlement_note}")
    print(f"Pooled ladder budget: ${TOTAL_BUDGET_DOLLARS:.2f}  |  "
          f"~${QUOTE_NOTIONAL_DOLLARS:.2f} notional/quote  |  "
          f"requote threshold {REQUOTE_THRESHOLD_CENTS}c")
    days_to_maturity = (settlement_utc - datetime.now(timezone.utc)).total_seconds() / 86400
    print(f"Time to maturity: {days_to_maturity:.4f} days")

    print("\nDiscovering strikes in this event...")
    all_markets = kc.find_markets(series_ticker=series_ticker, status="open")
    ladder = [m for m in all_markets if m.get("event_ticker") == event_ticker or m.get("ticker", "").startswith(event_ticker)]
    if not ladder:
        print(f"  WARNING: no open markets matched event_ticker={event_ticker} via find_markets "
              f"(series_ticker={series_ticker}). Falling back to just SAMPLE_TICKER.")
        ladder = [{"ticker": SAMPLE_TICKER}]
    strikes = []
    for m in ladder:
        ticker = m["ticker"]
        try:
            strike, _ = parse_strike_and_settlement(ticker, DAILY_SETTLEMENT_ET)
        except ValueError:
            continue
        strikes.append((strike, ticker))
    strikes.sort()
    print(f"  Found {len(strikes)} strikes: {[t for _, t in strikes]}")

    print("\nFitting GARCH (shared across the whole ladder, done once)...")
    pricer.refit()
    print("GARCH fit:")
    print_garch_summary(pricer)

    for i in range(loops):
        print(f"\n{'='*70}\nSNAPSHOT {i+1}/{loops}  [{datetime.now(timezone.utc).isoformat()}]\n{'='*70}")

        spot, conf, pub_ts = pricer.latest_price()
        days_to_maturity = (settlement_utc - datetime.now(timezone.utc)).total_seconds() / 86400
        print(f"spot={spot:.6f}  conf=±{conf:.6f}  maturity={days_to_maturity:.4f} days")

        # BUDGET: only FILLED exposure locks up capital. Resting (unfilled)
        # limit orders don't — no margin/capital is committed until they
        # actually fill, so they must NOT count against the cap. This is
        # shown below purely as a diagnostic, not subtracted from the pool.
        total_filled = 0.0
        total_resting = 0.0
        for strike, ticker in strikes:
            total_filled += get_filled_exposure_dollars(kc, ticker)
            total_resting += get_resting_notional_dollars(kc, ticker)
        remaining_pool = TOTAL_BUDGET_DOLLARS - total_filled
        print(f"[budget] filled=${total_filled:.2f} (counts against cap)  "
              f"resting=${total_resting:.2f} (informational only, doesn't lock capital)  "
              f"remaining=${remaining_pool:.2f} / ${TOTAL_BUDGET_DOLLARS:.2f}")

        for strike, ticker in strikes:
            fair = pricer.price_strike(spot, strike, settlement_utc)
            market = kc.best_market_prices(kc.get_orderbook(ticker))
            if market is not None:
                best_yes_bid, best_yes_ask = market
                best_no_bid, best_no_ask = 100 - best_yes_ask, 100 - best_yes_bid
                mkt_str = f"mkt={best_yes_bid}/{best_yes_ask}"
            else:
                best_yes_bid = best_yes_ask = best_no_bid = best_no_ask = None
                mkt_str = "mkt=thin"

            itm_side = determine_itm_side(spot, strike)
            target_yes = compute_target_price(fair["fair_yes_cents"], best_yes_bid, best_yes_ask)
            target_no = compute_target_price(fair["fair_no_cents"], best_no_bid, best_no_ask)
            yes_confident = itm_side in (None, "yes") and fair["fair_yes_cents"] >= HIGH_CONFIDENCE_THRESHOLD_CENTS
            no_confident = itm_side in (None, "no") and fair["fair_no_cents"] >= HIGH_CONFIDENCE_THRESHOLD_CENTS

            prefix = (f"  {ticker} strike={strike} fair_yes={fair['fair_yes_cents']}c "
                      f"fair_no={fair['fair_no_cents']}c itm={itm_side or 'both'} {mkt_str} -> ")

            # Only quote lines where true probability is >= HIGH_CONFIDENCE_THRESHOLD_CENTS —
            # everything else is SKIPPED entirely (not just priced with more edge).
            if yes_confident:
                size, notional, note = compute_would_be_size(target_yes, remaining_pool)
                print(prefix + f"QUOTE YES {size} @ {target_yes}c (${notional:.2f})"
                      + (f" [{note}]" if note else ""))
            elif no_confident:
                size, notional, note = compute_would_be_size(target_no, remaining_pool)
                print(prefix + f"QUOTE NO {size} @ {target_no}c (${notional:.2f})"
                      + (f" [{note}]" if note else ""))
            else:
                print(prefix + f"SKIP (not >= {HIGH_CONFIDENCE_THRESHOLD_CENTS}c confident or OTM)")

        if i < loops - 1:
            time.sleep(interval_sec)

    print(f"\n{'='*70}\nDRY RUN COMPLETE.\n{'='*70}")


def run_ladder_live():
    """Live loop: shared spot+vol fit per cycle, one pooled budget, quotes
    every strike's ITM side in sequence each cycle, debiting the pool as it
    goes (so strike #1 can't accidentally reserve the whole budget before
    strike #2 is even evaluated — later strikes see what's actually left)."""
    kc = KalshiClient(key_id=KALSHI_KEY_ID, private_key_path=KALSHI_PRIVATE_KEY_PATH)
    pricer = EURUSDPricer(
        symbol=PYTH_SYMBOL, history_symbol=PYTH_HISTORY_SYMBOL,
        history_days=HISTORY_DAYS, resolution_min=BAR_RESOLUTION_MIN,
        garch_scale=GARCH_SCALE, assume_zero_drift=ASSUME_ZERO_DRIFT,
        risk_free_rate=RISK_FREE_RATE, daily_settlement_et=DAILY_SETTLEMENT_ET,
    )

    series_ticker, event_ticker = parse_series_and_event(SAMPLE_TICKER)
    settlement_utc, settlement_note = get_authoritative_settlement(kc, SAMPLE_TICKER)

    print(f"Ladder event: {event_ticker}  (series {series_ticker})")
    print(f"Shared settlement (UTC): {settlement_note}")

    all_markets = kc.find_markets(series_ticker=series_ticker, status="open")
    ladder = [m for m in all_markets if m.get("event_ticker") == event_ticker or m.get("ticker", "").startswith(event_ticker)]
    strikes = []
    for m in ladder:
        ticker = m["ticker"]
        try:
            strike, _ = parse_strike_and_settlement(ticker, DAILY_SETTLEMENT_ET)
        except ValueError:
            continue
        strikes.append((strike, ticker))
    strikes.sort()
    print(f"Quoting {len(strikes)} strikes: {[t for _, t in strikes]}\n")

    print("Pulling any existing resting orders across the whole ladder before starting...")
    for _, ticker in strikes:
        kc.cancel_all_orders(ticker)
    print("Done.\n")

    print("Initial GARCH fit (shared across the ladder, this is the slow part)...")
    pricer.refit()
    last_refit_time = time.time()
    print("GARCH fit:")
    print_garch_summary(pricer)
    print("Done.\n")

    try:
        while True:
            cycle_start = time.time()
            try:
                if not DISABLE_PERIODIC_REFIT and time.time() - last_refit_time >= REFIT_INTERVAL_SEC:
                    print(f"[{datetime.now(timezone.utc).isoformat()}] Refitting GARCH...")
                    pricer.refit()
                    print_garch_summary(pricer)
                    last_refit_time = time.time()

                spot, conf, pub_ts = pricer.latest_price()
                days_to_maturity = (settlement_utc - datetime.now(timezone.utc)).total_seconds() / 86400

                # BUDGET: only FILLED exposure locks up capital — resting
                # (unfilled) limit orders don't commit any margin/capital
                # until they actually fill, so they must NOT count against
                # the cap. Shown below purely as a diagnostic.
                total_filled = 0.0
                total_resting = 0.0
                for strike, ticker in strikes:
                    total_filled += get_filled_exposure_dollars(kc, ticker)
                    total_resting += get_resting_notional_dollars(kc, ticker)
                remaining_pool = TOTAL_BUDGET_DOLLARS - total_filled

                print(f"\n[{datetime.now(timezone.utc).isoformat()}] spot={spot:.6f}  "
                      f"maturity={days_to_maturity:.4f} days  "
                      f"[budget] filled=${total_filled:.2f} (counts against cap)  "
                      f"resting=${total_resting:.2f} (informational only)  "
                      f"remaining=${remaining_pool:.2f}/${TOTAL_BUDGET_DOLLARS:.2f}")

                if total_filled >= TOTAL_BUDGET_DOLLARS:
                    print("  Filled exposure alone has reached/exceeded the cap. "
                          "Pulling all resting orders across the ladder this cycle.")
                    for _, ticker in strikes:
                        kc.cancel_all_orders(ticker)
                    time.sleep(POLL_INTERVAL_SEC)
                    continue

                for strike, ticker in strikes:
                    fair = pricer.price_strike(spot, strike, settlement_utc)
                    market = kc.best_market_prices(kc.get_orderbook(ticker))
                    if market is not None:
                        best_yes_bid, best_yes_ask = market
                        best_no_bid, best_no_ask = 100 - best_yes_ask, 100 - best_yes_bid
                    else:
                        best_yes_bid = best_yes_ask = best_no_bid = best_no_ask = None

                    itm_side = determine_itm_side(spot, strike)
                    target_yes = compute_target_price(fair["fair_yes_cents"], best_yes_bid, best_yes_ask)
                    target_no = compute_target_price(fair["fair_no_cents"], best_no_bid, best_no_ask)

                    yes_price, yes_size = kc.get_my_resting_price(ticker, "yes")
                    no_price, no_size = kc.get_my_resting_price(ticker, "no")

                    # ONLY quote a side when it's BOTH the ITM side AND the true
                    # (model) probability is >= HIGH_CONFIDENCE_THRESHOLD_CENTS
                    # (near-certain). Strikes that are ITM but not yet near-certain are
                    # NOT quoted at all — pulled just like the OTM side — this is
                    # the "only trade the lines you're basically sure of" rule,
                    # not just a pricing tweak on lines you'd quote anyway.
                    yes_confident = itm_side in (None, "yes") and fair["fair_yes_cents"] >= HIGH_CONFIDENCE_THRESHOLD_CENTS
                    no_confident = itm_side in (None, "no") and fair["fair_no_cents"] >= HIGH_CONFIDENCE_THRESHOLD_CENTS

                    mkt_str = f"mkt={best_yes_bid}/{best_yes_ask}" if market is not None else "mkt=thin"
                    print(f"  {ticker} strike={strike} fair_yes={fair['fair_yes_cents']}c "
                          f"fair_no={fair['fair_no_cents']}c itm={itm_side or 'both'} {mkt_str}"
                          + (" [confident]" if (yes_confident or no_confident) else ""))

                    # NOTE: remaining_pool is intentionally NOT debited after each
                    # quote below anymore — resting orders don't lock capital, so
                    # every strike gets sized against the same filled-exposure-only
                    # remaining_pool for this cycle, not a shrinking shared pool.
                    if yes_confident:
                        maybe_requote_side(kc, ticker, "yes", target_yes, remaining_pool,
                                            requote_threshold_cents=1)
                    elif yes_price is not None:
                        kc.pull_side(ticker, "yes")
                        print(f"    pulled YES (not >= {HIGH_CONFIDENCE_THRESHOLD_CENTS}c confident or OTM)")

                    if no_confident:
                        maybe_requote_side(kc, ticker, "no", target_no, remaining_pool,
                                            requote_threshold_cents=1)
                    elif no_price is not None:
                        kc.pull_side(ticker, "no")
                        print(f"    pulled NO (not >= {HIGH_CONFIDENCE_THRESHOLD_CENTS}c confident or OTM)")

                print(f"  [TOTAL CYCLE TIME] {(time.time()-cycle_start)*1000:.0f}ms")

            except Exception as e:  # noqa: BLE001
                print(f"ERROR during loop iteration: {type(e).__name__}: {e}")
                print("FAIL-SAFE: pulling all quotes across the entire ladder...")
                try:
                    for _, ticker in strikes:
                        kc.cancel_all_orders(ticker)
                    print("  All orders pulled successfully.")
                except Exception as cancel_err:  # noqa: BLE001
                    print(f"  WARNING: fail-safe pull itself failed: {cancel_err}. Check manually.")

            time.sleep(POLL_INTERVAL_SEC)

    except KeyboardInterrupt:
        print("\nInterrupted by user. Pulling all quotes across the ladder before exit...")
        for _, ticker in strikes:
            kc.cancel_all_orders(ticker)
        print("Done.")
    finally:
        try:
            for _, ticker in strikes:
                kc.cancel_all_orders(ticker)
        except Exception:  # noqa: BLE001
            pass


DRY_RUN = True
DRY_RUN_LOOPS = 1
DRY_RUN_INTERVAL_SEC = 5

if __name__ == "__main__":
    if DRY_RUN:
        run_ladder_dry_run(loops=DRY_RUN_LOOPS, interval_sec=DRY_RUN_INTERVAL_SEC)
    else:
        run_ladder_live()