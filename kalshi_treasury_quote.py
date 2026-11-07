#!/usr/bin/env python3
"""
Finds every eligible Treasury-yield line (fair_yes or fair_no > 99c, via
KalshiTreasuryFairValue) and quotes each one using a variant of Autoquoter
that prices at: JOIN/IMPROVE THE BEST BID BY 1c IF ONE EXISTS, ELSE QUOTE
AT FAIR VALUE (capped at 99c) -- not Autoquoter's default dynamic-edge
pricing.

WHY A SUBCLASS: Autoquoter's default compute_target_price() implements a
different rule (fair value minus a shrinking edge, capped by the current
ask). Rather than duplicate all of Autoquoter's other machinery (budget
cap, requote-threshold, ITM/OTM single-siding, closing-quote management),
BestBidOrFairAutoquoter only overrides compute_target_price() and inherits
everything else unchanged.

BUDGET MODEL -- IMPORTANT, and different from a naive "shared pool":
Kalshi does NOT lock capital for merely RESTING (unfilled) orders -- only
for FILLS. That means every tracked ticker can independently rest at up to
the FULL GLOBAL_BUDGET_DOLLARS notional simultaneously; that's not a bug,
it's how the exchange actually works, and there's no real contention to
divide between lines. The budget should only shrink in response to ACTUAL
FILLS (account-wide), and when it does, EVERY line gets the SAME reduced
ceiling applied independently -- not a fraction of it. Concretely: total
account-wide FILLED exposure is computed fresh every cycle, subtracted from
GLOBAL_BUDGET_DOLLARS, and that single number is written onto every
tracked bot's total_budget_dollars before its cycle runs. No shared-pool
reservation/locking is needed (there's no scarce resource being divided),
just a plain per-cycle refresh of a shared ceiling value.

ARCHITECTURE -- single-threaded round robin over N concurrently-executing
bot cycles: one Autoquoter (subclass) instance is created per eligible
ticker and kept alive across cycles; each outer loop iteration re-checks
eligibility, spins up bots for newly-eligible lines, tears down bots for
lines that dropped out of eligibility (pulling their quotes via
shutdown()), recomputes the shared ceiling from real filled exposure, and
runs run_cycle() for every live bot IN PARALLEL via a thread pool (safe:
each bot only touches its own ticker/state; get_positions() is fetched
once and shared read-only).

SAFETY: DRY_RUN defaults to True, same convention as everywhere else in
this project.
"""
from __future__ import annotations

import os

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from KalshiClient import KalshiClient
from kalshi_treasury_fair_value import KalshiTreasuryFairValue
from autoquoter import Autoquoter, resting_price_and_size

KEY_ID = os.environ.get("KALSHI_KEY_ID", "")
PRIVATE_KEY_PATH = os.environ.get("KALSHI_PRIVATE_KEY_PATH", "kalshi_private_key.pem")

DRY_RUN = True
VERBOSE = True   # per-cycle diagnostic logging (fair value, spot/strike, resting
                  # status) for every line, every cycle -- independent of DRY_RUN.
                  # Without this, LIVE runs only print when something actually
                  # CHANGES (a requote, a pull), so a cycle where nothing needed
                  # to change prints nothing at all -- correct, but gives zero
                  # visibility into whether the bot is alive and evaluating
                  # things correctly. Set to False once you trust it and want
                  # quieter logs.
POLL_INTERVAL_SEC = 10

# Which of the 5 tenors to actually track and quote. Restricting this
# skips unwanted tenors at the SOURCE (KalshiTreasuryFairValue passes this
# straight through to KalshiTreasuryLines) -- unwanted tenors are never
# discovered, priced, or quoted at all, not just filtered out of the
# output after the fact. This is where most of the API-call cost lives
# (one get_market() call per open strike per tenor), so narrowing this is
# a real efficiency win if you don't need all 5, not just a display filter.
# Valid values: "US2Y", "US5Y", "US7Y", "US10Y", "US30Y".
ENABLED_TENORS: tuple[str, ...] = ("US2Y", "US5Y", "US7Y", "US10Y", "US30Y")

GLOBAL_BUDGET_DOLLARS = 100.0  # TOTAL account risk ceiling. See module
                                  # docstring's BUDGET MODEL section -- this
                                  # shrinks only in response to real FILLS
                                  # (account-wide), and the resulting ceiling
                                  # applies identically to every tracked line
                                  # (not divided across them). There is
                                  # DELIBERATELY no separate "quote notional"
                                  # constant -- every quote always sizes to
                                  # the full current ceiling (see
                                  # run_one_cycle), since that's the actual
                                  # strategy. A leftover smaller notional cap
                                  # here once silently capped every line to
                                  # ~10 contracts regardless of how much
                                  # budget was free -- don't reintroduce one.

MATURITY_BUFFER_MINUTES = 30.0  # Padding added to maturity_hours before it
                                  # feeds the CIR pricing model (both for
                                  # ELIGIBILITY checks and live quoting) --
                                  # NOT a change to the displayed "maturity=
                                  # X.Xh" in log lines, which always shows
                                  # the true clock-based remaining time.
                                  # Exists to keep the model from claiming
                                  # near-100% certainty in the final stretch
                                  # before real settlement, when feed-vs-
                                  # official-announcement discrepancy risk
                                  # is proportionally largest. Made explicit
                                  # here (passed to KalshiTreasuryFairValue
                                  # below) rather than left as an invisible
                                  # default buried in a different file --
                                  # this project has been bitten by stale-
                                  # file mismatches enough times that hidden
                                  # defaults in other modules shouldn't be
                                  # trusted silently.

# If True, stray orders (see find_stray_orders below) are actually pulled
# when not in DRY_RUN. If False, they're only reported -- default False
# because auto-cancelling orders this script never placed is a bigger
# action than "quote what's eligible"; it's worth eyeballing the report
# at least once before letting the bot act on it automatically.
AUTO_CANCEL_STRAY_ORDERS = True


class BestBidOrFairAutoquoter(Autoquoter):
    """Prices at: best_bid + 1c if a best bid exists, else fair value --
    always capped at 99c and never above fair value (won't outbid its own
    model), and never crossing the current ask (post_only would reject
    that anyway, but this avoids a pointless rejected-order round trip).

    Purpose-built constructor: Autoquoter's edge_center_cents/
    edge_at_center_cents/edge_slope/edge_min_cents only feed
    dynamic_edge_cents(), which the base compute_target_price() uses --
    since this class overrides compute_target_price() entirely and never
    calls dynamic_edge_cents(), those four params are dead weight here and
    are dropped from this constructor rather than exposed for no reason.

    requote_threshold_cents defaults to 1 (not Autoquoter's general-
    purpose default of 3) -- this strategy's whole point is tracking the
    best bid closely; a 3c dead zone would mean a 1-2c move in the best
    bid gets ignored, defeating the point of "join/improve the best bid."
    1c is the tightest meaningful value given prices are integer cents:
    it requotes whenever the target actually changes, and only skips when
    it's truly identical to what's already resting.
    """

    def __init__(
        self,
        kc: KalshiClient,
        ticker: str,
        price_fn,
        *,
        quote_notional_dollars: float = 1200.0,
        total_budget_dollars: float = 1200.0,
        requote_threshold_cents: int = 1,
        dry_run: bool = True,
        verbose: bool | None = None,
        label: str | None = None,
    ):
        super().__init__(
            kc, ticker, price_fn,
            quote_notional_dollars=quote_notional_dollars,
            total_budget_dollars=total_budget_dollars,
            requote_threshold_cents=requote_threshold_cents,
            dry_run=dry_run,
            verbose=verbose,
            label=label,
        )

    def compute_target_price(self, fair_price_cents: int, best_bid: int | None, best_ask: int | None) -> int | None:
        # Kalshi's liquidity-reward policy requires liquidity on BOTH sides
        # of the book to qualify -- this strategy is deliberately single-
        # sided (only the ITM/target side is ever quoted), so joining/
        # matching an ALREADY-maximal best bid (99c, this strategy's own
        # cap) adds capital exposure with no reward benefit: it can't
        # improve past 99c, and it doesn't create the missing other-side
        # liquidity the policy actually requires. Returning None here (not
        # 99c) tells maybe_requote_side() to skip this side ENTIRELY --
        # including pulling any existing order already resting there.
        if best_bid is not None and best_bid >= 99:
            return None

        if best_bid is not None:
            candidate = best_bid + 1
        else:
            candidate = fair_price_cents

        candidate = min(candidate, fair_price_cents, 99)
        if best_ask is not None:
            candidate = min(candidate, best_ask - 1)

        return self.clamp_price(candidate)


def find_stray_orders(orders_by_ticker: dict[str, list[dict]], eligible_by_ticker: dict[str, dict],
                       positions: list[dict]) -> list[dict]:
    """Scans EVERY known Treasury-yield line (not just the currently
    eligible ones) for resting orders that shouldn't be there. Two kinds
    of stray flagged:
      1. A resting order on a ticker that isn't eligible at all right now
         (leftover from a prior run, a different script, or a manual
         order -- this bot never added it to `bots`, so the normal
         teardown-on-ineligibility logic in run_one_cycle never sees it).
      2. A resting order on the WRONG side of a currently-eligible ticker
         (e.g. a stray NO order sitting on a line whose target_side is
         "yes") -- EXCLUDING legitimate closing/exit quotes. If the
         account actually holds a filled position on this ticker, an
         order on the opposite side of target_side is NOT a stray -- it's
         Autoquoter's own manage_closing_quote() trying to exit that
         position (which is intentionally posted on the opposite side;
         see its MECHANICS NOTE). Without this check, every legitimate
         closing quote gets flagged as "wrong side" every single cycle,
         and if AUTO_CANCEL_STRAY_ORDERS were ever enabled, this would
         actively cancel your own exit quotes -- a serious bug, not just
         a noisy warning.

    NO API CALLS HERE -- operates entirely on orders_by_ticker, which the
    caller fetches ONCE per unique ticker (see run_one_cycle) and shares
    with this function, every bot's run_cycle(), and maybe_requote_side()'s
    existing-order lookups. This used to make its own 2-calls-per-line
    fetch via get_my_resting_price() (which has zero caching -- every call
    is a fresh HTTP request), redundant with what bots ALSO fetched for the
    same tickers moments later in the same cycle -- 100+ combined redundant
    calls per cycle across this function and Autoquoter, now down to one
    get_open_orders() call per unique ticker, shared everywhere.
    """
    held_side_by_ticker: dict[str, str] = {}
    for p in positions:
        ticker = p.get("ticker")
        try:
            position_fp = float(p.get("position_fp", 0))
        except (TypeError, ValueError):
            position_fp = 0.0
        if ticker and position_fp != 0:
            held_side_by_ticker[ticker] = "yes" if position_fp > 0 else "no"

    found = []
    for ticker, orders in orders_by_ticker.items():
        held_side = held_side_by_ticker.get(ticker)
        closing_side = ("no" if held_side == "yes" else "yes") if held_side else None

        for side in ("yes", "no"):
            price, size = resting_price_and_size(orders, side)
            if price is None:
                continue
            if ticker not in eligible_by_ticker:
                found.append({"ticker": ticker, "side": side, "price_cents": price,
                               "size": size, "reason": "ticker not currently eligible"})
            else:
                target_side = eligible_by_ticker[ticker]["target_side"]
                if side != target_side:
                    if side == closing_side:
                        continue  # legitimate closing quote for a held position, not a stray
                    found.append({"ticker": ticker, "side": side, "price_cents": price,
                                   "size": size, "reason": f"wrong side (target_side is {target_side})"})
    return found


def compute_total_filled_dollars(positions: list[dict]) -> float:
    """Real, account-wide FILLED exposure right now -- the ONLY thing that
    should shrink the quoting ceiling. Resting/unfilled orders are
    deliberately NOT included: Kalshi doesn't lock capital for them, so
    they don't consume the budget the way an actual fill does. This is
    the entire quantity GLOBAL_BUDGET_DOLLARS gets reduced by each cycle
    -- see the module docstring's BUDGET MODEL section."""
    return sum(abs(float(p.get("market_exposure_dollars", 0))) for p in positions)


def run_one_cycle(kc: KalshiClient, fv: KalshiTreasuryFairValue,
                   bots: dict[str, BestBidOrFairAutoquoter], dry_run: bool,
                   max_workers: int = 10) -> None:
    """One full iteration: refresh eligible lines, spin up bots for newly
    eligible tickers, tear down bots for tickers no longer eligible
    (pulling their quotes), recompute the shared ceiling from real filled
    exposure and apply it to every bot, and run one quoting cycle for
    every live bot.

    PARALLELIZED: each bot only reads/writes its own ticker and its own
    instance state (closing_quote_state, etc) -- there's no shared
    mutable state between bots other than the read-only KalshiClient
    reference (stateless requests.request() calls, not a shared Session)
    and the plain ceiling float written onto each bot before its cycle
    runs (a per-cycle READ, not a contested resource -- see module
    docstring). Running every bot's run_cycle() concurrently is therefore
    safe. Each bot's exception (if any) is caught individually so one bad
    ticker can't block the others in the same cycle.
    """
    fv.refresh()

    # Print the live spot rate for every tracked tenor right at the top of
    # the cycle -- fv._live_yields was just populated by the refresh() call
    # above (fetched once, shared with every price_fn() call this cycle --
    # see KalshiTreasuryFairValue.refresh()'s docstring), so this is a
    # free read, not an extra fetch.
    if fv._live_yields:
        spot_str = "  ".join(f"{t}={info['yield_pct']:.3f}%"
                              for t, info in sorted(fv._live_yields.items()))
        print(f"  [spot rates] {spot_str}")

    eligible = fv.get_eligible_lines()
    eligible_by_ticker = {line["ticker"]: line for line in eligible}

    all_lines = fv.lines_finder.get_lines()  # every known open line, not just eligible ones

    # Fetch the whole portfolio ONCE for this cycle -- get_positions() returns
    # everything regardless of which ticker asks, so sharing one fetch across
    # every bot instead of letting each bot call it independently turns N
    # redundant identical portfolio fetches into 1.
    positions = kc.get_positions()
    all_held_tickers = {
        p["ticker"] for p in positions
        if p.get("ticker") and abs(float(p.get("position_fp", 0) or 0)) > 0
    }
    # Only manage positions on THIS bot's own product (KXUST* Treasury-yield
    # tickers) -- get_positions() returns the WHOLE account, which can
    # include positions from entirely different bots/strategies (e.g. EUR/
    # USD contracts from a separate DXY bot). Without this filter, this
    # quoter would create a closing-quote-only bot for ANY held position on
    # the account and try to post exit orders on it -- directly interfering
    # with whatever OTHER bot/strategy already owns that position. Other-
    # product positions are still surfaced (so they're visible, not silently
    # invisible), just never touched by this bot.
    held_tickers = {t for t in all_held_tickers if t.startswith("KXUST")}
    other_product_tickers = all_held_tickers - held_tickers
    if other_product_tickers:
        print(f"  [other positions, not managed by this bot] {', '.join(sorted(other_product_tickers))}")

    total_filled = compute_total_filled_dollars(positions)
    effective_ceiling = max(0.0, GLOBAL_BUDGET_DOLLARS - total_filled)
    print(f"  [budget] filled=${total_filled:.2f} / ${GLOBAL_BUDGET_DOLLARS:.2f}  "
          f"ceiling per line=${effective_ceiling:.2f}")

    # --- Fetch get_open_orders() ONCE per unique ticker, shared across
    # EVERY consumer this cycle: the stray-order scan below, each bot's own
    # resting-price check inside run_cycle, and maybe_requote_side's
    # existing-order-id lookup when repricing. This single change is what
    # actually matters for CPU/network load at scale: get_my_resting_price()
    # has zero caching -- every call is its own fresh get_open_orders() HTTP
    # request -- and before this fix, the SAME ticker's orders were being
    # independently re-fetched 2-4 times across different code paths within
    # one cycle (once by the stray scan for "yes", once for "no", once more
    # by each bot's run_cycle for "yes"/"no", and again inside
    # maybe_requote_side whenever a reprice actually fired). With ~28
    # tracked tickers that was 100+ redundant HTTP round-trips every single
    # cycle -- the dominant remaining cost after every other optimization
    # pass, and the most likely explanation for sustained CPU load even on
    # otherwise-adequate hardware. Now: exactly one call per unique ticker.
    all_tickers = sorted({line["ticker"] for line in all_lines} | held_tickers)
    orders_by_ticker: dict[str, list[dict]] = {}
    if all_tickers:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {executor.submit(kc.get_open_orders, t): t for t in all_tickers}
            for future in as_completed(future_to_ticker):
                t = future_to_ticker[future]
                orders_by_ticker[t] = future.result()

    # --- DETERMINISTIC PER-CYCLE SUMMARY: fair prices first, then actual
    # quotes -- printed as one clean, ordered block using data ALREADY
    # fetched above (no extra API calls). This exists because the per-bot
    # action logs below run across N CONCURRENT threads, so their print
    # order is whatever order threads happen to finish in, not a stable
    # "here's every line" listing -- and a line that's currently being
    # pulled (e.g. by the best_bid>=99 liquidity-reward rule) doesn't
    # print a fair-value line the same way an unchanged "no action" line
    # does, making it easy to lose track of what's actually being priced
    # this cycle.
    if eligible:
        print("  --- FAIR PRICES (this cycle) ---")
        for line in sorted(eligible, key=lambda l: l["ticker"]):
            side = line["target_side"]
            fair_cents = line["fair_yes_cents"] if side == "yes" else line["fair_no_cents"]
            print(f"    [{line['ticker']}] {line['tenor']} strike={line['strike_pct']} "
                  f"target_side={side.upper()} fair={fair_cents:.2f}c")

        print("  --- QUOTES (currently resting, this cycle) ---")
        for line in sorted(eligible, key=lambda l: l["ticker"]):
            ticker, side = line["ticker"], line["target_side"]
            price, size = resting_price_and_size(orders_by_ticker.get(ticker, []), side)
            if price is None:
                print(f"    [{ticker}] {side.upper()}: nothing resting")
            else:
                print(f"    [{ticker}] {side.upper()}: {price}c x{size:.0f} contracts "
                      f"(${price/100*size:.2f} notional)")

    strays = find_stray_orders(orders_by_ticker, eligible_by_ticker, positions)
    if strays:
        print(f"  --- STRAY ORDERS DETECTED ({len(strays)}) "
              f"{'[DRY RUN -- reporting only]' if dry_run else ''} ---")
        for s in strays:
            print(f"    [{s['ticker']}] {s['side'].upper()} @ {s['price_cents']}c "
                  f"x{s['size']:.0f} -- {s['reason']}")
        if not dry_run and AUTO_CANCEL_STRAY_ORDERS:
            for s in strays:
                print(f"    Pulling stray {s['side'].upper()} on {s['ticker']}...")
                kc.pull_side(s["ticker"], s["side"])
        elif not dry_run:
            print("    AUTO_CANCEL_STRAY_ORDERS is False -- not pulling automatically. "
                  "Review the list above and either cancel manually or set that flag to True.")

    # --- Teardown: only fully shut down a bot if it's BOTH ineligible for
    # market-making AND holds no position. If it still holds a position, keep
    # it alive (market-making paused) purely so the closing-quote pass below
    # can keep managing its exit -- otherwise a position on a now-ineligible
    # ticker would be abandoned with no bot left to close it out. ---
    for ticker in list(bots):
        if ticker not in eligible_by_ticker:
            if ticker in held_tickers:
                print(f"  [{ticker}] no longer eligible for market-making, but still holds a "
                      f"position -- pausing market-making, keeping bot alive for closing quote only.")
            else:
                print(f"  [{ticker}] no longer eligible -- shutting down and pulling quotes.")
                bots[ticker].shutdown()
                del bots[ticker]

    for ticker, line in eligible_by_ticker.items():
        if ticker not in bots:
            print(f"  [{ticker}] newly eligible (tenor={line['tenor']} strike={line['strike_pct']} "
                  f"target_side={line['target_side']}) -- starting quoter.")
            price_fn = fv.make_price_fn(line)
            bots[ticker] = BestBidOrFairAutoquoter(
                kc, ticker=ticker, price_fn=price_fn,
                quote_notional_dollars=effective_ceiling,  # ALWAYS quote the full current
                                                             # ceiling, not a separate smaller
                                                             # target -- see run_one_cycle's
                                                             # refresh of this on every cycle.
                total_budget_dollars=effective_ceiling,
                dry_run=dry_run, verbose=VERBOSE, label=ticker,
            )
        else:
            # Refresh BOTH quote_notional_dollars and total_budget_dollars
            # on every cycle, even for already-running bots. This is the
            # actual sizing target: desired_size in maybe_requote_side is
            # min(quote_notional_dollars // price, total_budget_dollars //
            # price) -- if quote_notional_dollars stayed fixed at some
            # smaller number, it would silently cap every quote below the
            # ceiling regardless of how much budget is actually available
            # (this is exactly the bug that shipped once already: a leftover
            # QUOTE_NOTIONAL_DOLLARS=$10 default was capping every line to
            # ~10 contracts even with a full $511.94 ceiling free). Setting
            # both to effective_ceiling makes "quote at max" the only thing
            # that's ever possible, no separate cap to forget about.
            bots[ticker].quote_notional_dollars = effective_ceiling
            bots[ticker].total_budget_dollars = effective_ceiling

    # --- Positions with NO corresponding bot at all (e.g. filled before this
    # session ever tracked the ticker, or on a series never marked eligible)
    # still need to be closed out. Create a minimal closing-quote-only bot
    # for these -- manage_closing_quote() never calls price_fn(), so a
    # placeholder is fine; this bot will never run market-making (it's not
    # in eligible_by_ticker) but IS included in the closing-quote pass below. ---
    for ticker in held_tickers:
        if ticker not in bots:
            print(f"  [{ticker}] holds a position but was never tracked this session -- "
                  f"creating a closing-quote-only bot to manage the exit.")
            placeholder_price_fn = lambda: {"fair_yes_cents": 50.0, "fair_no_cents": 50.0,
                                             "spot": 0.0, "strike": 0.0}
            bots[ticker] = BestBidOrFairAutoquoter(
                kc, ticker=ticker, price_fn=placeholder_price_fn,
                total_budget_dollars=effective_ceiling,
                dry_run=dry_run, verbose=VERBOSE, label=ticker,
            )

    if not bots:
        return

    # --- PASS 1: market-making, only for currently-eligible tickers ---
    market_making_bots = {t: b for t, b in bots.items() if t in eligible_by_ticker}
    if market_making_bots:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {
                executor.submit(bot.run_cycle, positions, orders_by_ticker.get(ticker, [])): ticker
                for ticker, bot in market_making_bots.items()
            }
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    future.result()
                except Exception as e:  # noqa: BLE001
                    print(f"  [{ticker}] ERROR during run_cycle: {type(e).__name__}: {e}")

    # --- PASS 2: closing quotes, for EVERY ticker with a held position
    # (whether or not it's currently eligible for market-making), run
    # entirely separately from pass 1 -- see Autoquoter.run_cycle's
    # "TWO-PASS ARCHITECTURE" docstring note for why these can't safely be
    # interleaved. Each needs its own fresh orderbook fetch (best bid/ask
    # per side), since run_cycle no longer exposes those externally. ---
    closing_bots = {t: b for t, b in bots.items() if t in held_tickers}
    if closing_bots:
        def _run_closing_quote(ticker: str, bot: BestBidOrFairAutoquoter) -> None:
            market = kc.best_market_prices(kc.get_orderbook(ticker))
            if market is not None:
                best_yes_bid, best_yes_ask = market
                best_no_bid, best_no_ask = 100 - best_yes_ask, 100 - best_yes_bid
            else:
                best_yes_bid = best_yes_ask = best_no_bid = best_no_ask = None
            bot.manage_closing_quote(best_yes_bid, best_no_bid, positions,
                                      best_yes_ask=best_yes_ask, best_no_ask=best_no_ask)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {
                executor.submit(_run_closing_quote, ticker, bot): ticker
                for ticker, bot in closing_bots.items()
            }
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    future.result()
                except Exception as e:  # noqa: BLE001
                    print(f"  [{ticker}] ERROR during closing-quote pass: {type(e).__name__}: {e}")


def main():
    kc = KalshiClient(key_id=KEY_ID, private_key_path=PRIVATE_KEY_PATH)
    fv = KalshiTreasuryFairValue(kc, tenors=ENABLED_TENORS, maturity_buffer_minutes=MATURITY_BUFFER_MINUTES)

    print(f"{'DRY RUN — ' if DRY_RUN else 'LIVE — '}Best-Bid-or-Fair Treasury Quoter")
    print(f"Tenors: {', '.join(ENABLED_TENORS)}")
    print(f"Poll interval: {POLL_INTERVAL_SEC}s | Budget ceiling (per line, shrinks with "
          f"real fills): ${GLOBAL_BUDGET_DOLLARS:.2f}")
    print(f"Maturity buffer: +{MATURITY_BUFFER_MINUTES:.0f}min applied to pricing "
          f"(displayed maturity stays true, unbuffered)\n")

    bots: dict[str, BestBidOrFairAutoquoter] = {}
    cycle = 0
    while True:
        cycle += 1
        print(f"\n=== Cycle {cycle} — {datetime.now(timezone.utc).isoformat()} ===")
        cycle_start = time.monotonic()
        try:
            run_one_cycle(kc, fv, bots, dry_run=DRY_RUN)
        except Exception as e:  # noqa: BLE001
            print(f"ERROR during cycle: {type(e).__name__}: {e}")
            print("Continuing to next cycle -- the budget ceiling recomputes from real "
                  "filled exposure every cycle, so it stays correct even if a cycle "
                  "errors out partway through.")
        cycle_elapsed = time.monotonic() - cycle_start
        print(f"  [cycle {cycle} took {cycle_elapsed:.1f}s]")
        time.sleep(POLL_INTERVAL_SEC)


if __name__ == "__main__":
    main()