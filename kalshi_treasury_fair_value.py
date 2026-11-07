"""
KalshiTreasuryFairValue: class-based matching of Kalshi's live Treasury-
yield strike ladder (AD variant) to CIR fair values.

Combines:
  - KalshiTreasuryLines (kalshi_treasury_lines.py) -- pulls every open
    AD-variant line/strike directly from Kalshi (ticker, tenor, strike,
    settlement time).
  - CIRTreasuryPricer (cir_three_way_comparison.py) -- calibrated CIR fair
    value, same model used everywhere else in this project.

Each line is priced using its ACTUAL remaining time to settlement (pulled
from Kalshi's own occurrence_datetime/close_time via KalshiTreasuryLines),
not a fixed maturity assumption.

Usage:
    kc = KalshiClient(key_id=..., private_key_path=...)
    fv = KalshiTreasuryFairValue(kc)
    matched = fv.refresh()              # calibrates (first call only),
                                         # fetches live yields, prices every line
    eligible = fv.get_eligible_lines()  # just the lines where fair_yes or
                                         # fair_no exceeds the threshold

    # Hook for Autoquoter -- one line at a time:
    line = eligible[0]
    price_fn = fv.make_price_fn(line)
    bot = Autoquoter(kc, ticker=line["ticker"], price_fn=price_fn)
"""
from __future__ import annotations

import os

from datetime import datetime, timezone

from KalshiClient import KalshiClient
from kalshi_treasury_lines import KalshiTreasuryLines, business_hours_between
from cir_three_way_comparison import CIRTreasuryPricer
from live_treasury_yields_pyth import get_live_treasury_yields
# Back to the original single-file feed. Only this one file is needed --
# no dependency on live_treasury_yields_hybrid.py or live_treasury_yields_
# official.py. Note: raw Pyth Rates.* tracks the on-the-run note's live
# market yield, which was measured diverging from Treasury's own official
# CMT close by a few bps (confirmed in a 3-way comparison against CNBC).
# That's a real, known tradeoff of this simpler setup, not something to
# forget -- but if it's not currently worth the added file-management
# complexity, this is the simplest thing that works.

DEFAULT_FAIR_VALUE_TARGET_THRESHOLD_CENTS = 90  # 個人智慧財產,參數不提供 (placeholder)


class KalshiTreasuryFairValue:
    def __init__(
        self,
        kc: KalshiClient,
        pricer: CIRTreasuryPricer | None = None,
        fair_value_target_threshold_cents: float = DEFAULT_FAIR_VALUE_TARGET_THRESHOLD_CENTS,
        tenors: tuple[str, ...] | None = None,
        maturity_buffer_minutes: float = 30.0,
    ):
        self.kc = kc
        self.pricer = pricer or CIRTreasuryPricer()
        self.fair_value_target_threshold_cents = fair_value_target_threshold_cents

        # Deliberately treats every line as having this many EXTRA minutes
        # of unresolved uncertainty beyond the raw clock-based countdown to
        # settlement_utc. Rationale: the CIR model's uncertainty comes
        # purely from statistical diffusion up to a hard maturity moment --
        # it has no way to represent a SEPARATE, non-statistical risk that
        # our live rate feed might not exactly match whatever Treasury
        # ultimately announces (feed lag, revision, a discrepancy like the
        # Pyth-vs-official gap found earlier in this project). Padding
        # maturity_hours by this buffer keeps the model from claiming
        # near-100% certainty in the final stretch before the real
        # settlement moment, when that operational risk is proportionally
        # largest. Applied consistently in BOTH refresh() (which determines
        # ELIGIBILITY) and make_price_fn() (which prices LIVE quotes) --
        # using it in only one of the two would let a line's eligibility
        # and its actual quoted price be computed from different maturity
        # assumptions, which would be a real, confusing inconsistency.
        self.maturity_buffer_minutes = maturity_buffer_minutes

        # Passing tenors through restricts the ENTIRE pipeline (series
        # discovery, per-ticker get_market() calls, live yield fetches) to
        # only the tenors you actually want quoted -- not just filtering
        # the OUTPUT after the fact. If you only want US2Y and US5Y, this
        # means never even discovering or pricing US7Y/US10Y/US30Y lines,
        # which is where most of the API-call cost actually is (one
        # get_market() call per open strike, per tenor). Leave as None to
        # keep the previous default behavior (all 5 tenors).
        self.lines_finder = KalshiTreasuryLines(kc, tenors=tenors) if tenors else KalshiTreasuryLines(kc)
        self._calibrated = False
        self._matched: list[dict] = []
        self._live_yields: dict | None = None  # shared across all price_fn() calls in a cycle

    # -----------------------------------------------------------------
    # CALIBRATION (cached -- same convention as CIRTreasuryPricer itself)
    # -----------------------------------------------------------------
    def calibrate(self) -> None:
        if not self._calibrated:
            self.pricer.calibrate()
            self._calibrated = True

    # -----------------------------------------------------------------
    # MATCHING: pull lines, price each one
    # -----------------------------------------------------------------
    def refresh(self, live_yields: dict | None = None) -> list[dict]:
        """Pulls every open AD-variant Kalshi line and prices each one with
        the CIR model, using its actual remaining time to settlement.
        Calibrates on first call if not already done. Fetches live yields
        (once, reused across every line) if not provided. Caches and
        returns the matched list."""
        self.calibrate()
        if live_yields is None:
            live_yields = get_live_treasury_yields()
        self._live_yields = live_yields  # shared with make_price_fn()'s closures -- see there for why

        lines = self.lines_finder.refresh()

        matched = []
        for line in lines:
            tenor = line["tenor"]
            ticker = line["ticker"]

            if tenor not in live_yields:
                print(f"  SKIP {ticker}: no live yield available for tenor {tenor}")
                continue
            if line["strike_raw"] is None:
                print(f"  SKIP {ticker}: no strike available")
                continue
            if line["hours_remaining"] is None:
                print(f"  SKIP {ticker}: no settlement time available")
                continue

            hours_remaining = line["hours_remaining"]
            if hours_remaining <= 0:
                print(f"  SKIP {ticker}: already past settlement")
                continue

            # Buffer applied AFTER the above check (which correctly uses the
            # REAL clock-based remaining time to decide whether a line is
            # even still tradeable) -- only the value fed to the PRICING
            # model gets padded. See maturity_buffer_minutes' docstring in
            # __init__ for why.
            buffered_hours_remaining = hours_remaining + (self.maturity_buffer_minutes / 60.0)

            r0_pct = live_yields[tenor]["yield_pct"]
            priced = self.pricer.price_contract(
                tenor, strike_pct=line["strike_raw"], r0_pct=r0_pct,
                maturity_hours=buffered_hours_remaining,
            )

            matched.append({
                "ticker": ticker,
                "tenor": tenor,
                "strike_pct": line["strike_raw"],
                "hours_remaining": hours_remaining,
                "settlement_utc": line["settlement_utc"],
                "r0_pct": r0_pct,
                "fair_yes_cents": priced["fair_yes_cents"],
                "fair_no_cents": priced["fair_no_cents"],
                "settlement_is_fallback": line["settlement_is_fallback"],
            })

        self._matched = matched
        return matched

    def get_matched_lines(self) -> list[dict]:
        """Returns the cached matched lines from the last refresh() call,
        without hitting the API again."""
        return self._matched

    # -----------------------------------------------------------------
    # ELIGIBLE LINES: where the model puts one side near-certain
    # -----------------------------------------------------------------
    def get_eligible_lines(self, threshold: float | None = None) -> list[dict]:
        """Returns just the lines where fair_yes_cents or fair_no_cents
        exceeds threshold (default: self.fair_value_target_threshold_cents).
        Tags every line in self._matched with 'is_target'/'target_side' IN
        PLACE (so get_matched_lines() reflects the same tags), and returns
        only the ones flagged True.

        Since fair_yes_cents + fair_no_cents == 100 always, at most ONE
        side can exceed the threshold per line -- target_side is
        unambiguous whenever is_target is True."""
        threshold = self.fair_value_target_threshold_cents if threshold is None else threshold

        eligible = []
        for m in self._matched:
            is_yes_target = m["fair_yes_cents"] > threshold
            is_no_target = m["fair_no_cents"] > threshold
            m["is_target"] = is_yes_target or is_no_target
            m["target_side"] = "yes" if is_yes_target else ("no" if is_no_target else None)
            if m["is_target"]:
                eligible.append(m)
        return eligible

    # -----------------------------------------------------------------
    # AUTOQUOTER HOOK
    # -----------------------------------------------------------------
    def make_price_fn(self, line: dict):
        """Returns a zero-arg callable matching Autoquoter's price_fn
        signature: () -> {fair_yes_cents, fair_no_cents, spot, strike,
        hours_remaining}.

        PERFORMANCE NOTE -- this used to call get_live_treasury_yields()
        (all 5 tenors, its own internal 5-thread pool) INDEPENDENTLY on
        every call. With N eligible lines each getting their own bot, and
        each bot calling price_fn() once per cycle, that meant N*5
        redundant Pyth fetches per cycle (e.g. 30 bots -> 150 fetches when
        only 5 unique values exist), plus N nested thread pools spawning
        concurrently -- the actual cause of high CPU usage under load, not
        underpowered scalar math (GPU wouldn't help here; the bottleneck
        was redundant thread/network churn, not compute).

        Fix: this now reads self._live_yields, which refresh() populates
        ONCE per orchestrator cycle (already being fetched anyway, for the
        eligibility check) and every bot's price_fn() reuses that SAME
        cached dict instead of fetching its own copy. Net effect: same
        freshness (still one fetch per cycle, not per bot), 1/N the
        network calls and thread churn. Falls back to a fresh fetch only
        if called before refresh() has ever run (e.g. standalone/test
        usage of make_price_fn() without going through the normal
        refresh()-then-run_cycle() flow).

        Time-to-settlement is still recomputed fresh on every call
        (cheap, local, no network/thread cost) -- only the live RATE is
        now cycle-cached, not the time decay.
        """
        tenor = line["tenor"]
        strike_pct = line["strike_pct"]
        settlement_utc = line["settlement_utc"]

        def _price_fn():
            live = self._live_yields if self._live_yields is not None else get_live_treasury_yields()
            r0_pct = live[tenor]["yield_pct"]

            if settlement_utc is not None:
                now = datetime.now(timezone.utc)
                if self.lines_finder.use_business_hours:
                    hours_remaining = business_hours_between(now, settlement_utc)
                else:
                    hours_remaining = max((settlement_utc - now).total_seconds() / 3600, 0)
            else:
                hours_remaining = line["hours_remaining"]  # fallback to the stale value

            # Same buffer as refresh() (see maturity_buffer_minutes'
            # docstring in __init__) -- padding ONLY the value fed to the
            # pricing model, not the 'hours_remaining' returned below (that
            # one drives the displayed "maturity=X.Xh" in every log line,
            # which should show the TRUE remaining time, not a padded one).
            buffered_hours_remaining = hours_remaining + (self.maturity_buffer_minutes / 60.0)

            priced = self.pricer.price_contract(
                tenor, strike_pct=strike_pct, r0_pct=r0_pct, maturity_hours=buffered_hours_remaining,
            )
            return {
                "fair_yes_cents": priced["fair_yes_cents"],
                "fair_no_cents": priced["fair_no_cents"],
                "spot": r0_pct / 100.0,
                "strike": strike_pct / 100.0,
                "hours_remaining": hours_remaining,
                "priced_maturity_hours": buffered_hours_remaining,  # what price_contract() ACTUALLY
                                                                      # used -- previously computed and
                                                                      # then discarded, meaning the
                                                                      # buffer's effect was invisible
                                                                      # anywhere in the output.
            }
        return _price_fn


if __name__ == "__main__":
    KEY_ID = os.environ.get("KALSHI_KEY_ID", "")
    PRIVATE_KEY_PATH = os.environ.get("KALSHI_PRIVATE_KEY_PATH", "kalshi_private_key.pem")

    kc = KalshiClient(key_id=KEY_ID, private_key_path=PRIVATE_KEY_PATH)
    fv = KalshiTreasuryFairValue(kc)

    print("Calibrating CIR parameters from historical Treasury data...")
    fv.calibrate()

    print("Pulling Kalshi lines and matching to fair value...\n")
    matched = fv.refresh()

    print(f"\n{len(matched)} lines priced.\n")
    print(f"{'ticker':<30} {'tenor':<7} {'strike':>8} {'hrs_left':>9} "
          f"{'fair_yes_c':>11} {'fair_no_c':>10}")
    for m in matched:
        fallback_flag = " *" if m["settlement_is_fallback"] else ""
        print(f"{m['ticker']:<30} {m['tenor']:<7} {m['strike_pct']:8.4f} "
              f"{m['hours_remaining']:9.2f} {m['fair_yes_cents']:11.4f} "
              f"{m['fair_no_cents']:10.4f}{fallback_flag}")

    if any(m["settlement_is_fallback"] for m in matched):
        print("\n* = settlement time used close_time as a fallback (no "
              "occurrence_datetime available) -- may be less accurate.")

    eligible = fv.get_eligible_lines()
    print(f"\n--- ELIGIBLE LINES (fair_yes or fair_no > "
          f"{fv.fair_value_target_threshold_cents}c) ({len(eligible)}/{len(matched)}) ---")
    if not eligible:
        print("  (none right now)")
    else:
        print(f"{'ticker':<30} {'tenor':<7} {'strike':>8} {'hrs_left':>9} "
              f"{'target_side':>11} {'fair_cents':>10}")
        for m in eligible:
            fair_cents = m["fair_yes_cents"] if m["target_side"] == "yes" else m["fair_no_cents"]
            print(f"{m['ticker']:<30} {m['tenor']:<7} {m['strike_pct']:8.4f} "
                  f"{m['hours_remaining']:9.2f} {m['target_side']:>11} {fair_cents:10.4f}")