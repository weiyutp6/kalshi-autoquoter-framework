"""
KalshiTreasuryLines: class-based automatic line/strike discovery for
Kalshi's Treasury-yield (KXUST*) markets, currently locked to the AD
(Daily) variant per tenor -- confirmed via the live Kalshi site to be the
actively-traded one (e.g. KXUST2AD, kalshi.com/markets/kxust2ad/...).

CONFIRMED STRUCTURE (from a live diagnostic dump against category==
'Financials'):
  - Tenor = the digits immediately after "KXUST" (2, 5, 7, 10, 30).
  - Multiple frequency-suffix variants exist per tenor (bare/A/AD/AW/AM/M);
    AD is the one to use for now. Note not every tenor's AD ticker settles
    on the same cadence necessarily -- each is its own series, discovered
    independently.

Usage:
    kc = KalshiClient(key_id=..., private_key_path=...)
    lines_finder = KalshiTreasuryLines(kc)
    lines = lines_finder.refresh()          # discovers series + pulls all open lines
    # or, if you've already called refresh() once and just want the cached data:
    lines = lines_finder.get_lines()

Liquidity-reward eligibility is intentionally NOT part of this class right
now (that check needs more diagnosis -- see kalshi_incentive_diagnostic.py)
-- this only handles line/strike discovery.
"""
from __future__ import annotations

import os

import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from KalshiClient import KalshiClient

NY = ZoneInfo("America/New_York")

_US_HOLIDAY_CALENDAR = None  # lazy-loaded, see business_hours_between()


def business_hours_between(start: datetime, end: datetime, exclude_holidays: bool = True) -> float:
    """Hours between start and end, counting ONLY time that falls on a US
    business day (Mon-Fri, optionally excluding US federal holidays as a
    reasonable free proxy for the actual SIFMA bond market holiday
    calendar -- close enough for this purpose, not pretending to be
    perfectly precise). Weekends (and holidays, if excluded) contribute
    ZERO hours, regardless of how many calendar hours they span.

    WHY: Treasury's official Daily Par Yield Curve mechanically doesn't
    exist on weekends -- there's no Saturday/Sunday observation, the
    sequence only ever advances Friday -> Monday. More importantly, this
    project's own CIR calibration (calibrate_treasury.py) was fit via a
    discretized Euler scheme on the historical CSV, which only contains
    business-day rows -- every consecutive row-to-row step was treated as
    ONE EQUAL-SIZED unit of time in that fit, regardless of whether it
    spanned 1 calendar day (weekday-to-weekday) or 3 calendar days
    (Friday-to-Monday). That means the calibrated sigma is ALREADY
    implicitly a per-business-day volatility, not a per-calendar-day one.
    Feeding maturity_hours in raw calendar time (including weekend hours)
    creates a real, mechanically-motivated mismatch between how the model
    was calibrated and how it's used to price live -- this function fixes
    that by measuring time the same way the calibration implicitly did.

    Handles partial start/end days correctly (e.g. "3:21pm Friday to
    7:00pm Monday" correctly counts only the Friday-afternoon remainder
    plus Monday-up-to-7pm, zero for the weekend in between).
    """
    global _US_HOLIDAY_CALENDAR
    if end <= start:
        return 0.0

    holiday_dates = set()
    if exclude_holidays:
        if _US_HOLIDAY_CALENDAR is None:
            from pandas.tseries.holiday import USFederalHolidayCalendar
            _US_HOLIDAY_CALENDAR = USFederalHolidayCalendar()
        holidays = _US_HOLIDAY_CALENDAR.holidays(start=start.date(), end=end.date())
        holiday_dates = {ts.date() for ts in holidays}

    total_hours = 0.0
    cur = start
    while cur < end:
        day_end = min(end, datetime(cur.year, cur.month, cur.day, tzinfo=cur.tzinfo) + timedelta(days=1))
        is_weekday = cur.weekday() < 5  # Mon=0 ... Fri=4, Sat=5, Sun=6
        is_holiday = cur.date() in holiday_dates
        if is_weekday and not is_holiday:
            total_hours += (day_end - cur).total_seconds() / 3600
        cur = day_end
    return total_hours

TICKER_PREFIX = "KXUST"
TENOR_REGEX = re.compile(r"^KXUST(\d+)([A-Z]*)$")
DEFAULT_TENORS = ("US2Y", "US5Y", "US7Y", "US10Y", "US30Y")
DEFAULT_SUFFIX = "AD"
DEFAULT_SERIES_REDISCOVERY_INTERVAL_SEC = 3600  # re-fetch the full ~12000-series — 個人智慧財產,參數不提供 (placeholder)
                                                  # list at most once per hour --
                                                  # the 5 KXUST*AD tickers this
                                                  # bot cares about don't change
                                                  # within a session, so paying
                                                  # ~61 sequential paginated
                                                  # /series requests EVERY
                                                  # polling cycle (every 30s) was
                                                  # pure waste and a real
                                                  # contributor to CPU load.


class KalshiTreasuryLines:
    def __init__(
        self,
        kc: KalshiClient,
        tenors: tuple[str, ...] = DEFAULT_TENORS,
        suffix: str = DEFAULT_SUFFIX,
        series_rediscovery_interval_sec: float = DEFAULT_SERIES_REDISCOVERY_INTERVAL_SEC,
        daily_settlement_et: str = "15:30:00",
        manual_settlement_override_et: str | None = None,
        use_business_hours: bool = True,
    ):
        self.kc = kc
        self.tenors = tenors
        self.suffix = suffix
        self.series_rediscovery_interval_sec = series_rediscovery_interval_sec
        # CONFIRMED, not guessed: a real diagnostic run (settlement_is_
        # fallback=False for every line) showed Kalshi's own authoritative
        # occurrence_datetime field resolves to 19:30 UTC = 3:30 PM ET for
        # these markets -- this is NOT this fallback constant's doing, it's
        # what Kalshi's API directly returns. 3:30 PM ET is also exactly
        # when the U.S. Treasury takes its official Daily Par Yield Curve
        # snapshot, which is what these contracts' rules_primary text
        # actually resolves against ("the par yield ... on [date]") -- so
        # this isn't a coincidence, it's the real mechanism.
        # occurrence_datetime has been reliably populated for every line
        # checked so far, meaning this fallback constant currently has NO
        # effect on real behavior at all -- it only matters if some future
        # ticker is ever missing occurrence_datetime, in which case this is
        # a sane default matching the confirmed-correct value rather than
        # an earlier guess.
        self.daily_settlement_et = daily_settlement_et

        # DELIBERATE OVERRIDE -- when set, this takes priority over EVEN
        # Kalshi's own authoritative occurrence_datetime field, not just the
        # fallback path above. Use this ONLY when you have a specific,
        # considered reason to distrust occurrence_datetime for a market
        # (e.g. the API and Kalshi's own website genuinely disagree, as
        # confirmed happening here -- API said 3:30pm ET, site said 7pm ET).
        # This is a real, consequential choice, not a safe default -- every
        # line's returned dict includes 'settlement_source' so it's always
        # visible which of the three possible sources (manual_override,
        # occurrence_datetime, or close_time_fallback) actually determined
        # that line's settlement time and therefore its priced maturity.
        self.manual_settlement_override_et = manual_settlement_override_et

        # See business_hours_between()'s docstring for the full reasoning.
        # Default True: this project's own CIR calibration was implicitly
        # fit on business-day-indexed data, so business-hours maturity is
        # the more CORRECT choice, not just an alternative convention. Set
        # False to revert to raw calendar hours (including weekends) if
        # you want to compare or have a specific reason to prefer it.
        self.use_business_hours = use_business_hours

        self._series_map: dict[str, str] = {}   # series_ticker -> tenor_label, AD only
        self._lines: list[dict] = []
        self._series_last_discovered: float | None = None  # time.monotonic() of last discover_series() fetch

    # -----------------------------------------------------------------
    # SERIES DISCOVERY
    # -----------------------------------------------------------------
    def _fetch_all_series(self) -> list[dict]:
        all_series = []
        cursor = None
        while True:
            params = {"limit": 200}
            if cursor:
                params["cursor"] = cursor
            resp = self.kc.request("GET", "/series", params=params, auth=False)
            all_series.extend(resp.get("series", []) if isinstance(resp, dict) else resp)
            cursor = resp.get("cursor") if isinstance(resp, dict) else None
            if not cursor:
                break
        return all_series

    def discover_series(self, force: bool = False) -> dict[str, str]:
        """Finds the {suffix}-variant series ticker for each configured
        tenor (default: AD). Prints a warning for any tenor missing that
        variant instead of silently dropping it. Returns and caches
        {series_ticker: tenor_label}.

        CACHED: skips the actual ~12000-series paginated fetch entirely
        if it already has a result AND that result isn't older than
        series_rediscovery_interval_sec (default 1 hour) -- the specific
        KXUST*AD tickers this bot cares about don't change within a
        session, so there's no reason to pay ~61 sequential HTTP requests
        for this on every polling cycle. Pass force=True to bypass the
        cache and re-fetch immediately regardless of staleness.
        """
        if (not force and self._series_map
                and self._series_last_discovered is not None
                and (time.monotonic() - self._series_last_discovered) < self.series_rediscovery_interval_sec):
            return self._series_map

        all_series = self._fetch_all_series()

        variants: dict[str, dict[str, str]] = {}
        for s in all_series:
            ticker = s.get("ticker", "")
            if s.get("category") != "Financials":
                continue
            m = TENOR_REGEX.match(ticker.upper())
            if not m:
                continue
            tenor_label = f"US{m.group(1)}Y"
            variants.setdefault(tenor_label, {})[m.group(2)] = ticker

        series_map = {}
        for tenor in self.tenors:
            suffix_map = variants.get(tenor, {})
            if self.suffix in suffix_map:
                series_map[suffix_map[self.suffix]] = tenor
            else:
                print(f"  WARNING: no '{self.suffix}' variant found for {tenor}. "
                      f"Available variants: {suffix_map or '(none at all)'}")

        self._series_map = series_map
        self._series_last_discovered = time.monotonic()
        return series_map

    # -----------------------------------------------------------------
    # LINE/STRIKE PULLING
    # -----------------------------------------------------------------
    def _find_all_markets(self, series_ticker: str, status: str) -> list[dict]:
        """Paginated replacement for kc.find_markets(), which only makes a
        SINGLE unpaginated GET /markets call -- same class of bug already
        found and fixed for /series and /incentive_programs. If a series
        has more open strikes than fit on one page, everything past page 1
        was being silently dropped, which would explain a real strike
        (e.g. a manually-placed test order's ticker) never showing up in
        pull_lines()'s output at all -- not just missed by eligibility
        filtering, but never discovered in the first place, meaning the
        stray-order scanner had no way to catch it either.
        """
        markets = []
        cursor = None
        while True:
            params = {"series_ticker": series_ticker, "status": status, "limit": 200}
            if cursor:
                params["cursor"] = cursor
            resp = self.kc.request("GET", "/markets", params=params)
            markets.extend(resp.get("markets", []))
            cursor = resp.get("cursor")
            if not cursor:
                break
        return markets

    def pull_lines(self, status: str = "open", max_workers: int = 10) -> list[dict]:
        """Pulls every open strike across the discovered series. Calls
        discover_series() first -- which is itself cached/staleness-aware
        (see its docstring), so this does NOT force a full series
        re-fetch on every call despite calling it unconditionally here.

        For EACH ticker, find_markets() (the list endpoint) is only used to
        discover WHICH tickers exist -- it's not trusted for strike/
        settlement data, since list endpoints don't reliably include
        occurrence_datetime. Instead, get_market(ticker) is called per
        ticker to get the AUTHORITATIVE strike + settlement time, same
        pattern dxy_autoquoter.py already established (occurrence_datetime
        is the true resolution moment; close_time is only a fallback --
        it's when trading stops, not necessarily when the contract
        resolves). hours_remaining is computed against the current time
        AFTER that authoritative fetch, not against a stale timestamp.

        PARALLELIZED: with N tenors x M strikes each, this was N*M
        sequential get_market() round-trips (the dominant cost -- often
        70+ individual HTTP calls). Since each call only reads its own
        ticker and KalshiClient issues stateless requests.request() calls
        (not a shared Session), fetching them concurrently via a thread
        pool is safe -- there's no shared mutable state between calls.
        max_workers caps how many requests run at once (keep this
        reasonable to avoid tripping Kalshi's rate limits).

        Returns and caches a list of {ticker, tenor, series_ticker,
        strike_raw, settlement_utc, settlement_is_fallback, hours_remaining},
        sorted by ticker (thread completion order isn't submission order,
        so this keeps output stable/readable run to run).
        """
        self.discover_series()  # cached/staleness-aware -- see its docstring

        # Step 1: find out which tickers exist. Uses _find_all_markets()
        # (paginated) instead of kc.find_markets() (a single unpaginated
        # call) -- see _find_all_markets' docstring for why that matters.
        pending = []
        for series_ticker, tenor in self._series_map.items():
            market_stubs = self._find_all_markets(series_ticker, status=status)
            for stub in market_stubs:
                ticker = stub.get("ticker")
                if ticker:
                    pending.append((ticker, tenor, series_ticker))

        # Step 2: the expensive part -- one authoritative get_market() call
        # per ticker. This is what actually gets parallelized.
        def _fetch_one(item: tuple[str, str, str]) -> dict:
            ticker, tenor, series_ticker = item
            market_info = self.kc.get_market(ticker)

            strike_raw = market_info.get("floor_strike") or market_info.get("cap_strike")
            occurrence_str = market_info.get("occurrence_datetime")
            close_time_str = market_info.get("close_time")

            if self.manual_settlement_override_et:
                # HIGHEST PRIORITY -- deliberately overrides even Kalshi's
                # own authoritative occurrence_datetime. Still needs SOME
                # date to combine with the override time-of-day; prefers
                # occurrence_datetime's date (Kalshi's own confirmed date
                # for this specific market) if available, else close_time's
                # date, since the override is only meant to correct the
                # TIME, not guess at which calendar day this market is for.
                date_source = occurrence_str or close_time_str
                if date_source:
                    reference_utc = datetime.fromisoformat(date_source.replace("Z", "+00:00"))
                    reference_et_date = reference_utc.astimezone(NY).date()
                    hh, mm, ss = (int(x) for x in self.manual_settlement_override_et.split(":"))
                    settlement_local = datetime(
                        reference_et_date.year, reference_et_date.month, reference_et_date.day,
                        hh, mm, ss, tzinfo=NY,
                    )
                    settlement_utc = settlement_local.astimezone(timezone.utc)
                    settlement_is_fallback = False  # not a fallback -- a deliberate override
                    settlement_source = "manual_override"
                else:
                    settlement_utc, settlement_is_fallback, settlement_source = None, True, "none_available"
            elif occurrence_str:
                settlement_utc = datetime.fromisoformat(occurrence_str.replace("Z", "+00:00"))
                settlement_is_fallback = False
                settlement_source = "occurrence_datetime"
            elif close_time_str:
                # DON'T just trust close_time's literal timestamp -- it's
                # "when trading stops," not necessarily the true resolution
                # moment. IMPORTANT: this fallback time has been WRONG
                # twice already (first assumed ~2pm, then told 7pm, now
                # told 3:30pm) purely from being told a number verbally --
                # that pattern of guessing isn't reliable. rules_primary
                # (captured below, unconditionally, for every line) is the
                # actual source of truth: READ THAT TEXT DIRECTLY rather
                # than trusting this fallback's assumed time at all. This
                # constant is a best-effort default only, easily wrong for
                # any specific market -- verify against rules_primary
                # before trusting it for anything real.
                close_time_utc = datetime.fromisoformat(close_time_str.replace("Z", "+00:00"))
                close_time_et_date = close_time_utc.astimezone(NY).date()
                hh, mm, ss = (int(x) for x in self.daily_settlement_et.split(":"))
                settlement_local = datetime(
                    close_time_et_date.year, close_time_et_date.month, close_time_et_date.day,
                    hh, mm, ss, tzinfo=NY,
                )
                settlement_utc = settlement_local.astimezone(timezone.utc)
                settlement_is_fallback = True
                settlement_source = "close_time_fallback"
            else:
                settlement_utc, settlement_is_fallback, settlement_source = None, True, "none_available"

            now = datetime.now(timezone.utc)
            if settlement_utc is None:
                hours_remaining = None
            elif self.use_business_hours:
                hours_remaining = business_hours_between(now, settlement_utc)
            else:
                hours_remaining = max((settlement_utc - now).total_seconds() / 3600, 0)

            return {
                "ticker": ticker,
                "tenor": tenor,
                "series_ticker": series_ticker,
                "strike_raw": strike_raw,
                "settlement_utc": settlement_utc,
                "settlement_is_fallback": settlement_is_fallback,
                "settlement_source": settlement_source,  # "manual_override", "occurrence_datetime",
                                                            # "close_time_fallback", or "none_available"
                                                            # -- always visible which one actually
                                                            # determined this line's priced maturity
                "rules_primary": market_info.get("rules_primary"),  # the ACTUAL source of
                                                                      # truth for settlement
                                                                      # timing -- read this,
                                                                      # don't trust the
                                                                      # fallback time above
                "hours_remaining": hours_remaining,
            }

        lines = []
        if pending:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(_fetch_one, item) for item in pending]
                for future in as_completed(futures):
                    lines.append(future.result())

        lines.sort(key=lambda l: l["ticker"])

        self._lines = lines
        return lines

    def refresh(self, status: str = "open") -> list[dict]:
        """Convenience: pull_lines() (which itself calls the cached/
        staleness-aware discover_series() as needed)."""
        return self.pull_lines(status=status)

    def get_lines(self) -> list[dict]:
        """Returns the cached lines from the last pull_lines()/refresh()
        call, without hitting the API again."""
        return self._lines

    def get_lines_for_tenor(self, tenor: str) -> list[dict]:
        return [l for l in self._lines if l["tenor"] == tenor]


if __name__ == "__main__":
    KEY_ID = os.environ.get("KALSHI_KEY_ID", "")
    PRIVATE_KEY_PATH = os.environ.get("KALSHI_PRIVATE_KEY_PATH", "kalshi_private_key.pem")

    kc = KalshiClient(key_id=KEY_ID, private_key_path=PRIVATE_KEY_PATH)
    lines_finder = KalshiTreasuryLines(kc)

    print("Discovering AD series per tenor...")
    series_map = lines_finder.discover_series()
    print(f"  {series_map}\n")

    print("Pulling lines...")
    lines = lines_finder.refresh()
    print(f"  {len(lines)} total open strikes.\n")

    now = datetime.now(timezone.utc)
    print(f"{'ticker':<30} {'tenor':<7} {'strike':>8} {'hrs_left':>9}")
    for line in lines:
        hrs_left = line["hours_remaining"] if line["hours_remaining"] is not None else float("nan")
        print(f"{line['ticker']:<30} {line['tenor']:<7} {str(line['strike_raw']):>8} {hrs_left:9.1f}")

    print(f"\n{'='*70}")
    print("ACTUAL RULES TEXT PER TENOR -- read this directly, don't trust")
    print("daily_settlement_et's assumed fallback time without checking it")
    print(f"against what's actually printed below.")
    print(f"{'='*70}")
    seen_tenors = set()
    for line in lines:
        if line["tenor"] in seen_tenors:
            continue
        seen_tenors.add(line["tenor"])
        print(f"\n[{line['tenor']}] {line['ticker']}")
        print(f"  settlement_is_fallback={line['settlement_is_fallback']}  "
              f"(if True, the printed settlement_utc below used the ASSUMED "
              f"daily_settlement_et={lines_finder.daily_settlement_et}, not a value read "
              f"directly from Kalshi -- verify against rules_primary text below)")
        print(f"  computed settlement_utc: {line['settlement_utc']}")
        print(f"  rules_primary: {line['rules_primary']}")