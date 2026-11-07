"""
Live U.S. Treasury yields (2Y, 5Y, 7Y, 10Y, 30Y), sourced from Pyth's
InterestRate.*/USD BOND PRICE feeds instead of the Rates.* yield-direct
feeds -- same structure and interface as live_treasury_yields_pyth.py, but
does the actual price -> yield (YTM) conversion this project's earlier
notes flagged as a "dead end without an external coupon source."

WHY THIS EXISTS AS A SEPARATE SCRIPT: InterestRate.US5Y/USD (etc.) gives a
BOND PRICE (~99-101, per $100 face value) for the current on-the-run note,
not a yield. Converting price -> yield needs the note's exact coupon rate
and maturity date, which Pyth doesn't publish alongside the price. Those
come from TreasuryDirect's own public Auction Query API (TA_WS) instead --
free, official, no API key -- looked up once per day per tenor (the coupon/
maturity of the on-the-run note doesn't change intraday, only the price
does). The actual price->yield math is a standard semiannual bond-equivalent
yield solve (Street/ICMA convention, Actual/Actual day count) via bisection,
the same method Excel's YIELD() function and Bloomberg's YAS page use.

WHETHER THIS IS BETTER OR WORSE THAN Rates.* (live_treasury_yields_pyth.py)
OR THE HYBRID/OFFICIAL MODULES: genuinely worth treating as an open
question, not an assumed improvement -- run this alongside them and compare.
This still tracks the on-the-run NOTE's market yield (same category of
methodology gap against Treasury's official CMT curve that motivated
live_treasury_yields_hybrid.py in the first place), just computed via price
instead of pulled as yield directly. It's most useful as an independent
CROSS-CHECK against the Rates.* feed (if they disagree sharply, that's a
strong signal one of them is wrong) rather than a guaranteed fix on its own.
"""
from __future__ import annotations

import os

import time
from datetime import datetime, date, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from scipy.optimize import brentq

HERMES = "https://hermes.pyth.network"
HERMES_NEW = "https://pyth.dourolabs.app/hermes"
TREASURY_DIRECT_SEARCH = "https://www.treasurydirect.gov/TA_WS/securities/search"

TENORS = ["US2Y", "US5Y", "US7Y", "US10Y", "US30Y"]

# Maps this project's tenor labels to TreasuryDirect's securityTerm values
# (what a FRESH original auction is labeled). Reopenings of the same note
# get shorter remaining-term labels (e.g. a 5-Year note reopened after 3
# months shows as "4-Year 9-Month"), which is why the on-the-run lookup
# below matches on the LEADING term number and takes the most-recently-
# issued match, rather than requiring an exact string match.
TENOR_TO_SECURITY_TERM_YEARS = {
    "US2Y": 2, "US5Y": 5, "US7Y": 7, "US10Y": 10, "US30Y": 30,
}
TENOR_TO_SECURITY_TYPE = {
    "US2Y": "Note", "US5Y": "Note", "US7Y": "Note", "US10Y": "Note", "US30Y": "Bond",
}

# Same pinning pattern as live_treasury_yields_pyth.py -- see that file's
# PINNED_FEED_IDS docstring for the full reasoning. These are DIFFERENT
# feed IDs than the Rates.* ones (different feed family entirely).
PINNED_FEED_IDS: dict[str, str | None] = {
    "US2Y": None,
    "US5Y": None,
    "US7Y": None,
    "US10Y": None,
    "US30Y": None,
}

# On-the-run note reference data (coupon, maturity, dated date) only
# changes when a new note is auctioned -- roughly quarterly per tenor, not
# intraday. Cached per calendar day per tenor to avoid hammering
# TreasuryDirect every cycle for data that provably hasn't changed.
_ref_cache: dict[str, dict] = {}
_ref_cache_date: dict[str, date] = {}


def _get_json(url: str, params=None, retries: int = 4):
    err = None
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=15)
            if r.status_code == 429:
                err = f"429 rate limited (attempt {i+1}/{retries})"
                time.sleep(min(2 ** i, 8))
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
            if i + 1 < retries:
                time.sleep(min(2 ** i, 8))
    raise RuntimeError(f"Request failed: {url}: {err}")


# ---------------------------------------------------------------------
# PYTH: discover feed id -> fetch latest BOND PRICE (not yield)
# ---------------------------------------------------------------------
def discover_price_feed_id(tenor: str) -> str:
    """Finds the Pyth feed id for InterestRate.<tenor>/USD. Same
    pin-first-then-fuzzy-search pattern as discover_rates_feed_id() in
    live_treasury_yields_pyth.py."""
    pinned = PINNED_FEED_IDS.get(tenor)
    if pinned:
        return pinned

    all_candidates_seen = []  # for self-diagnosis if every attempt fails

    for base in (HERMES, HERMES_NEW):
        # Try WITH the "rates" asset_type filter first (matches how
        # discover_rates_feed_id() in live_treasury_yields_pyth.py does
        # it), then WITHOUT any asset_type filter at all as a fallback --
        # the InterestRate.*/USD bond-price family may simply not be
        # catalogued under asset_type=rates the way Rates.* is, since
        # they're a structurally different kind of feed (a price, not a
        # rate). Relying on the exact symbol match below to narrow results
        # is safe either way.
        for params in ({"query": tenor, "asset_type": "rates"}, {"query": tenor}):
            try:
                data = _get_json(f"{base}/v2/price_feeds", params)
            except RuntimeError:
                continue
            items = data if isinstance(data, list) else data.get("price_feeds", data.get("data", []))
            for x in items:
                attrs = x.get("attributes", {})
                symbol = str(attrs.get("symbol") or x.get("symbol") or "")
                all_candidates_seen.append(symbol)
                if symbol.upper() == f"INTERESTRATE.{tenor}/USD":
                    fid = x.get("id") or x.get("price_feed_id") or attrs.get("id")
                    if fid:
                        return fid if fid.startswith("0x") else "0x" + fid

    unique_candidates = sorted(set(all_candidates_seen))
    raise RuntimeError(
        f"Could not find InterestRate.{tenor}/USD feed id. "
        f"Searched with and without asset_type=rates, across both Hermes endpoints. "
        f"Symbols actually returned by the search (for manual inspection -- the "
        f"correct one may just be named slightly differently than expected): "
        f"{unique_candidates if unique_candidates else '(none -- the search itself returned nothing at all)'}"
    )


def latest_bond_price(feed_id: str) -> tuple[float, int]:
    """Returns (price_per_100_face, publish_time_unix)."""
    for base in (HERMES, HERMES_NEW):
        try:
            d = _get_json(f"{base}/v2/updates/price/latest", [("ids[]", feed_id)])
            p = d["parsed"][0]["price"]
            scale = 10.0 ** int(p["expo"])
            return float(p["price"]) * scale, int(p["publish_time"])
        except Exception:  # noqa: BLE001
            continue
    raise RuntimeError(f"Could not fetch latest price for feed {feed_id}")


# ---------------------------------------------------------------------
# TREASURYDIRECT: look up the on-the-run note's coupon/maturity/dated date
# ---------------------------------------------------------------------
def lookup_on_the_run_reference(tenor: str, force_refresh: bool = False) -> dict:
    """Returns {'cusip', 'coupon_pct', 'maturity_date', 'dated_date'} for
    the CURRENT on-the-run security for this tenor, via TreasuryDirect's
    free public Auction Query API. Cached per calendar day per tenor.

    Matches on the LEADING year count of securityTerm (e.g. "5-Year",
    "4-Year 9-Month" both match tenor US5Y's target of 5) rather than an
    exact string, since reopenings of the same original note carry a
    shorter remaining-term label. Among matches, picks the most recently
    issued (largest issueDate) that hasn't matured yet -- that's the
    current on-the-run security by definition.
    """
    today = datetime.now(timezone.utc).date()
    if not force_refresh and _ref_cache.get(tenor) and _ref_cache_date.get(tenor) == today:
        return _ref_cache[tenor]

    target_years = TENOR_TO_SECURITY_TERM_YEARS[tenor]
    security_type = TENOR_TO_SECURITY_TYPE[tenor]

    resp = requests.get(TREASURY_DIRECT_SEARCH, params={
        "format": "json", "type": security_type, "days": 3650,
    }, timeout=20)
    resp.raise_for_status()
    securities = resp.json()

    candidates = []
    for sec in securities:
        term = str(sec.get("securityTerm", ""))
        first_token = term.split("-", 1)[0].strip()
        try:
            term_years = int(first_token)
        except ValueError:
            continue
        if term_years != target_years:
            continue
        try:
            maturity = datetime.fromisoformat(sec["maturityDate"]).date()
            issue = datetime.fromisoformat(sec["issueDate"]).date()
        except (KeyError, ValueError):
            continue
        if maturity <= today:
            continue  # already matured, not eligible to be on-the-run
        candidates.append((issue, sec, maturity))

    if not candidates:
        raise RuntimeError(f"No non-matured {security_type} with {target_years}-year original "
                            f"term found via TreasuryDirect for tenor {tenor}.")

    candidates.sort(key=lambda c: c[0])  # ascending by issue date
    _, best, maturity = candidates[-1]  # most recently issued = on-the-run

    dated_date_raw = best.get("datedDate") or best["issueDate"]
    result = {
        "cusip": best.get("cusip"),
        "coupon_pct": float(best["interestRate"]),
        "maturity_date": maturity,
        "dated_date": datetime.fromisoformat(dated_date_raw).date(),
    }
    _ref_cache[tenor] = result
    _ref_cache_date[tenor] = today
    return result


# ---------------------------------------------------------------------
# BOND MATH: clean price -> semiannual bond-equivalent yield (YTM)
# ---------------------------------------------------------------------
def _semiannual_coupon_dates(maturity_date: date, dated_date: date) -> list[date]:
    """Every semiannual coupon date from dated_date to maturity_date,
    inclusive of maturity, stepping backward 6 months at a time from
    maturity (standard convention -- coupon dates are anchored to the
    maturity date's month/day, not the original dated date's)."""
    import calendar
    dates = [maturity_date]
    cur = maturity_date
    while True:
        month = cur.month - 6
        year = cur.year
        if month <= 0:
            month += 12
            year -= 1
        # Clamp to the last valid day of the target month (handles ALL
        # short months -- Apr/Jun/Sep/Nov with 30 days, Feb with 28/29 --
        # not just February specifically.
        last_day_of_month = calendar.monthrange(year, month)[1]
        day = min(cur.day, last_day_of_month)
        cur = date(year, month, day)
        if cur <= dated_date:
            break
        dates.append(cur)
    dates.reverse()
    return dates


def bond_price_to_yield(clean_price: float, coupon_pct: float, maturity_date: date,
                         dated_date: date, settlement_date: date | None = None) -> float:
    """Converts a clean price (per 100 face value) to a semiannual
    bond-equivalent yield (%), via the standard Street/ICMA convention
    (Actual/Actual day count), solved by bisection -- same approach as
    Excel's YIELD() function. Returns yield as a percent (e.g. 4.32).
    """
    if settlement_date is None:
        settlement_date = datetime.now(timezone.utc).date()

    coupon_dates = [d for d in _semiannual_coupon_dates(maturity_date, dated_date) if d > settlement_date]
    if not coupon_dates:
        raise ValueError("No remaining coupon dates after settlement date -- security may have matured.")

    semiannual_coupon = coupon_pct / 2.0  # per 100 face, per period

    # Accrued interest: fraction of the current coupon period elapsed.
    prev_coupon = dated_date
    for d in _semiannual_coupon_dates(maturity_date, dated_date):
        if d > settlement_date:
            next_coupon = d
            break
        prev_coupon = d
    period_days = (next_coupon - prev_coupon).days
    accrued_days = (settlement_date - prev_coupon).days
    accrued_interest = semiannual_coupon * (accrued_days / period_days) if period_days > 0 else 0.0
    dirty_price = clean_price + accrued_interest

    # Standard Street/ICMA convention: w = fraction of the settlement's
    # CURRENT coupon period already elapsed... no -- w is the fraction of
    # that period REMAINING until the next coupon (i.e. how far settlement
    # sits before the next coupon, as a fraction of that period's length).
    # Every coupon after the first is then just t = w + (whole periods),
    # NOT recomputed per-period -- periods vary slightly in actual day
    # count (181 vs 184 days for Jan-Jul vs Jul-Jan), and re-deriving a
    # fresh fractional adjustment from each period's own length like a
    # naive implementation would is what caused this to compound into
    # nonsense values (t exceeding 19 "periods" for what should be a
    # 10-period, 5-year bond). This is the same convention Excel's YIELD()
    # function and Bloomberg's YAS page use.
    w = accrued_days / period_days if period_days > 0 else 0.0
    # w as computed above is the fraction of the period ALREADY elapsed;
    # the discounting exponent needs the fraction REMAINING until the
    # first coupon, which is (1 - w) plus zero whole periods for the very
    # first coupon itself.
    first_period_fraction = 1.0 - w

    def pv_at_yield(y_pct: float) -> float:
        y = y_pct / 100.0
        per_period_rate = y / 2.0
        pv = 0.0
        for i, cd in enumerate(coupon_dates, start=1):
            cf = semiannual_coupon + (100.0 if cd == coupon_dates[-1] else 0.0)
            t = first_period_fraction + (i - 1)
            pv += cf / (1.0 + per_period_rate) ** t
        return pv

    def objective(y_pct: float) -> float:
        return pv_at_yield(y_pct) - dirty_price

    return brentq(objective, -5.0, 25.0, xtol=1e-8)


# ---------------------------------------------------------------------
# PUBLIC INTERFACE -- same shape as live_treasury_yields_pyth.py
# ---------------------------------------------------------------------
def get_live_treasury_yields(max_workers: int = 5) -> dict:
    """Returns {tenor: {'yield_pct': ..., 'publish_time': ..., 'feed_id': ...,
    'bond_price': ..., 'coupon_pct': ..., 'maturity_date': ..., 'cusip': ...}}.

    'yield_pct' is the CONVERTED yield -- same key other modules in this
    project use, so this is a drop-in replacement in terms of interface.
    The extra fields are there so you can sanity-check the conversion
    itself (does the coupon/maturity/CUSIP look right for the tenor?)
    rather than just trusting the output number blindly.
    """
    def _fetch_one(tenor: str) -> tuple[str, dict]:
        feed_id = discover_price_feed_id(tenor)
        price, ts = latest_bond_price(feed_id)
        ref = lookup_on_the_run_reference(tenor)
        yield_pct = bond_price_to_yield(
            clean_price=price, coupon_pct=ref["coupon_pct"],
            maturity_date=ref["maturity_date"], dated_date=ref["dated_date"],
        )
        return tenor, {
            "yield_pct": yield_pct, "publish_time": ts, "feed_id": feed_id,
            "bond_price": price, "coupon_pct": ref["coupon_pct"],
            "maturity_date": ref["maturity_date"], "cusip": ref["cusip"],
        }

    out = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_fetch_one, tenor) for tenor in TENORS]
        for future in as_completed(futures):
            tenor, info = future.result()
            out[tenor] = info
    return out


def main():
    yields = get_live_treasury_yields()

    print("U.S. Treasury Yields — via Pyth BOND PRICE feeds, converted to yield\n")
    for tenor in TENORS:
        info = yields[tenor]
        print(f"  {tenor}: {info['yield_pct']:.3f}%   "
              f"(price={info['bond_price']:.4f}, coupon={info['coupon_pct']:.3f}%, "
              f"maturity={info['maturity_date']}, cusip={info['cusip']})")

    print("\nCROSS-CHECK RECOMMENDED: compare these values against "
          "live_treasury_yields_pyth.py's Rates.* output and/or Treasury.gov's "
          "official close. A large disagreement here points to a bad coupon/maturity "
          "match (verify the CUSIP above against TreasuryDirect's site) rather than "
          "necessarily meaning THIS feed is the correct one.")


if __name__ == "__main__":
    main()