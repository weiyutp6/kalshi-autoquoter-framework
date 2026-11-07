import os
"""
Live U.S. Treasury yields (2Y, 5Y, 7Y, 10Y, 30Y) via Pyth's Rates feeds.

IMPORTANT DISTINCTION -- there are TWO different Pyth feed families for
Treasuries, and it's easy to grab the wrong one (this is what was showing
in the Pyth Terminal):

  InterestRate.US2Y/USD, InterestRate.US10Y/USD, etc.
      -> BOND PRICE quotes (~99, per $100 face value). To get a yield from
         these you'd need the coupon rate of the specific on-the-run note
         and a full price->yield (YTM) solve. Pyth doesn't publish the
         coupon alongside these, so this path is a dead end without an
         external coupon source.

  Rates.US2Y, Rates.US5Y, Rates.US7Y, Rates.US10Y, Rates.US30Y
      -> YIELD directly, already expressed as a percent (e.g. 3.75).
         No coupon, no price/yield conversion, no bond math needed at all.
         Source: https://www.pyth.network/blog/collecting-the-price-of-
         everything-pyth-rates-feeds-us-treasury-yields

This script uses the second family exclusively, via the same
discover-feed-id -> fetch-latest-price pattern as DXYPricer.py (Hermes
public API, no auth needed).
"""
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

HERMES = "https://hermes.pyth.network"
HERMES_NEW = "https://pyth.dourolabs.app/hermes"

TENORS = ["US2Y", "US5Y", "US7Y", "US10Y", "US30Y"]

# Once you've confirmed a feed_id is correct (run this file directly --
# main() prints the feed_id it resolved for each tenor -- and cross-check
# against Pyth's price feed explorer), paste it here to STOP re-discovering
# it via fuzzy text search every run. Pyth feed IDs are STABLE/PERMANENT
# once assigned; they don't change day to day. Pinning eliminates the
# specific failure mode of "usually fine, today it's acting up": fuzzy
# search over Pyth's feed catalog can silently resolve to a DIFFERENT feed
# on a different day if Pyth adds a new similarly-scored feed, deprecates
# one, or search result ordering shifts -- none of which would be visible
# from the outside, since the code would just keep running against
# whatever it matched that day. A pinned ID can't drift.
# Leave a tenor's value as None to keep using live discovery for it (with
# that same drift risk).
PINNED_FEED_IDS: dict[str, str | None] = {
    "US2Y": None,
    "US5Y": None,
    "US7Y": None,
    "US10Y": None,
    "US30Y": None,
}

# Rolling sanity check: if a fresh reading jumps more than this many bps
# from the last known-good reading (within this same process's lifetime),
# flag it loudly instead of silently trusting it. This catches ANY kind of
# "acting up" -- stale publisher, a wrong feed some other bug let through,
# a genuine data glitch -- regardless of root cause. A real Fed-driven move
# can occasionally exceed this in one step; the point isn't to reject
# automatically (that risks blocking legitimate fast-moving days), it's to
# make anomalies VISIBLE so you notice before trading on bad data.
MAX_PLAUSIBLE_JUMP_BPS = 20.0
_last_good_yield_pct: dict[str, float] = {}


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


def discover_rates_feed_id(tenor: str) -> str:
    """Finds the Pyth feed id for Rates.<tenor> (e.g. 'US2Y' -> Rates.US2Y),
    same discovery pattern as DXYPricer.discover_feed_id.

    Checks PINNED_FEED_IDS first -- if a tenor has a pinned value, that's
    returned directly with NO network call and no risk of resolving to a
    different feed than intended. Only falls back to the fuzzy text search
    below for tenors that haven't been pinned yet.
    """
    pinned = PINNED_FEED_IDS.get(tenor)
    if pinned:
        return pinned

    for base in (HERMES, HERMES_NEW):
        try:
            data = _get_json(f"{base}/v2/price_feeds", {"query": tenor, "asset_type": "rates"})
        except RuntimeError:
            continue
        items = data if isinstance(data, list) else data.get("price_feeds", data.get("data", []))
        for x in items:
            attrs = x.get("attributes", {})
            symbol = str(attrs.get("symbol") or x.get("symbol") or "")
            # Match exactly "Rates.US2Y" etc, not a substring like "US20Y"
            if symbol.upper() == f"RATES.{tenor}":
                fid = x.get("id") or x.get("price_feed_id") or attrs.get("id")
                if fid:
                    return fid if fid.startswith("0x") else "0x" + fid
    raise RuntimeError(f"Could not find Rates.{tenor} feed id.")


def latest_yield(feed_id: str) -> tuple[float, int]:
    """Returns (yield_percent, publish_time_unix) for a given feed id."""
    for base in (HERMES, HERMES_NEW):
        try:
            d = _get_json(f"{base}/v2/updates/price/latest", [("ids[]", feed_id)])
            p = d["parsed"][0]["price"]
            scale = 10.0 ** int(p["expo"])
            return float(p["price"]) * scale, int(p["publish_time"])
        except Exception:  # noqa: BLE001
            continue
    raise RuntimeError(f"Could not fetch latest price for feed {feed_id}")


def get_live_treasury_yields(max_workers: int = 5) -> dict:
    """Returns {tenor: {'yield_pct': ..., 'publish_time': ...}} for all
    configured tenors.

    PARALLELIZED: each tenor needs two sequential Hermes round-trips
    (discover_rates_feed_id then latest_yield) -- doing that for all 5
    tenors one after another means up to 10 sequential HTTP calls. Since
    each tenor's fetch is fully independent, running them concurrently in
    a thread pool cuts wall-clock time roughly to that of the SLOWEST
    single tenor instead of the sum of all of them.
    """
    def _fetch_one(tenor: str) -> tuple[str, dict]:
        feed_id = discover_rates_feed_id(tenor)
        yield_pct, ts = latest_yield(feed_id)

        is_pinned = PINNED_FEED_IDS.get(tenor) is not None
        last_good = _last_good_yield_pct.get(tenor)
        jump_bps = abs(yield_pct - last_good) * 100 if last_good is not None else None
        suspicious = jump_bps is not None and jump_bps > MAX_PLAUSIBLE_JUMP_BPS
        if suspicious:
            print(f"  WARNING: {tenor} jumped {jump_bps:.1f}bps since the last fetch this "
                  f"session ({last_good:.3f}% -> {yield_pct:.3f}%) -- exceeds the "
                  f"{MAX_PLAUSIBLE_JUMP_BPS:.0f}bps sanity threshold. This COULD be a genuine "
                  f"fast market move, or it could be exactly the kind of 'Pyth acting up' "
                  f"glitch worth double-checking against another source before trading on it.")
        else:
            _last_good_yield_pct[tenor] = yield_pct

        return tenor, {"yield_pct": yield_pct, "publish_time": ts, "feed_id": feed_id,
                        "is_pinned": is_pinned, "suspicious_jump_bps": jump_bps if suspicious else None}

    out = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_fetch_one, tenor) for tenor in TENORS]
        for future in as_completed(futures):
            tenor, info = future.result()
            out[tenor] = info
    return out


def main():
    yields = get_live_treasury_yields()

    print("U.S. Treasury Yields — live via Pyth Rates feeds\n")
    any_unpinned = False
    for tenor in TENORS:
        info = yields[tenor]
        pin_str = "PINNED" if info["is_pinned"] else "discovered via fuzzy search (not pinned)"
        any_unpinned = any_unpinned or not info["is_pinned"]
        print(f"  {tenor}: {info['yield_pct']:.3f}%   "
              f"(published {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(info['publish_time']))})")
        print(f"    feed_id={info['feed_id']}  [{pin_str}]")

    spread = yields["US30Y"]["yield_pct"] - yields["US5Y"]["yield_pct"]
    print(f"\nSpread (30Y - 5Y), live: {spread:+.3f}pp  ({spread*100:+.1f} bps)")

    if any_unpinned:
        print("\nOnce you've cross-checked the feed_id(s) above against Pyth's price feed "
              "explorer and confirmed they're correct, copy them into PINNED_FEED_IDS at the "
              "top of this file to stop re-discovering them via fuzzy search every run.")


if __name__ == "__main__":
    main()