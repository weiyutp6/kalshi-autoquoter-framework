"""
DXY (USDXY) Pyth data + GARCH + Black-Scholes fair-value pricer — class
wrapper. Import DXYPricer into other scripts (e.g. an autoquoter) to fetch
spot, maintain a cached/updating history, refit GARCH periodically, and
price a KXDXYDUD-style Kalshi contract from its ticker.
"""
from __future__ import annotations

import math
import os
import re
import time
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from arch import arch_model
from scipy.special import ndtr

NY = ZoneInfo("America/New_York")
HERMES = "https://hermes.pyth.network"
HERMES_NEW = "https://pyth.dourolabs.app/hermes"
BENCHMARKS = "https://benchmarks.pyth.network"


class PythError(RuntimeError):
    pass


@dataclass
class Garch:
    mu: float
    omega: float
    alpha: float
    beta: float
    last_eps: float
    last_h: float


def parse_strike_and_settlement(ticker: str, daily_settlement_et: str = "13:59:00") -> tuple[float, datetime]:
    """
    Parses e.g. 'KXDXYDUD-26JUL13-T100.9650' -> (100.9650, settlement in UTC).

    IMPORTANT: the date embedded in the ticker (JUL13 here) is the contract's
    OPEN/listing date, NOT its settlement date. Settlement actually occurs at
    daily_settlement_et on the FOLLOWING calendar day (confirmed against
    Kalshi's own displayed rewards-window end time, e.g. a ticker dated
    Jul 13 settles at 1:58/1:59 PM ET on Jul 14). This was previously wrong
    (assumed same-day settlement) — fixed to add one day.
    """
    m = re.match(r"^[A-Z]+-(\d{2})([A-Z]{3})(\d{2})-T([\d.]+)$", ticker)
    if not m:
        raise ValueError(f"Could not parse ticker: {ticker}")
    yy, mon_str, dd, strike_str = m.groups()
    month_map = {
        "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
        "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
    }
    year = 2000 + int(yy)
    month = month_map[mon_str]
    day = int(dd)
    hh, mm, ss = (int(x) for x in daily_settlement_et.split(":"))
    open_date = datetime(year, month, day, tzinfo=NY)
    settlement_local = (open_date + timedelta(days=1)).replace(hour=hh, minute=mm, second=ss)
    return float(strike_str), settlement_local.astimezone(timezone.utc)


class DXYPricer:
    def __init__(self,
                 symbol: str = "FX.USDXY",
                 asset_type: str = "fx",
                 history_days: int = 45,
                 resolution_min: int = 5,
                 csv_path: str = "dxy_history_cache.csv",
                 garch_scale: float = 10000.0,
                 assume_zero_drift: bool = True,
                 risk_free_rate: float = 0.0,
                 daily_settlement_et: str = "13:59:00",
                 garch_cache_path: str | None = None,
                 garch_cache_max_age_sec: float | None = None):
        """
        garch_cache_path: where fitted GARCH params are saved/loaded from disk.
            Defaults to <csv_path>.garch_cache.json if not given.
        garch_cache_max_age_sec: if set and a cache file exists that's younger
            than this, refit() loads it instead of re-running the optimizer
            at all — a big speedup for iterative testing (e.g. dry runs) where
            you don't need a brand-new fit every single call. Set to None
            (default) to disable and always fit fresh. Not recommended for
            live trading beyond a few minutes of staleness — this is mainly
            a testing/dev-iteration speedup.
        """
        self.symbol = symbol
        self.asset_type = asset_type
        self.history_days = history_days
        self.resolution_min = resolution_min
        self.csv_path = csv_path
        self.garch_scale = garch_scale
        self.assume_zero_drift = assume_zero_drift
        self.risk_free_rate = risk_free_rate
        self.daily_settlement_et = daily_settlement_et
        self.garch_cache_path = garch_cache_path or (csv_path.rsplit(".", 1)[0] + "_garch_cache.json")
        self.garch_cache_max_age_sec = garch_cache_max_age_sec

        self.feed_id: str | None = None
        self.df: pd.DataFrame | None = None
        self.returns: np.ndarray | None = None
        self.g: Garch | None = None
        self.bars_per_year = 252 * 24 * 60 / self.resolution_min

    # -----------------------------------------------------------------
    # LOW-LEVEL PYTH HTTP
    # -----------------------------------------------------------------
    @staticmethod
    def _pyth_headers() -> dict:
        key = os.getenv("PYTH_API_KEY", "").strip()
        return {"Authorization": f"Bearer {key}"} if key else {}

    def _get_json(self, url: str, params=None, retries: int = 4):
        err = None
        for i in range(retries):
            try:
                r = requests.get(url, params=params, headers=self._pyth_headers(), timeout=30)
                if r.status_code == 429:
                    err = f"429 rate limited (attempt {i+1}/{retries})"
                    time.sleep(min(2 ** i, 8))
                    continue
                r.raise_for_status()
                return r.json()
            except Exception as e:  # noqa: BLE001
                err = f"{type(e).__name__}: {e}" + (
                    f" | body={e.response.text[:300]}"
                    if hasattr(e, "response") and e.response is not None else "")
                if i + 1 < retries:
                    time.sleep(min(2 ** i, 8))
        raise PythError(f"Request failed: {url}: {err}")

    # -----------------------------------------------------------------
    # FEED DISCOVERY / LIVE PRICE
    # -----------------------------------------------------------------
    def discover_feed_id(self, force: bool = False) -> str:
        if self.feed_id is not None and not force:
            return self.feed_id
        for base in (HERMES, HERMES_NEW):
            try:
                data = self._get_json(f"{base}/v2/price_feeds", {"query": "USDXY", "asset_type": self.asset_type})
            except PythError:
                continue
            items = data if isinstance(data, list) else data.get("price_feeds", data.get("data", []))
            best = []
            for x in items:
                attrs = x.get("attributes", {})
                text = " ".join(
                    str(v) for v in [x.get("symbol"), x.get("description"), attrs.get("symbol"),
                                      attrs.get("description"), attrs.get("display_symbol")] if v
                ).upper()
                fid = x.get("id") or x.get("price_feed_id") or attrs.get("id") or attrs.get("price_feed_id")
                if fid:
                    score = 10 * ("USDXY" in text) + 5 * ("DOLLAR INDEX" in text) + 2 * ("FX" in text)
                    best.append((score, str(fid)))
            if best:
                best.sort(key=lambda t: -t[0])
                fid = best[0][1]
                self.feed_id = fid if fid.startswith("0x") else "0x" + fid
                return self.feed_id
        raise PythError("Could not discover USDXY feed ID.")

    def latest_price(self) -> tuple[float, float, int]:
        """Returns (price, confidence, publish_time_unix)."""
        fid = self.discover_feed_id()
        params = [("ids[]", fid)]
        errors = []
        for base in (HERMES, HERMES_NEW):
            try:
                d = self._get_json(f"{base}/v2/updates/price/latest", params)
                p = d["parsed"][0]
                q = p["price"]
                scale = 10.0 ** int(q["expo"])
                return float(q["price"]) * scale, float(q.get("conf", 0)) * scale, int(q["publish_time"])
            except Exception as e:  # noqa: BLE001
                errors.append(str(e))
        raise PythError("Latest price failed: " + " | ".join(errors))

    # -----------------------------------------------------------------
    # HISTORY (chunked fetch + CSV cache)
    # -----------------------------------------------------------------
    def _history_range(self, from_ts: int, to_ts: int, chunk_days: int = 5) -> pd.DataFrame:
        frames = []
        chunk_seconds = chunk_days * 86400
        cursor = from_ts
        while cursor < to_ts:
            chunk_end = min(cursor + chunk_seconds, to_ts)
            d = self._get_json(
                f"{BENCHMARKS}/v1/shims/tradingview/history",
                {"symbol": self.symbol, "resolution": str(self.resolution_min), "from": cursor, "to": chunk_end},
            )
            if d.get("s") == "no_data":
                cursor = chunk_end
                continue
            if d.get("s") != "ok":
                raise PythError(f"History status: {d}")
            frames.append(pd.DataFrame({
                "time": pd.to_datetime(d["t"], unit="s", utc=True),
                "open": d["o"], "high": d["h"], "low": d["l"], "close": d["c"],
            }))
            cursor = chunk_end
        if not frames:
            return pd.DataFrame(columns=["time", "open", "high", "low", "close"])
        return pd.concat(frames, ignore_index=True)

    def _full_history(self) -> pd.DataFrame:
        now = int(time.time())
        df = self._history_range(now - self.history_days * 86400, now)
        if df.empty:
            raise PythError("No history data returned.")
        df = df.drop_duplicates("time").sort_values("time")
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        return df[df.close > 0].reset_index(drop=True)

    def load_or_update_history(self, verbose: bool = True) -> pd.DataFrame:
        """
        Loads cached history from self.csv_path if present, fetches only
        bars newer than the last cached timestamp, appends, trims to the
        rolling window, saves, and returns the combined dataframe. Falls
        back to a full pull if no usable cache exists.
        """
        now = int(time.time())
        min_needed_start = now - self.history_days * 86400

        if os.path.exists(self.csv_path):
            cached = pd.read_csv(self.csv_path, parse_dates=["time"])
            cached["time"] = pd.to_datetime(cached["time"], utc=True)
            cached = cached.sort_values("time").reset_index(drop=True)

            if len(cached) > 0:
                last_ts = int(cached["time"].iloc[-1].timestamp())
                if verbose:
                    print(f"  Cache found: {len(cached)} bars, last={cached['time'].iloc[-1]}")

                fetch_from = last_ts + self.resolution_min * 60
                if fetch_from < now:
                    new_df = self._history_range(fetch_from, now)
                    combined = pd.concat([cached, new_df], ignore_index=True) if len(new_df) else cached
                    if verbose and len(new_df):
                        print(f"  Got {len(new_df)} new bars.")
                else:
                    combined = cached

                combined = combined.drop_duplicates("time").sort_values("time")
                combined = combined[combined["time"] >= pd.Timestamp(min_needed_start, unit="s", tz="UTC")]
                combined = combined.replace([np.inf, -np.inf], np.nan).dropna()
                combined = combined[combined.close > 0].reset_index(drop=True)

                if len(combined) >= 300:
                    combined.to_csv(self.csv_path, index=False)
                    self.df = combined
                    self.returns = np.diff(np.log(combined.close.to_numpy(float)))
                    return combined
                if verbose:
                    print("  Cache too short after trimming; falling back to full pull.")

        if verbose:
            print(f"  No usable cache — pulling full {self.history_days}-day history...")
        df = self._full_history()
        df.to_csv(self.csv_path, index=False)
        self.df = df
        self.returns = np.diff(np.log(df.close.to_numpy(float)))
        return df

    # -----------------------------------------------------------------
    # GARCH FIT (with optional disk cache + warm-start)
    # -----------------------------------------------------------------
    def _load_garch_cache(self) -> Garch | None:
        """Returns a cached Garch if garch_cache_max_age_sec is set, a cache
        file exists, and it's fresh enough. Otherwise returns None."""
        if self.garch_cache_max_age_sec is None:
            return None
        if not os.path.exists(self.garch_cache_path):
            return None
        try:
            with open(self.garch_cache_path, "r") as f:
                data = json.load(f)
            age = time.time() - data.get("_cached_at", 0)
            if age > self.garch_cache_max_age_sec:
                return None
            print(f"  Using cached GARCH fit ({age:.0f}s old, max age "
                  f"{self.garch_cache_max_age_sec:.0f}s) — skipping optimizer.")
            return Garch(mu=data["mu"], omega=data["omega"], alpha=data["alpha"],
                         beta=data["beta"], last_eps=data["last_eps"], last_h=data["last_h"])
        except (json.JSONDecodeError, KeyError, OSError):
            return None

    def _save_garch_cache(self, g: Garch) -> None:
        try:
            data = asdict(g)
            data["_cached_at"] = time.time()
            with open(self.garch_cache_path, "w") as f:
                json.dump(data, f)
        except OSError as e:
            print(f"  WARNING: could not write GARCH cache: {e}")

    def fit_garch(self, use_cache: bool = True) -> Garch:
        if use_cache:
            cached = self._load_garch_cache()
            if cached is not None:
                self.g = cached
                return self.g

        if self.returns is None:
            self.load_or_update_history()
        rp = self.returns * self.garch_scale

        # Warm-start from the previous in-memory fit if we have one — the
        # optimizer converges much faster starting near a known-good
        # solution than from scratch. arch expects starting_values in the
        # model's own parameter order: [mu, omega, alpha[1], beta[1], nu].
        # nu (Student's-t degrees of freedom) isn't something we track
        # ourselves, so we default it to 8 and let the optimizer adjust it.
        model = arch_model(rp, mean="Constant", vol="GARCH", p=1, q=1, dist="StudentsT", rescale=False)
        res = None
        if self.g is not None:
            try:
                starting = np.array([
                    self.g.mu * self.garch_scale,
                    self.g.omega * (self.garch_scale ** 2),
                    self.g.alpha,
                    self.g.beta,
                    8.0,
                ])
                res = model.fit(disp="off", show_warning=False, starting_values=starting)
            except Exception:  # noqa: BLE001
                res = None  # fall back to a cold-start fit below

        if res is None:
            res = model.fit(disp="off", show_warning=False)

        p = res.params
        eps = np.asarray(res.resid) / self.garch_scale
        vol = np.asarray(res.conditional_volatility) / self.garch_scale
        m = np.isfinite(eps) & np.isfinite(vol) & (vol > 0)
        alpha, beta = float(p["alpha[1]"]), float(p["beta[1]"])
        if alpha + beta >= 0.9999:
            beta = max(0.0, 0.9998 - alpha)
        candidate = Garch(
            mu=float(p["mu"]) / self.garch_scale,
            omega=float(p["omega"]) / (self.garch_scale ** 2),
            alpha=alpha, beta=beta,
            last_eps=float(eps[m][-1]), last_h=float(vol[m][-1] ** 2),
        )

        # --- Sanity guard, part 1: ABSOLUTE bounds, centered on normal vol ---
        # Applies even on the very FIRST fit of a session, when there's no
        # previous fit to compare against (the relative guard below only
        # protects refits). With HISTORY_DAYS of dense 1-min data behind every
        # fit, the TRUE underlying annualized vol shouldn't swing wildly
        # from one fit to the next — normal DXY readings have consistently
        # landed in the ~8-10% range. A fit that "explodes" way outside a
        # reasonable band around that (e.g. 89%) is essentially always a
        # bad optimizer convergence (see fit_garch's docstring discussion),
        # not a real regime change, so it gets clamped back toward the
        # known-normal center rather than just the nearest edge of some
        # arbitrarily wide range.
        REFERENCE_BARS = 60
        NORMAL_VOL_LOW = 0.08     # bottom of the stated "normal" range
        NORMAL_VOL_HIGH = 0.10    # top of the stated "normal" range
        NORMAL_VOL_CENTER = (NORMAL_VOL_LOW + NORMAL_VOL_HIGH) / 2  # 9%, clamp target
        MAX_REASONABLE_VOL = 0.15  # anything above this is treated as an explosion
        MIN_REASONABLE_VOL = 0.03  # floor, guards against a degenerate near-zero fit too

        previous_g = self.g  # save before we start overwriting self.g for measurement
        self.g = candidate
        abs_vol = self.annualized_vol_forecast(REFERENCE_BARS)
        if abs_vol > 0 and not (MIN_REASONABLE_VOL <= abs_vol <= MAX_REASONABLE_VOL):
            target_vol = NORMAL_VOL_CENTER  # clamp to the known-normal baseline, not the nearest edge
            scale = (target_vol / abs_vol) ** 2
            print(f"  WARNING: GARCH fit produced an implausible {abs_vol*100:.3f}% annualized vol "
                  f"(outside reasonable range {MIN_REASONABLE_VOL*100:.0f}%-{MAX_REASONABLE_VOL*100:.0f}%, "
                  f"normal is {NORMAL_VOL_LOW*100:.0f}-{NORMAL_VOL_HIGH*100:.0f}%). "
                  f"Clamping omega/last_h/last_eps by {scale:.6f}x to bring it back to ~{target_vol*100:.3f}%.")
            # IMPORTANT: last_eps must also be scaled (by sqrt(scale), since it's
            # squared in h_0 = omega + alpha*last_eps^2 + beta*last_h). Scaling
            # only omega/last_h left last_eps^2 unscaled and dominating the
            # near-term forecast — the clamp looked correct when measured at
            # the 60-bar reference horizon, but a SHORT horizon (few bars to
            # settlement) is dominated by h_0, which stayed contaminated.
            candidate = Garch(mu=candidate.mu, omega=candidate.omega * scale,
                               alpha=candidate.alpha, beta=candidate.beta,
                               last_eps=candidate.last_eps * math.sqrt(scale),
                               last_h=candidate.last_h * scale)
            self.g = candidate

            # Re-check after the corrected clamp — should now land close to
            # target_vol regardless of horizon. If it still doesn't, something
            # deeper is wrong and it's worth knowing rather than trusting silently.
            rechecked_vol = self.annualized_vol_forecast(REFERENCE_BARS)
            print(f"  Post-clamp check (60-bar reference): {rechecked_vol*100:.3f}%")

        # --- Sanity guard, part 2: RELATIVE bounds (vs. previous fit) ---
        # The optimizer occasionally converges to a bad local optimum (seen
        # in practice: persistence dropping from the usual ~0.9998 to ~0.94,
        # producing a fair-value swing of ~18 points with almost no change
        # in underlying data). Compare the new fit's implied vol at a fixed
        # reference horizon against the previous fit; if it's a wild outlier,
        # keep the old (already-validated) fit instead of trusting this one.
        REJECT_RATIO_BOUNDS = (0.4, 2.5)  # reject if new/old vol ratio falls outside this
        if previous_g is not None:
            self.g = previous_g
            old_vol = self.annualized_vol_forecast(REFERENCE_BARS)

            self.g = candidate
            new_vol = self.annualized_vol_forecast(REFERENCE_BARS)

            if old_vol > 0:
                ratio = new_vol / old_vol
                if not (REJECT_RATIO_BOUNDS[0] <= ratio <= REJECT_RATIO_BOUNDS[1]):
                    print(f"  WARNING: rejecting unstable GARCH fit — new sigma_annual "
                          f"({new_vol*100:.3f}%) is {ratio:.2f}x the previous fit's "
                          f"({old_vol*100:.3f}%), outside the sane range "
                          f"{REJECT_RATIO_BOUNDS}. Keeping the previous fit instead.")
                    self.g = previous_g
                    return self.g

        self._save_garch_cache(self.g)
        return self.g

    def refit(self, verbose: bool = True, use_cache: bool = True) -> Garch:
        """Convenience: update history from Pyth, then refit GARCH (or load
        from disk cache if use_cache=True and a fresh-enough one exists)."""
        self.load_or_update_history(verbose=verbose)
        return self.fit_garch(use_cache=use_cache)

    def _garch_expected_variances(self, n: int) -> np.ndarray:
        g = self.g
        out = np.empty(n)
        h = g.omega + g.alpha * g.last_eps ** 2 + g.beta * g.last_h
        for i in range(n):
            out[i] = max(h, 1e-16)
            h = g.omega + (g.alpha + g.beta) * h
        return out

    def annualized_vol_forecast(self, bars: int) -> float:
        total_variance = float(self._garch_expected_variances(bars).sum())
        years = bars / self.bars_per_year
        if years <= 0:
            return 0.0
        return math.sqrt(total_variance / years)

    def long_run_unconditional_vol(self) -> float:
        g = self.g
        persistence = g.alpha + g.beta
        if persistence >= 0.999:
            return float("nan")
        return math.sqrt((g.omega / (1 - persistence)) * self.bars_per_year)

    def trailing_realized_vol(self, window_bars: int) -> float:
        r = self.returns
        if len(r) < window_bars:
            window_bars = len(r)
        return float(r[-window_bars:].std() * math.sqrt(self.bars_per_year))

    # -----------------------------------------------------------------
    # BLACK-SCHOLES DIGITAL PRICING
    # -----------------------------------------------------------------
    @staticmethod
    def bs_digital_call_prob(spot: float, strike: float, sigma_annual: float,
                              t_years: float, r: float = 0.0, q: float = 0.0) -> tuple[float, float, float]:
        """Standard closed-form cash-or-nothing digital call: N(d2)."""
        if t_years <= 0:
            return (1.0 if spot > strike else 0.0), float("nan"), float("nan")
        if sigma_annual <= 0:
            forward = spot * math.exp((r - q) * t_years)
            return (1.0 if forward > strike else 0.0), float("nan"), float("nan")
        sqrt_t = math.sqrt(t_years)
        d1 = (math.log(spot / strike) + (r - q + 0.5 * sigma_annual ** 2) * t_years) / (sigma_annual * sqrt_t)
        d2 = d1 - sigma_annual * sqrt_t
        return float(ndtr(d2)), d1, d2

    def fair_yes_prob(self, spot: float, strike: float, bars_remaining: int) -> dict:
        """Returns dict: prob, sigma_annual, t_years, d1, d2."""
        if bars_remaining <= 0:
            prob = 1.0 if spot > strike else 0.0
            return {"prob": prob, "sigma_annual": 0.0, "t_years": 0.0, "d1": float("nan"), "d2": float("nan")}

        t_years = bars_remaining / self.bars_per_year
        sigma_annual = self.annualized_vol_forecast(bars_remaining)
        r = 0.0 if self.assume_zero_drift else self.risk_free_rate
        prob, d1, d2 = self.bs_digital_call_prob(spot, strike, sigma_annual, t_years, r=r, q=0.0)
        return {"prob": prob, "sigma_annual": sigma_annual, "t_years": t_years, "d1": d1, "d2": d2}

    def implied_vol_from_market_price(self, spot: float, strike: float, t_years: float,
                                       market_prob: float, vol_lo: float = 1e-5, vol_hi: float = 5.0,
                                       tol: float = 1e-9, max_iter: int = 200) -> float:
        r = 0.0 if self.assume_zero_drift else self.risk_free_rate

        def f(sigma):
            p, _, _ = self.bs_digital_call_prob(spot, strike, sigma, t_years, r, 0.0)
            return p

        lo, hi = vol_lo, vol_hi
        p_lo, p_hi = f(lo), f(hi)
        if not (min(p_lo, p_hi) - 1e-9 <= market_prob <= max(p_lo, p_hi) + 1e-9):
            raise ValueError(f"market_prob={market_prob} unreachable in vol range "
                              f"[{vol_lo}, {vol_hi}] (bracket gives [{p_lo:.4f}, {p_hi:.4f}])")
        increasing = f(hi) > f(lo)
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            p_mid = f(mid)
            if abs(p_mid - market_prob) < tol:
                return mid
            if (p_mid < market_prob) == increasing:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    # -----------------------------------------------------------------
    # HIGH-LEVEL: PRICE A CONTRACT FROM ITS TICKER
    # -----------------------------------------------------------------
    def price_contract(self, ticker: str, strike_override: float | None = None,
                        settlement_utc_override: datetime | None = None) -> dict:
        """
        One-call convenience for the hot loop: prices the contract given its
        strike and settlement time, pulls the latest spot, computes
        bars_remaining, and returns the fair-value dict. Does NOT refit
        GARCH — call refit() separately/periodically, since it's the
        expensive part.

        strike_override / settlement_utc_override: pass these (fetched
        directly from Kalshi's GET /markets/{ticker} via KalshiClient.get_market,
        i.e. floor_strike/cap_strike and close_time) to use the AUTHORITATIVE
        values instead of guessing from the ticker string. Ticker-string
        parsing (parse_strike_and_settlement) has already been wrong once —
        prefer passing real values from Kalshi whenever they're available.
        """
        if self.g is None:
            raise RuntimeError("Call refit() at least once before price_contract().")

        if strike_override is not None and settlement_utc_override is not None:
            strike, settlement_utc = strike_override, settlement_utc_override
        else:
            strike, settlement_utc = parse_strike_and_settlement(ticker, self.daily_settlement_et)

        spot, conf, pub_ts = self.latest_price()

        now = datetime.now(timezone.utc)
        mins_remaining = max((settlement_utc - now).total_seconds() / 60, 0)
        bars_remaining = max(0, math.ceil(mins_remaining / self.resolution_min))

        fair = self.fair_yes_prob(spot, strike, bars_remaining)
        fair["spot"] = spot
        fair["conf"] = conf
        fair["strike"] = strike
        fair["settlement_utc"] = settlement_utc
        fair["bars_remaining"] = bars_remaining
        fair["mins_remaining"] = mins_remaining
        fair["fair_yes_cents"] = round(fair["prob"] * 100)
        fair["fair_no_cents"] = 100 - fair["fair_yes_cents"]
        return fair


def _get_kalshi_market_yes_price(ticker: str) -> float | None:
    """Public orderbook, no authentication needed — used only for the
    standalone diagnostic main() below, kept independent of KalshiClient
    so this file doesn't require a private key just to read public data."""
    try:
        resp = requests.get(
            f"https://api.elections.kalshi.com/trade-api/v2/markets/{ticker}/orderbook",
            params={"depth": 5}, timeout=15,
        )
        resp.raise_for_status()
        book = resp.json().get("orderbook_fp", {})
    except Exception as e:  # noqa: BLE001
        print(f"  Could not fetch Kalshi orderbook: {e}")
        return None
    yes_levels = book.get("yes_dollars", [])
    no_levels = book.get("no_dollars", [])
    if not yes_levels or not no_levels:
        return None
    best_yes_bid = max(float(p) for p, _ in yes_levels)
    best_no_bid = max(float(p) for p, _ in no_levels)
    best_yes_ask = 1.0 - best_no_bid
    return (best_yes_bid + best_yes_ask) / 2.0


# ===========================================================================
# STANDALONE MAIN — same full diagnostic dump as before (spot, history,
# GARCH fit, BS pricing, vol cross-checks, live market comparison), now
# built on the DXYPricer class. Read-only, no Kalshi trading. Run this file
# directly to sanity-check the model before wiring it into an autoquoter.
# ===========================================================================
if __name__ == "__main__":
    TICKER = "KXDXYDUD-26JUL13-T100.9650"   # contract to price

    pricer = DXYPricer(
        symbol="FX.USDXY",
        asset_type="fx",
        history_days=45,
        resolution_min=1,
        csv_path="dxy_history_cache.csv",
        garch_scale=10000.0,
        assume_zero_drift=True,
        risk_free_rate=0.0,
        daily_settlement_et="13:59:00",
    )

    strike, settlement_utc = parse_strike_and_settlement(TICKER, pricer.daily_settlement_et)
    print(f"Contract: {TICKER}")
    print(f"Strike: {strike}")
    print(f"Settlement (UTC): {settlement_utc.isoformat()}")

    print("\nDiscovering Pyth feed id...")
    fid = pricer.discover_feed_id()
    print(f"  Using feed_id: {fid}")

    print("\nFetching latest spot price...")
    spot, conf, pub_ts = pricer.latest_price()
    print(f"  spot={spot:.6f}  conf=±{conf:.6f}  "
          f"publish_time={datetime.fromtimestamp(pub_ts, timezone.utc).isoformat()}")

    print(f"\nLoading {pricer.history_days} days of {pricer.resolution_min}-min history "
          f"(cached: {pricer.csv_path})...")
    df = pricer.load_or_update_history()
    print(f"  bars: {len(df)}  |  first={df.time.iloc[0]}  |  last={df.time.iloc[-1]}")
    print(df.head(3))
    print("  ...")
    print(df.tail(3))

    returns = pricer.returns
    print(f"\nreturns: {len(returns)}  |  mean={returns.mean():.8f}  |  std={returns.std():.8f}  |  "
          f"min={returns.min():.8f}  |  max={returns.max():.8f}")

    print("\nFitting GARCH(1,1)...")
    g = pricer.fit_garch()
    print(f"  mu={g.mu:.8f}  omega={g.omega:.10f}  alpha={g.alpha:.6f}  "
          f"beta={g.beta:.6f}  last_h={g.last_h:.10f}  persistence(alpha+beta)={g.alpha+g.beta:.6f}")

    now = datetime.now(timezone.utc)
    mins_remaining = max((settlement_utc - now).total_seconds() / 60, 0)
    bars_remaining = max(0, math.ceil(mins_remaining / pricer.resolution_min))
    print(f"\nTime to settlement: {mins_remaining:.1f} min  ->  bars_remaining={bars_remaining}")

    fair = pricer.fair_yes_prob(spot, strike, bars_remaining)
    fair_yes_cents = round(fair["prob"] * 100)
    print("\n--- Black-Scholes digital pricing ---")
    print(f"  t_years={fair['t_years']:.8f}  sigma_annual={fair['sigma_annual']:.6f} "
          f"({fair['sigma_annual']*100:.3f}% ann. vol)")
    print(f"  d1={fair['d1']:.6f}  d2={fair['d2']:.6f}")
    print(f"\n>>> FAIR YES PROBABILITY: {fair['prob']:.4f}  ({fair_yes_cents}c)")
    print(f">>> FAIR NO PROBABILITY:  {1-fair['prob']:.4f}  ({100-fair_yes_cents}c)")

    print("\n--- Volatility cross-checks (all annualized) ---")
    print(f"  GARCH forecast (used above)     : {fair['sigma_annual']*100:7.3f}%")
    long_run = pricer.long_run_unconditional_vol()
    print(f"  GARCH long-run unconditional    : {long_run*100:7.3f}%"
          if not math.isnan(long_run) else "  GARCH long-run unconditional    :  n/a (persistence ~1, near unit root)")
    for window_min, label in [(60, "1hr"), (360, "6hr"), (1440, "24hr")]:
        window_bars = max(1, window_min // pricer.resolution_min)
        if window_bars <= len(returns):
            rv = pricer.trailing_realized_vol(window_bars)
            print(f"  Trailing realized vol ({label:>4})       : {rv*100:7.3f}%")

    print("\n--- Comparing against live Kalshi market (public orderbook, no trading) ---")
    market_mid = _get_kalshi_market_yes_price(TICKER)
    if market_mid is not None:
        print(f"  Kalshi mid price: {market_mid*100:.2f}c YES")
        try:
            implied_vol = pricer.implied_vol_from_market_price(spot, strike, fair["t_years"], market_mid)
            print(f"  Market-implied annualized vol: {implied_vol*100:.3f}%")
            print(f"  (GARCH forecast is {fair['sigma_annual']/implied_vol:.2f}x the market-implied vol)")
        except ValueError as e:
            print(f"  Could not back out implied vol: {e}")
    else:
        print("  Market orderbook unavailable or too thin.")
