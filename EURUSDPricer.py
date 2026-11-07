"""
EUR/USD Pyth data + GARCH + Black-Scholes fair-value pricer — adapted from
DXYPricer.py. Uses Pyth HERMES (same feed-discovery + latest-price pattern
as DXYPricer, NOT Lazer) so it's testable/discoverable the same way DXY was
— discover_feed_id() searches Hermes for a "EUR/USD" feed rather than
requiring a hardcoded feed id or a separate Lazer API key.

TICKER FORMAT CAVEAT: one example ticker is
"KXEURUSDAW-26JUL31-1.1170" — NO "T" prefix before the strike (DXY's was
"T100.9650"). I could NOT independently verify the settlement time or the
series' actual behavior online (Kalshi's site is JS-rendered, the API needs
auth to browse, and I found no public confirmation of "KXEURUSDAW"
specifically). parse_strike_and_settlement below currently assumes
same-calendar-day settlement at daily_settlement_et, based only on the
exchange UI's "5:00 PM EDT" label — DO NOT trust this for live trading
until you've called KalshiClient.get_market()/get_event() on a real
KXEURUSDAW ticker and confirmed occurrence_datetime (or equivalent) against
it, the same way DXY's ticker-date assumption turned out to be wrong once
before and had to be fixed against real API data.

Also note: KalshiClient.py's base_url is "https://api.elections.kalshi.com",
but Kalshi's own public API docs reference "https://external-api.kalshi.com"
as the base for GET /markets/{ticker}. Verify which one is actually live/
correct for the account before relying on get_market() results here.
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


def parse_strike_and_settlement(ticker: str, daily_settlement_et: str = "17:00:00") -> tuple[float, datetime]:
    """
    Parses e.g. 'KXEURUSDAW-26JUL31-1.1170' -> (1.1170, settlement in UTC).

    UNLIKE DXY: no "T" before the strike, and settlement is assumed to be
    the SAME calendar day embedded in the ticker (per the "5:00 PM EDT"
    label in the exchange UI), not day+1. This is an assumption from the
    the UI label alone — verify against get_market()/get_event() on Kalshi
    before relying on it for real settlement timing, the same way DXY's
    occurrence_datetime turned out to be the authoritative source rather
    than the ticker-parsed guess.
    """
    m = re.match(r"^([A-Z]+)-(\d{2})([A-Z]{3})(\d{2})-([\d.]+)$", ticker)
    if not m:
        raise ValueError(f"Could not parse ticker: {ticker}")
    _series, yy, mon_str, dd, strike_str = m.groups()
    month_map = {
        "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
        "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
    }
    year = 2000 + int(yy)
    month = month_map[mon_str]
    day = int(dd)
    hh, mm, ss = (int(x) for x in daily_settlement_et.split(":"))
    settlement_local = datetime(year, month, day, hh, mm, ss, tzinfo=NY)
    return float(strike_str), settlement_local.astimezone(timezone.utc)


def parse_series_and_event(ticker: str) -> tuple[str, str]:
    """'KXEURUSDAW-26JUL31-1.1170' -> ('KXEURUSDAW', 'KXEURUSDAW-26JUL31')."""
    m = re.match(r"^([A-Z]+)-(\d{2}[A-Z]{3}\d{2})-", ticker)
    if not m:
        raise ValueError(f"Could not parse series/event from ticker: {ticker}")
    series, date_part = m.groups()
    return series, f"{series}-{date_part}"


class EURUSDPricer:
    def __init__(self,
                 symbol: str = "FX.EURUSD",
                 history_symbol: str = "FX.EURUSD",
                 history_days: int = 45,
                 resolution_min: int = 5,
                 csv_path: str = "eurusd_history_cache.csv",
                 garch_scale: float = 10000.0,
                 assume_zero_drift: bool = True,
                 risk_free_rate: float = 0.0,
                 daily_settlement_et: str = "17:00:00",
                 garch_cache_path: str | None = None,
                 garch_cache_max_age_sec: float | None = None):
        """
        symbol: search query used to discover the Hermes feed id for live
            spot (searched the same way DXYPricer looked up "USDXY" — see
            discover_feed_id below). "FX.EURUSD" mirrors DXY's "FX.USDXY".
        history_symbol: symbol passed to the benchmarks TradingView-shim
            history endpoint. Kept as a separate param in case the two
            endpoints end up wanting different naming — verify both
            actually return data before trusting a live run; if either
            404s/empty-returns, try "EUR/USD" (with slash) instead.
        """
        self.symbol = symbol
        self.history_symbol = history_symbol
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
    # LOW-LEVEL PYTH HERMES (feed discovery + live price) — identical
    # pattern to DXYPricer.discover_feed_id/latest_price, just searching
    # "EUR/USD" instead of "USDXY".
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

    def discover_feed_id(self, force: bool = False) -> str:
        if getattr(self, "feed_id", None) is not None and not force:
            return self.feed_id
        for base in (HERMES, HERMES_NEW):
            try:
                data = self._get_json(f"{base}/v2/price_feeds", {"query": "EUR/USD", "asset_type": "fx"})
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
                    score = 10 * ("EUR/USD" in text or "EURUSD" in text) + 2 * ("FX" in text)
                    best.append((score, str(fid)))
            if best:
                best.sort(key=lambda t: -t[0])
                fid = best[0][1]
                self.feed_id = fid if fid.startswith("0x") else "0x" + fid
                return self.feed_id
        raise PythError("Could not discover EUR/USD feed ID via Hermes.")

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
    # HISTORY (chunked fetch + CSV cache) — identical to DXYPricer, just
    # a different symbol against the same TradingView-shim endpoint.
    # -----------------------------------------------------------------

    def _history_range(self, from_ts: int, to_ts: int, chunk_days: int = 5) -> pd.DataFrame:
        frames = []
        chunk_seconds = chunk_days * 86400
        cursor = from_ts
        while cursor < to_ts:
            chunk_end = min(cursor + chunk_seconds, to_ts)
            d = self._get_json(
                f"{BENCHMARKS}/v1/shims/tradingview/history",
                {"symbol": self.history_symbol, "resolution": str(self.resolution_min),
                 "from": cursor, "to": chunk_end},
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
    # GARCH FIT (identical logic/guards to DXYPricer — same reasoning
    # applies: reject implausible fits, clamp to a normal-vol center)
    # -----------------------------------------------------------------
    def _load_garch_cache(self) -> Garch | None:
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
                res = None

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

        # NOTE: EUR/USD's "normal" vol range is different from DXY's ~8-10%
        # (FX pairs typically run lower, roughly 5-9% annualized, but this
        # has NOT been empirically re-derived for EURUSD specifically —
        # treat these bounds as a starting placeholder and tighten them
        # once you've watched a few real fits, the same way DXY's 8-10%
        # band came from observed sessions, not a formula).
        REFERENCE_BARS = 60
        NORMAL_VOL_LOW = 0.05
        NORMAL_VOL_HIGH = 0.09
        NORMAL_VOL_CENTER = (NORMAL_VOL_LOW + NORMAL_VOL_HIGH) / 2
        MAX_REASONABLE_VOL = 0.14
        MIN_REASONABLE_VOL = 0.02

        previous_g = self.g
        self.g = candidate
        abs_vol = self.annualized_vol_forecast(REFERENCE_BARS)
        if abs_vol > 0 and not (MIN_REASONABLE_VOL <= abs_vol <= MAX_REASONABLE_VOL):
            target_vol = NORMAL_VOL_CENTER
            scale = (target_vol / abs_vol) ** 2
            print(f"  WARNING: GARCH fit produced an implausible {abs_vol*100:.3f}% annualized vol "
                  f"(outside {MIN_REASONABLE_VOL*100:.0f}%-{MAX_REASONABLE_VOL*100:.0f}%, "
                  f"placeholder-normal is {NORMAL_VOL_LOW*100:.0f}-{NORMAL_VOL_HIGH*100:.0f}%). "
                  f"Clamping by {scale:.6f}x toward ~{target_vol*100:.3f}%.")
            candidate = Garch(mu=candidate.mu, omega=candidate.omega * scale,
                               alpha=candidate.alpha, beta=candidate.beta,
                               last_eps=candidate.last_eps * math.sqrt(scale),
                               last_h=candidate.last_h * scale)
            self.g = candidate
            rechecked_vol = self.annualized_vol_forecast(REFERENCE_BARS)
            print(f"  Post-clamp check (60-bar reference): {rechecked_vol*100:.3f}%")

        REJECT_RATIO_BOUNDS = (0.4, 2.5)
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
                          f"({old_vol*100:.3f}%). Keeping the previous fit instead.")
                    self.g = previous_g
                    return self.g

        self._save_garch_cache(self.g)
        return self.g

    def refit(self, verbose: bool = True, use_cache: bool = True) -> Garch:
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
    # BLACK-SCHOLES DIGITAL PRICING — identical to DXY. This is what makes
    # the "one shared model, many strikes" ladder approach work: fit
    # sigma_annual ONCE per spot/vol state, then evaluate this closed-form
    # at each strike in the ladder — no per-strike refitting needed.
    # -----------------------------------------------------------------
    @staticmethod
    def bs_digital_call_prob(spot: float, strike: float, sigma_annual: float,
                              t_years: float, r: float = 0.0, q: float = 0.0) -> tuple[float, float, float]:
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
        if bars_remaining <= 0:
            prob = 1.0 if spot > strike else 0.0
            return {"prob": prob, "sigma_annual": 0.0, "t_years": 0.0, "d1": float("nan"), "d2": float("nan")}
        t_years = bars_remaining / self.bars_per_year
        sigma_annual = self.annualized_vol_forecast(bars_remaining)
        r = 0.0 if self.assume_zero_drift else self.risk_free_rate
        prob, d1, d2 = self.bs_digital_call_prob(spot, strike, sigma_annual, t_years, r=r, q=0.0)
        return {"prob": prob, "sigma_annual": sigma_annual, "t_years": t_years, "d1": d1, "d2": d2}

    def price_strike(self, spot: float, strike: float, settlement_utc: datetime,
                      now: datetime | None = None) -> dict:
        """Ladder-style entry point: given ONE already-fetched spot and the
        shared GARCH fit, price a SINGLE strike. Call this once per strike
        in the ladder per loop iteration — no new spot fetch, no refit."""
        if self.g is None:
            raise RuntimeError("Call refit() at least once before price_strike().")
        now = now or datetime.now(timezone.utc)
        mins_remaining = max((settlement_utc - now).total_seconds() / 60, 0)
        bars_remaining = max(0, math.ceil(mins_remaining / self.resolution_min))
        fair = self.fair_yes_prob(spot, strike, bars_remaining)
        fair["spot"] = spot
        fair["strike"] = strike
        fair["settlement_utc"] = settlement_utc
        fair["bars_remaining"] = bars_remaining
        fair["mins_remaining"] = mins_remaining
        fair["fair_yes_cents"] = round(fair["prob"] * 100)
        fair["fair_no_cents"] = 100 - fair["fair_yes_cents"]
        return fair
