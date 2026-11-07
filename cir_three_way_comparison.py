import os
"""
CIRTreasuryPricer: class-based version of the three pricing views (European
options, vertical-spread binary, direct-CDF binary) on CIR-calibrated
Treasury yields.

WHY A CLASS: the original script-style version did everything --
calibration, live fetch, chain-building, CSV writing -- inside one main()
function. Autoquoter needs a zero-argument price_fn() that returns just
{"fair_yes_cents", "fair_no_cents", "spot", "strike"} for ONE specific
contract, called fresh every polling cycle -- not a whole 41-strike chain
dumped to CSV every time. This class separates those concerns:

  - calibrate() / get_params()   -- cached CIR (kappa,theta,sigma) per tenor
  - price_contract()             -- prices ONE contract (spot/strike/tenor)
  - make_price_fn()              -- wraps price_contract() into the exact
                                     zero-arg callable Autoquoter expects,
                                     e.g.:
                                         pricer = CIRTreasuryPricer()
                                         price_fn = pricer.make_price_fn(
                                             tenor="US2Y", strike_pct=4.20)
                                         bot = Autoquoter(kc, ticker, price_fn)
  - build_chain() / build_all_chains() -- the full 41-strike three-way
                                     comparison table, RETURNED as data
                                     (no file I/O) -- same numbers as
                                     before, just no longer tied to a
                                     print/CSV side effect.

The class itself never writes a file or prints a chain table. All of the
CSV-writing / console-printing behavior from the original script now lives
in the __main__ block at the bottom, which uses the class the same way any
other caller would.

Composes:
  - calibrate_cir()              from calibrate_treasury.py
  - get_live_treasury_yields()   from live_treasury_yields_pyth.py
  - cir_survival_probability()   from main.py
  - cir_call_price/cir_put_price from cir_european_chain.py
"""
import io
import contextlib
import csv as csvmod

with contextlib.redirect_stdout(io.StringIO()):
    from main import cir_survival_probability, time_to_years

from calibrate_treasury import load_treasury_csvs, calibrate_cir
from live_treasury_yields_pyth import get_live_treasury_yields
from cir_european_chain import cir_call_price, cir_put_price

TREASURY_CSV_PATHS = [
    "1784132787241_daily-treasury-rates.csv",
    "1784132787241_daily-treasury-rates2025.csv",
]

TENOR_MAP = {
    "US2Y": "2 Yr", "US5Y": "5 Yr", "US7Y": "7 Yr", "US10Y": "10 Yr", "US30Y": "30 Yr",
}

DEFAULT_MATURITY_HOURS = 24  # 個人智慧財產,參數不提供 (placeholder)
DEFAULT_STRIKE_RANGE_BPS = 20  # 個人智慧財產,參數不提供 (placeholder)
DEFAULT_STRIKE_STEP_BPS = 1  # 個人智慧財產,參數不提供 (placeholder)


class CIRTreasuryPricer:
    def __init__(
        self,
        treasury_csv_paths: list[str] = TREASURY_CSV_PATHS,
        tenor_map: dict[str, str] = TENOR_MAP,
        maturity_hours: float = DEFAULT_MATURITY_HOURS,
        strike_range_bps: int = DEFAULT_STRIKE_RANGE_BPS,
        strike_step_bps: int = DEFAULT_STRIKE_STEP_BPS,
    ):
        self.treasury_csv_paths = treasury_csv_paths
        self.tenor_map = tenor_map
        self.maturity_hours = maturity_hours
        self.strike_range_bps = strike_range_bps
        self.strike_step_bps = strike_step_bps

        self._hist_df = None            # lazy-loaded, cached
        self._calibrations: dict[str, tuple[float, float, float]] = {}  # tenor -> (kappa, theta, sigma)

    # -----------------------------------------------------------------
    # CALIBRATION (cached -- historical data + CIR fit are expensive-ish,
    # no reason to redo them every polling cycle)
    # -----------------------------------------------------------------
    def calibrate(self, tenors: list[str] | None = None) -> dict[str, tuple[float, float, float]]:
        """Loads historical data (once, cached) and fits (kappa, theta,
        sigma) for the given tenors (default: every tenor in tenor_map).
        Safe to call again later to refresh calibration."""
        if self._hist_df is None:
            self._hist_df = load_treasury_csvs(self.treasury_csv_paths)

        for tenor in (tenors or list(self.tenor_map.keys())):
            csv_col = self.tenor_map[tenor]
            cal = calibrate_cir(self._hist_df, tenor=csv_col)
            self._calibrations[tenor] = (cal["kappa_hat"], cal["theta_hat_decimal"], cal["sigma_hat"])
        return self._calibrations

    def get_params(self, tenor: str) -> tuple[float, float, float]:
        """Returns (kappa, theta, sigma) for a tenor, calibrating on first
        use if not already cached."""
        if tenor not in self._calibrations:
            self.calibrate([tenor])
        return self._calibrations[tenor]

    # -----------------------------------------------------------------
    # SINGLE-CONTRACT PRICING -- the Autoquoter hook
    # -----------------------------------------------------------------
    def price_contract(self, tenor: str, strike_pct: float, r0_pct: float | None = None,
                        maturity_hours: float | None = None) -> dict:
        """Prices ONE contract. strike_pct/r0_pct are in PERCENT (e.g. 4.20
        meaning 4.20%), matching Kalshi's floor_strike convention. If
        r0_pct isn't given, fetches the live rate from Pyth for this tenor.

        Returns a dict shaped exactly for Autoquoter's price_fn:
            fair_yes_cents, fair_no_cents, spot, strike
        plus extra diagnostic fields (T_years, kappa, theta, sigma) that
        Autoquoter ignores but are useful for logging/debugging.
        """
        kappa, theta, sigma = self.get_params(tenor)
        if r0_pct is None:
            live = get_live_treasury_yields()
            r0_pct = live[tenor]["yield_pct"]

        r0 = r0_pct / 100.0
        strike = strike_pct / 100.0
        T = time_to_years((0, 0, 0, maturity_hours or self.maturity_hours))

        fair_yes = cir_survival_probability(r0, strike, T, kappa, theta, sigma) * 100
        return {
            "fair_yes_cents": fair_yes,
            "fair_no_cents": 100 - fair_yes,
            "spot": r0,
            "strike": strike,
            "T_years": T,
            "kappa": kappa,
            "theta": theta,
            "sigma": sigma,
        }

    def make_price_fn(self, tenor: str, strike_pct: float, maturity_hours: float | None = None):
        """Returns a zero-argument callable: () -> price_contract(...).
        This is what you hand to Autoquoter as its price_fn -- it re-fetches
        the live Pyth rate fresh on every call, exactly as Autoquoter's
        run_cycle() expects (one fresh fair-value read per polling cycle).

        Example:
            pricer = CIRTreasuryPricer()
            price_fn = pricer.make_price_fn(tenor="US2Y", strike_pct=4.20)
            bot = Autoquoter(kc, ticker="KXFED2Y-...-T4.20", price_fn=price_fn)
        """
        def _price_fn():
            return self.price_contract(tenor, strike_pct, maturity_hours=maturity_hours)
        return _price_fn

    # -----------------------------------------------------------------
    # FULL CHAIN (three-way comparison table) -- returned as data, no I/O
    # -----------------------------------------------------------------
    def build_chain(self, tenor: str, r0_pct: float | None = None) -> list[dict]:
        """Returns the full +/-strike_range_bps chain for one tenor as a
        list of row dicts (same fields as the original script's CSV rows).
        No file is written and nothing is printed -- purely returns data."""
        kappa, theta, sigma = self.get_params(tenor)
        if r0_pct is None:
            live = get_live_treasury_yields()
            r0_pct = live[tenor]["yield_pct"]

        r0 = r0_pct / 100.0
        T = time_to_years((0, 0, 0, self.maturity_hours))
        h = self.strike_step_bps * 1e-4

        rows = []
        for k in range(-self.strike_range_bps, self.strike_range_bps + 1, self.strike_step_bps):
            K = r0 + k * 1e-4

            call = cir_call_price(r0, K, T, kappa, theta, sigma)
            put = cir_put_price(r0, K, T, kappa, theta, sigma)

            call_lo = cir_call_price(r0, K - h / 2, T, kappa, theta, sigma)
            call_hi = cir_call_price(r0, K + h / 2, T, kappa, theta, sigma)
            vspread_yes = (call_lo - call_hi) / h * 100

            direct_yes = cir_survival_probability(r0, K, T, kappa, theta, sigma) * 100
            diff = vspread_yes - direct_yes

            rows.append({
                "tenor": tenor, "r0_pct": r0 * 100, "kappa": kappa,
                "theta_pct": theta * 100, "sigma": sigma,
                "strike_offset_bps": k, "strike_pct": K * 100,
                "call_bps": call * 1e4, "put_bps": put * 1e4,
                "vertical_spread_yes_cents": vspread_yes,
                "direct_binary_yes_cents": direct_yes,
                "diff_cents": diff,
            })
        return rows

    def build_all_chains(self) -> dict[str, list[dict]]:
        """Returns {tenor: [rows...]} for every tenor in tenor_map, fetching
        live yields once and reusing them across tenors (one Pyth round-
        trip instead of one per tenor)."""
        live = get_live_treasury_yields()
        return {
            tenor: self.build_chain(tenor, r0_pct=live[tenor]["yield_pct"])
            for tenor in self.tenor_map
        }


# ===========================================================================
# STANDALONE SCRIPT BEHAVIOR -- printing + CSV writing live here now, not
# inside the class. Run this file directly to reproduce the old script's
# console output and per-tenor CSV files.
# ===========================================================================
def _print_chain(tenor: str, rows: list[dict]) -> None:
    first = rows[0]
    print(f"=== {tenor} ===  r0(live)={first['r0_pct']:.3f}%  "
          f"kappa={first['kappa']:.3f}  theta={first['theta_pct']:.3f}%  sigma={first['sigma']:.4f}\n")
    print(f"{'bps':>5} {'strike%':>9} {'call_bps':>9} {'put_bps':>9} "
          f"{'vspread_c':>10} {'direct_c':>9} {'diff':>7}")
    for row in rows:
        print(f"{row['strike_offset_bps']:5d} {row['strike_pct']:9.4f} "
              f"{row['call_bps']:9.4f} {row['put_bps']:9.4f} "
              f"{row['vertical_spread_yes_cents']:10.4f} {row['direct_binary_yes_cents']:9.4f} "
              f"{row['diff_cents']:7.4f}")
    print()


def main():
    OUTPUT_CSV_TEMPLATE = "cir_three_way_comparison_{tenor}.csv"

    print("Calibrating CIR parameters from historical Treasury data...")
    pricer = CIRTreasuryPricer()
    pricer.calibrate()

    print("Fetching live rates from Pyth (Rates.* feeds) and building chains...")
    all_chains = pricer.build_all_chains()

    written_paths = []
    for tenor, rows in all_chains.items():
        _print_chain(tenor, rows)

        out_path = OUTPUT_CSV_TEMPLATE.format(tenor=tenor)
        with open(out_path, "w", newline="") as f:
            writer = csvmod.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        written_paths.append(out_path)
        print(f"Wrote {len(rows)} rows to {out_path}\n")

    print(f"Done — {len(written_paths)} files written: {written_paths}")


if __name__ == "__main__":
    main()
