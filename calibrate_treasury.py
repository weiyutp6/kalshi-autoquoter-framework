"""
Calibrates CIR (kappa, theta, sigma) to historical daily Treasury par-yield
data, so main.py's cir_iv() can be seeded with historically-grounded
parameters instead of the hardcoded kappa=1.0 / theta=current_rate guesses.

Data format expected: the standard treasury.gov daily par yield curve CSV
export, columns like "1 Mo", "2 Yr", "10 Yr", etc., one row per business day.
Pass one or more file paths — overlapping/duplicate dates across files are
deduplicated automatically, so you can feed e.g. a prior-year file and a
current-year file together.

Method: discretized CIR regression (Euler scheme), NOT market-implied.
    dr = kappa*(theta - r)*dt + sigma*sqrt(r)*dW
Dividing the discretized version by sqrt(r0*dt) linearizes it into an OLS
regression:
    (r1-r0)/sqrt(r0*dt) = [kappa*theta]*sqrt(dt)/sqrt(r0) - kappa*sqrt(dt)*sqrt(r0) + sigma*eps
which recovers kappa (slope on the second regressor), theta (= intercept
coefficient / kappa), and sigma (residual std of the regression). This is a
standard GMM/OLS-style CIR estimator — not a full MLE, but doesn't require
an optimizer and is stable/fast for this use case.

dt is computed from ACTUAL calendar-day gaps between observations (not a
fixed 1/252), so weekends/holidays in the daily series don't bias the fit.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd


def load_treasury_csvs(paths: list[str]) -> pd.DataFrame:
    """Loads and merges one or more treasury.gov daily-rate CSVs, deduped
    and sorted by date ascending."""
    frames = [pd.read_csv(p) for p in paths]
    combined = pd.concat(frames, ignore_index=True)
    combined["Date"] = pd.to_datetime(combined["Date"], format="%m/%d/%Y")
    combined = combined.drop_duplicates("Date").sort_values("Date").reset_index(drop=True)
    return combined


def calibrate_cir(df: pd.DataFrame, tenor: str = "2 Yr") -> dict:
    """
    Fits CIR (kappa, theta, sigma) to one tenor column of a treasury
    dataframe (as returned by load_treasury_csvs). Rates are read as
    percent (e.g. 4.21) and converted to decimal internally.

    Returns kappa_hat (per-year mean reversion speed), theta_hat (decimal
    long-run mean), sigma_hat (CIR vol-of-vol param), plus the sample mean/
    std and the implied "local normal annual vol" at the most recent
    observed rate (sigma_hat * sqrt(r_current)) — directly comparable to
    main.py's local_normal_annual_vol_bps output, but historical/realized
    rather than market-implied.
    """
    if tenor not in df.columns:
        raise ValueError(f"Tenor '{tenor}' not found. Available: {[c for c in df.columns if c != 'Date']}")

    r = df[tenor].to_numpy(float) / 100.0
    dates = df["Date"].to_numpy()
    dt = np.diff(dates).astype("timedelta64[D]").astype(float) / 365.25

    r0, r1 = r[:-1], r[1:]
    if np.any(r0 <= 0):
        raise ValueError("CIR calibration requires strictly positive historical rates; "
                          "found a zero/negative value in this tenor's history.")

    y = (r1 - r0) / np.sqrt(r0 * dt)
    x1 = np.sqrt(dt) / np.sqrt(r0)
    x2 = -np.sqrt(dt) * np.sqrt(r0)
    X = np.column_stack([x1, x2])

    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    kappa_theta, kappa_hat = coef
    theta_hat = kappa_theta / kappa_hat if kappa_hat != 0 else float("nan")

    resid = y - X @ coef
    sigma_hat = float(resid.std(ddof=2))

    current_rate = float(r[-1])
    local_vol = sigma_hat * np.sqrt(current_rate)

    return {
        "tenor": tenor,
        "n_obs": len(r),
        "date_start": str(df["Date"].min().date()),
        "date_end": str(df["Date"].max().date()),
        "kappa_hat": float(kappa_hat),
        "theta_hat_decimal": float(theta_hat),
        "theta_hat_pct": float(theta_hat) * 100,
        "sigma_hat": sigma_hat,
        "sample_mean_pct": float(r.mean()) * 100,
        "sample_std_pct": float(r.std()) * 100,
        "current_rate_pct": current_rate * 100,
        "historical_local_normal_annual_vol_bps": local_vol * 10000,
    }


def print_calibration(result: dict) -> None:
    for k, v in result.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    import sys

    paths = sys.argv[1:] or [
        "1784132787241_daily-treasury-rates.csv",
        "1784132787241_daily-treasury-rates2025.csv",
    ]
    df = load_treasury_csvs(paths)

    for tenor in ["1 Yr", "2 Yr", "5 Yr", "10 Yr"]:
        result = calibrate_cir(df, tenor=tenor)
        print(f"\n=== {tenor} ===")
        print_calibration(result)
