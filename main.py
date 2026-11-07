import numpy as np
from scipy.stats import norm, ncx2
from scipy.optimize import brentq


def time_to_years(maturity=(0, 0, 0, 22), day_count=365.25):
    """
    maturity = (years, months, days, hours)
    """
    years, months, days, hours = maturity

    total_days = (
        years * day_count
        + months * (day_count / 12)
        + days
        + hours / 24
    )

    return total_days / day_count


def normalize_side(side):
    side = side.lower().strip()

    if side in ["above", "over", ">=", "yes_above"]:
        return "above"

    if side in ["below", "under", "<=", "yes_below"]:
        return "below"

    raise ValueError("side must be 'above' or 'below'")


def hull_white_iv(
    current_rate,
    strike,
    price_percent,
    maturity=(0, 0, 0, 22),
    side="above",
    day_count=365.25,
):
    """
    Binary Treasury-yield IV under short-horizon Hull-White normal approximation.

    Above contract:
        price = P(r_T >= K)

    Below contract:
        price = P(r_T <= K)

    Rates are decimals:
        4.21% = 0.0421
        4.35% = 0.0435
    """

    side = normalize_side(side)
    q = price_percent / 100
    T = time_to_years(maturity, day_count)

    if not 0 < q < 1:
        raise ValueError("price_percent must be between 0 and 100")

    distance = strike - current_rate

    if side == "above":
        # q = P(r_T >= K) = 1 - Phi((K-r0)/std)
        z = norm.ppf(1 - q)

    else:
        # q = P(r_T <= K) = Phi((K-r0)/std)
        z = norm.ppf(q)

    if abs(z) < 1e-12:
        raise ValueError("Price too close to 50%; IV is not identifiable from distance alone.")

    horizon_std = abs(distance / z)
    annual_iv = horizon_std / np.sqrt(T)

    model_prob_check = (
        1 - norm.cdf(distance / horizon_std)
        if side == "above"
        else norm.cdf(distance / horizon_std)
    )

    return {
        "model": "Hull-White normal approximation",
        "side": side,
        "price_probability": q,
        "current_rate_pct": current_rate * 100,
        "strike_pct": strike * 100,
        "distance_bps": abs(distance) * 10000,
        "maturity_years": T,
        "z_score": z,
        "horizon_std_decimal": horizon_std,
        "horizon_std_bps": horizon_std * 10000,
        "annual_iv_decimal": annual_iv,
        "annual_iv_bps": annual_iv * 10000,
        "model_probability_check": model_prob_check,
    }


def cir_survival_probability(r0, strike, T, kappa, theta, sigma):
    """
    CIR exact transition:

        dr = kappa(theta-r)dt + sigma sqrt(r)dW

        r_T = c * noncentral_chi_square(df, nc)

    Returns:
        P(r_T >= strike)
    """

    if sigma <= 0:
        return float(r0 >= strike)

    exp_term = np.exp(-kappa * T)

    c = sigma**2 * (1 - exp_term) / (4 * kappa)
    df = 4 * kappa * theta / sigma**2
    nc = 4 * kappa * exp_term * r0 / (sigma**2 * (1 - exp_term))

    x = strike / c

    return 1 - ncx2.cdf(x, df, nc)


def cir_iv(
    current_rate,
    strike,
    price_percent,
    maturity=(0, 0, 0, 22),
    side="above",
    kappa=1.0,
    theta=None,
    day_count=365.25,
    sigma_upper=5.0,
):
    """
    Binary Treasury-yield IV under CIR.

    Above contract:
        price = P(r_T >= K)

    Below contract:
        price = P(r_T <= K)

    Rates are decimals.
    """

    side = normalize_side(side)
    q = price_percent / 100
    T = time_to_years(maturity, day_count)

    if theta is None:
        theta = current_rate

    if not 0 < q < 1:
        raise ValueError("price_percent must be between 0 and 100")

    if current_rate <= 0 or strike <= 0 or theta <= 0:
        raise ValueError("CIR requires positive current_rate, strike, and theta.")

    def model_probability(sigma):
        p_above = cir_survival_probability(
            r0=current_rate,
            strike=strike,
            T=T,
            kappa=kappa,
            theta=theta,
            sigma=sigma,
        )

        if side == "above":
            return p_above
        else:
            return 1 - p_above

    def objective(sigma):
        return model_probability(sigma) - q

    low = 1e-4
    high = sigma_upper

    f_low = objective(low)
    f_high = objective(high)

    if f_low * f_high > 0:
        return {
            "model": "CIR",
            "side": side,
            "error": "No CIR implied volatility found in search range.",
            "price_probability": q,
            "current_rate_pct": current_rate * 100,
            "strike_pct": strike * 100,
            "distance_bps": abs(strike - current_rate) * 10000,
            "maturity_years": T,
            "low_probability": model_probability(low),
            "high_probability": model_probability(high),
        }

    sigma_cir = brentq(objective, low, high, maxiter=1000)

    local_normal_annual_vol = sigma_cir * np.sqrt(current_rate)
    model_prob_check = model_probability(sigma_cir)

    return {
        "model": "CIR exact transition",
        "side": side,
        "price_probability": q,
        "current_rate_pct": current_rate * 100,
        "strike_pct": strike * 100,
        "distance_bps": abs(strike - current_rate) * 10000,
        "maturity_years": T,
        "kappa": kappa,
        "theta_pct": theta * 100,
        "cir_sigma_decimal": sigma_cir,
        "cir_sigma_percent": sigma_cir * 100,
        "local_normal_annual_vol_decimal": local_normal_annual_vol,
        "local_normal_annual_vol_bps": local_normal_annual_vol * 10000,
        "model_probability_check": model_prob_check,
    }


def contract_iv(
    current_rate,
    strike,
    price_percent,
    maturity=(0, 0, 0, 22),
    side="above",
    kappa=1.0,
    theta=None,
):
    """
    Computes both Hull-White and CIR implied volatility.
    """

    hw = hull_white_iv(
        current_rate=current_rate,
        strike=strike,
        price_percent=price_percent,
        maturity=maturity,
        side=side,
    )

    cir = cir_iv(
        current_rate=current_rate,
        strike=strike,
        price_percent=price_percent,
        maturity=maturity,
        side=side,
        kappa=kappa,
        theta=theta,
    )

    return {
        "input": {
            "current_rate_pct": current_rate * 100,
            "strike_pct": strike * 100,
            "price_percent": price_percent,
            "maturity": maturity,
            "side": normalize_side(side),
        },
        "hull_white": hw,
        "cir": cir,
    }


def print_result(result):
    for section, values in result.items():
        print(f"\n{section.upper()}")

        if isinstance(values, dict):
            for k, v in values.items():
                print(f"{k}: {v}")
        else:
            print(values)


if __name__ == "__main__":
    # =====================================================
    # Examples
    # =====================================================

    # Example 1:
    # Current 2Y = 4.21%
    # Contract = 4.35% or below
    # Price = 98%
    # Expiry = 22 hours

    below_result = contract_iv(
        current_rate=0.0421,
        strike=0.0435,
        price_percent=98,
        maturity=(0, 0, 0, 24),
        side="below",
        kappa=1.0,
        theta=0.0421,
    )

    print_result(below_result)


    # Example 2:
    # Current 2Y = 4.21%
    # Contract = 4.10% or above
    # Price = 97%
    # Expiry = 22 hours

    above_result = contract_iv(
        current_rate=0.0421,
        strike=0.0410,
        price_percent=97,
        maturity=(0, 0, 0, 22),
        side="above",
        kappa=1.0,
        theta=0.0421,
    )

    print_result(above_result)