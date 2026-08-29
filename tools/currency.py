"""Dated currency conversion against the fx_rates table (see docs/data_model_spec.md).

fx_rates has monthly granularity, keyed by (from_currency, to_currency, rate_month). A
conversion is anchored to a specific date and must return the exact rate and month used --
never a silently-interpolated or nearest-available rate, since that would break provenance.
"""

from __future__ import annotations

import pandas as pd


def convert_currency(
    amount: float, from_currency: str, to_currency: str, as_of_date, fx_rates: pd.DataFrame
) -> dict:
    """Convert `amount` from `from_currency` to `to_currency` using the fx_rates row for the
    calendar month containing `as_of_date`. Raises if that exact month isn't in fx_rates --
    it should never guess.
    """
    inputs = {
        "amount": amount,
        "from_currency": from_currency,
        "to_currency": to_currency,
        "as_of_date": as_of_date,
    }

    if from_currency == to_currency:
        return {
            "inputs": inputs,
            "converted_amount": amount,
            "rate": 1.0,
            "rate_month": None,
            "source_rows": [],
        }

    rate_month = pd.Timestamp(as_of_date).replace(day=1)
    match = fx_rates[
        (fx_rates.from_currency == from_currency)
        & (fx_rates.to_currency == to_currency)
        & (fx_rates.rate_month == rate_month)
    ]
    if match.empty:
        raise ValueError(
            f"No FX rate for {from_currency}->{to_currency} in {rate_month.date()}. "
            "Refusing to fall back to a different month or an inverted/derived rate."
        )

    rate = float(match.iloc[0].rate)
    return {
        "inputs": inputs,
        "converted_amount": amount * rate,
        "rate": rate,
        "rate_month": rate_month,
        "source_rows": match.to_dict("records"),
    }
