"""Unit conversion utilities for climate data."""

from typing import Optional


def fahrenheit_to_celsius(f: float) -> float:
    """Convert Fahrenheit to Celsius."""
    return (f - 32) * 5 / 9


def kelvin_to_celsius(k: float) -> float:
    """Convert Kelvin to Celsius."""
    return k - 273.15


def convert_to_celsius(value: float, unit: str) -> Optional[float]:
    """Convert a temperature value to Celsius based on its unit string.

    Returns None if the unit is already Celsius or unrecognized.
    """
    unit_upper = unit.upper()
    if "F" in unit_upper or "FAHRENHEIT" in unit_upper:
        return fahrenheit_to_celsius(value)
    elif "KELVIN" in unit_upper or (len(unit.strip()) == 1 and unit_upper.strip() == "K"):
        return kelvin_to_celsius(value)
    return None


def format_value_with_conversion(value: float, unit: str) -> str:
    """Format a value with its unit, appending Celsius conversion if applicable."""
    base = f"{value:.2f} {unit}" if unit else f"{value:.2f}"
    celsius = convert_to_celsius(value, unit)
    if celsius is not None:
        return f"{base} ({celsius:.2f}\u00b0C)"
    return base
