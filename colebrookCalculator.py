from __future__ import annotations
import math
from typing import Iterable

G = 9.81  # m/s^2


def friction_factor_single(
    re: float,
    roughness_m: float,
    diameter_m: float,
    tolerance: float = 1e-10,
    max_iterations: int = 10000,
) -> float:
    """
    Darcy friction factor using a hard cutoff at Re = 2300.

    Rules used:
        Re < 2300   -> laminar, f = 64 / Re
        Re >= 2300  -> Colebrook iteration

    Inputs must be SI:
        re           : Reynolds number [-]
        roughness_m  : absolute roughness [m]
        diameter_m   : pipe inner diameter [m]
    """
    if re <= 0.0:
        raise ValueError("Reynolds number must be positive.")
    if diameter_m <= 0.0:
        raise ValueError("Diameter must be positive.")
    if roughness_m < 0.0:
        raise ValueError("Roughness cannot be negative.")

    # Laminar
    if re < 2300.0:
        return 64.0 / re

    # Colebrook for Re >= 2300
    relative_roughness = roughness_m / diameter_m
    f_old = 0.02  # initial guess

    for _ in range(max_iterations):
        term = (relative_roughness / 3.7) + (2.51 / (re * math.sqrt(f_old)))

        if term <= 0.0:
            raise RuntimeError("Invalid Colebrook term encountered during iteration.")

        rhs = -2.0 * math.log10(term)
        f_new = 1.0 / (rhs ** 2)

        relative_error = abs((f_new - f_old) / f_new)

        if relative_error <= tolerance:
            return f_new

        f_old = f_new

    raise RuntimeError("Colebrook iteration did not converge.")


def friction_factor_list(
    re_list: Iterable[float],
    roughness_m_list: Iterable[float],
    diameter_m_list: Iterable[float],
    tolerance: float = 1e-10,
    max_iterations: int = 10000,
) -> list[float]:
    """
    Returns a list of Darcy friction factors corresponding to the input lists.
    All lists must have the same length.
    All dimensional inputs must be in SI units.
    """
    re_list = list(re_list)
    roughness_m_list = list(roughness_m_list)
    diameter_m_list = list(diameter_m_list)

    if not (len(re_list) == len(roughness_m_list) == len(diameter_m_list)):
        raise ValueError("All input lists must have the same length.")

    return [
        friction_factor_single(re, roughness_m, diameter_m, tolerance, max_iterations)
        for re, roughness_m, diameter_m in zip(re_list, roughness_m_list, diameter_m_list)
    ]
    
re_list = [90900]
roughness_m = [0.001]
diameter_m = [0.1]
f_list = friction_factor_list(re_list, roughness_m, diameter_m)
print(f_list)