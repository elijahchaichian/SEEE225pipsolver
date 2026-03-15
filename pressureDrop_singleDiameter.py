from __future__ import annotations
import math
import scipy

G = 9.81  # m/s^2

# ---------------------------------------------------------------------
# Simple user input helper
# ---------------------------------------------------------------------

def ask_float(prompt: str) -> float:
    while True:
        raw = input(prompt).strip()
        try:
            return float(raw)
        except ValueError:
            print("Invalid number. Try again.")

def ask_choice(prompt: str, valid_choices: tuple[str, ...]) -> str:
    while True:
        raw = input(prompt).strip().lower()
        if raw in valid_choices:
            return raw
        print(f"Valid choices are: {', '.join(valid_choices)}")

# ---------------------------------------------------------------------
# Basic pipe and fluid calculations
# ---------------------------------------------------------------------

def pipe_area(diameter_m: float) -> float:
    return math.pi * diameter_m**2 / 4.0

def reynolds_number(rho: float, mu: float, velocity_m_s: float, diameter_m: float) -> float:
    return rho * abs(velocity_m_s) * diameter_m / mu

def alpha_from_reynolds(re: float) -> float:
    """
    Requested rule:
    Re > 2300  -> alpha = 1
    Re <= 2300 -> alpha = 2
    """
    if re > 2300.0:
        return 1.0
    return 2.0

def friction_factor(re: float, roughness_m: float, diameter_m: float,
                    tolerance: float = 0.00001, max_iterations: int = 200) -> tuple[float, int]:
    """
    Darcy friction factor. Laminar flow: = 64 / Re

    Turbulent flow:
        solve the Colebrook equation iteratively
        1/sqrt(f) = -2.0 log10( (epsilon/D)/3.7 + 2.51/(Re*sqrt(f)) )

    Relative error target:
        0.1% = 0.001
    """
    if re <= 0.0:
        raise ValueError("Reynolds number must be positive.")

    if re <= 2300.0:
        return 64.0 / re, 1

    relative_roughness = roughness_m / diameter_m
    f_old = 0.02  # initial guess for turbulent flow

    for iteration in range(1, max_iterations + 1):
        rhs = -2.0 * math.log10(
            (relative_roughness / 3.7) + (2.51 / (re * math.sqrt(f_old)))
        )
        f_new = 1.0 / (rhs**2)

        relative_error = abs((f_new - f_old) / f_new)

        if relative_error <= tolerance:
            return f_new, iteration

        f_old = f_new

    raise RuntimeError("Colebrook iteration did not converge.")

# ---------------------------------------------------------------------
# Flow input using continuity
# ---------------------------------------------------------------------

def get_flow_values(rho: float, diameter_m: float) -> tuple[float, float, float]:
    """
    Continuity:
        Q = A * V
        mdot = rho * Q = rho * A * V

    The user provides one of:
        1 = velocity
        2 = volumetric flow rate
        3 = mass flow rate
    """
    area = pipe_area(diameter_m)

    print("\nFLOW INPUT")
    print("Choose the quantity you know:")
    print("1 = velocity V [m/s]")
    print("2 = volumetric flow rate Q [m^3/s]")
    print("3 = mass flow rate mdot [kg/s]")

    choice = ask_choice("Enter 1, 2, or 3: ", ("1", "2", "3"))

    if choice == "1":
        velocity = ask_float("Velocity V [m/s]: ")
        q = area * velocity
        mdot = rho * q
    elif choice == "2":
        q = ask_float("Volumetric flow rate Q [m^3/s]: ")
        velocity = q / area
        mdot = rho * q
    else:
        mdot = ask_float("Mass flow rate mdot [kg/s]: ")
        q = mdot / rho
        velocity = q / area

    return velocity, q, mdot

# ---------------------------------------------------------------------
# Pressure-loss solver
# ---------------------------------------------------------------------

def solve_pressure_loss(
    rho: float,
    mu: float,
    length_m: float,
    diameter_m: float,
    roughness_m: float,
    z_in_m: float,
    z_out_m: float,
    velocity_m_s: float
) -> dict:
    """
    General energy equation used as the starting point:
        Pin/(rho*g) + alpha_in*Vin^2/(2g) + z_in
        =
        Pout/(rho*g) + alpha_out*Vout^2/(2g) + z_out + h_major + h_minor
    For one incompressible pipe with one constant diameter:
        Vin = Vout = V
        alpha_in = alpha_out = alpha(Re)
        h_minor = 0
    Continuity is kept explicitly through:
        Q = A*V
        mdot = rho*Q = rho*A*V
    Rearranged for pressure loss:
        DeltaP = Pin - Pout
               = rho*g * [
                    (alpha_out*V^2 - alpha_in*V^2)/(2g)
                    + (z_out - z_in)
                    + h_major
                    + h_minor
                 ]
    In this simplified single-diameter case, the kinetic-energy term cancels.
    """
    area = pipe_area(diameter_m)
    q_m3_s = area * velocity_m_s
    mdot_kg_s = rho * q_m3_s

    re = reynolds_number(rho, mu, velocity_m_s, diameter_m)
    alpha = alpha_from_reynolds(re)
    f, iterations = friction_factor(re, roughness_m, diameter_m)

    velocity_head_m = velocity_m_s**2 / (2.0 * G)
    h_major_m = f * (length_m / diameter_m) * velocity_head_m
    h_minor_m = 0.0  # simplified version requested

    kinetic_term_m = ((alpha * velocity_m_s**2) - (alpha * velocity_m_s**2)) / (2.0 * G)

    delta_p_pa = rho * G * (
        kinetic_term_m
        + (z_out_m - z_in_m)
        + h_major_m
        + h_minor_m
    )

    return {
        "area_m2": area,
        "velocity_m_s": velocity_m_s,
        "volumetric_flow_m3_s": q_m3_s,
        "mass_flow_kg_s": mdot_kg_s,
        "reynolds_number": re,
        "alpha_used": alpha,
        "darcy_friction_factor": f,
        "colebrook_iterations": iterations,
        "velocity_head_m": velocity_head_m,
        "major_head_loss_m": h_major_m,
        "minor_head_loss_m": h_minor_m,
        "delta_p_pa": delta_p_pa,
        "delta_p_kpa": delta_p_pa / 1000.0,
        "delta_p_bar": delta_p_pa / 100000.0,
    }


# ---------------------------------------------------------------------
# Main program
# ---------------------------------------------------------------------

def main() -> None:
    print("Single-Pipe Pressure-Loss Solver")
    print("SI units only")

    print("\nFLUID INPUT")
    rho = ask_float("Density rho [kg/m^3]: ")
    mu = ask_float("Coefficient of Viscosity mu [Pa*s]: ")

    print("\nPIPE INPUT")
    length_m = ask_float("Pipe length L [m]: ")
    diameter_m = ask_float("Pipe diameter D [m]: ")
    roughness_m = ask_float("Absolute roughness epsilon [m]: ")
    z_in_m = ask_float("Inlet elevation z_in [m]: ")
    z_out_m = ask_float("Outlet elevation z_out [m]: ")

    velocity_m_s, q_m3_s, mdot_kg_s = get_flow_values(rho, diameter_m)
    _ = (q_m3_s, mdot_kg_s)  # kept so continuity stays visible in the workflow

    results = solve_pressure_loss(
        rho=rho,
        mu=mu,
        length_m=length_m,
        diameter_m=diameter_m,
        roughness_m=roughness_m,
        z_in_m=z_in_m,
        z_out_m=z_out_m,
        velocity_m_s=velocity_m_s
    )

    print("\nRESULTS")
    print(f"Area A                           = {results['area_m2']:.6f} m^2")
    print(f"Velocity V                       = {results['velocity_m_s']:.6f} m/s")
    print(f"Volumetric flow rate Q           = {results['volumetric_flow_m3_s']:.6f} m^3/s")
    print(f"Mass flow rate mdot              = {results['mass_flow_kg_s']:.6f} kg/s")
    print(f"Reynolds number Re               = {results['reynolds_number']:.6f}")
    print(f"Alpha used                       = {results['alpha_used']:.6f}")
    print(f"Darcy friction factor f          = {results['darcy_friction_factor']:.6f}")
    print(f"Colebrook iterations             = {results['colebrook_iterations']}")
    print(f"Velocity head V^2/(2g)           = {results['velocity_head_m']:.6f} m")
    print(f"Major head loss                  = {results['major_head_loss_m']:.6f} m")
    print(f"Minor head loss                  = {results['minor_head_loss_m']:.6f} m")
    print(f"Pressure loss DeltaP = Pin-Pout  = {results['delta_p_pa']:.6f} Pa")
    print(f"Pressure loss DeltaP             = {results['delta_p_kpa']:.6f} kPa")
    print(f"Pressure loss DeltaP             = {results['delta_p_bar']:.6f} bar")

    if results["delta_p_pa"] < 0.0:
        print("\nNote: DeltaP is negative.")
        print("This means outlet pressure is greater than inlet pressure for the values entered.")


if __name__ == "__main__":
    main()