from CoolProp.CoolProp import PropsSI

T = 298.15      # K ~ 25 °C
P = 101325      # Pa ~ 1 atm

# General form:
# PropsSI(output, input1, value1, input2, value2, fluid)

# Density (kg/m^3)
rho = PropsSI('D', 'T', T, 'P', P, 'Water')

# Dynamic viscosity (Pa·s)
mu = PropsSI('V', 'T', T, 'P', P, 'Water')

# Specific heat capacity (J/kg·K)
cp = PropsSI('C', 'T', T, 'P', P, 'Water')

# Thermal conductivity (W/m·K)
k = PropsSI('L', 'T', T, 'P', P, 'Water')

# Enthalpy (J/kg)
h = PropsSI('H', 'T', T, 'P', P, 'Water')

values = {"rho": rho, "mu": mu, "cp": cp, "k": k, "h": h}
for name, value in values.items():
    print(name, "=", value)