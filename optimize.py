import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations

# -----------------------------
# 1) Given data from Appendix 1
# -----------------------------
months = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
]

stages = ["I", "II", "III", "IV", "V"]

top_elevation_m = np.array([2090, 1860, 1680, 1470, 1260], dtype=float)
bottom_elevation_m = np.array([1860, 1680, 1470, 1260, 1190], dtype=float)

# Monthly average flow for each stream stage [m^3/s]
flow_data = np.array([
    [0.08, 0.15, 0.23, 0.29, 0.50],  # Jan
    [0.08, 0.16, 0.24, 0.33, 0.55],  # Feb
    [0.14, 0.27, 0.45, 0.54, 0.90],  # Mar
    [0.44, 0.77, 1.10, 1.32, 2.20],  # Apr
    [0.96, 1.22, 1.72, 1.98, 3.30],  # May
    [1.26, 1.44, 1.84, 2.16, 3.60],  # Jun
    [0.48, 0.72, 1.32, 1.44, 2.40],  # Jul
    [0.27, 0.54, 0.90, 1.08, 1.80],  # Aug
    [0.18, 0.36, 0.60, 0.72, 1.20],  # Sep
    [0.14, 0.27, 0.45, 0.54, 0.90],  # Oct
    [0.11, 0.23, 0.38, 0.45, 0.75],  # Nov
    [0.09, 0.14, 0.21, 0.34, 0.55],  # Dec
], dtype=float)

# -----------------------------
# 2) Basic assumptions
# -----------------------------
rho = 1000.0         # kg/m^3, simple water density assumption
g = 9.81             # m/s^2
diversion_fraction = 0.10   # max 10% diversion
drought_factor = 0.60       # 40% below average flow
target_kW = 100.0

# -----------------------------
# 3) Build clean tables
# -----------------------------
flow_df = pd.DataFrame(flow_data, index=months, columns=stages)

# Head drop for each stage [m]
head_m = top_elevation_m - bottom_elevation_m
head_s = pd.Series(head_m, index=stages, name="Head drop (m)")

# Max diverted flow [m^3/s]
q_diverted_df = diversion_fraction * flow_df

# Gross hydraulic power [kW]
# P = rho * g * Q * delta_z
power_kW_df = q_diverted_df * (rho * g * head_s / 1000.0)

# Drought-year power [kW]
drought_power_kW_df = drought_factor * power_kW_df

# -----------------------------
# 4) Summary tables
# -----------------------------
stage_summary = pd.DataFrame({
    "Top elevation (m)": top_elevation_m,
    "Bottom elevation (m)": bottom_elevation_m,
    "Head drop (m)": head_m
}, index=stages)

average_power_kW = power_kW_df.mean(axis=0)
minimum_power_kW = power_kW_df.min(axis=0)
average_drought_power_kW = drought_power_kW_df.mean(axis=0)
minimum_drought_power_kW = drought_power_kW_df.min(axis=0)

power_summary = pd.DataFrame({
    "Avg power, normal year (kW)": average_power_kW,
    "Min power, normal year (kW)": minimum_power_kW,
    "Avg power, drought year (kW)": average_drought_power_kW,
    "Min power, drought year (kW)": minimum_drought_power_kW
})

print("\nSTAGE GEOMETRY")
print(stage_summary.round(2))

print("\nMONTHLY POWER BY STAGE - NORMAL YEAR [kW]")
print(power_kW_df.round(2))

print("\nMONTHLY POWER BY STAGE - DROUGHT YEAR [kW]")
print(drought_power_kW_df.round(2))

print("\nPOWER SUMMARY BY STAGE [kW]")
print(power_summary.round(2))

# -----------------------------
# 5) Smallest stage combination to hit 100 kW
#    on a typical-year average basis
# -----------------------------
best_typical_combo = None
best_typical_power = None

for r in range(1, len(stages) + 1):
    valid = []
    for combo in combinations(stages, r):
        total_avg_power = average_power_kW[list(combo)].sum()
        if total_avg_power >= target_kW:
            valid.append((combo, total_avg_power))

    if valid:
        best_typical_combo, best_typical_power = min(valid, key=lambda x: x[1])
        break

print("\nSMALLEST COMBINATION THAT HITS 100 kW ON A TYPICAL-YEAR AVERAGE BASIS")
print("Stages selected:", best_typical_combo)
print("Average power [kW]:", round(best_typical_power, 2))

# -----------------------------
# 6) Check all combinations under drought year
#    and see if they hit 100 kW in EVERY month
# -----------------------------
combo_results = []

for r in range(1, len(stages) + 1):
    for combo in combinations(stages, r):
        monthly_total = drought_power_kW_df[list(combo)].sum(axis=1)
        min_month_power = monthly_total.min()
        meets_every_month = (monthly_total >= target_kW).all()

        combo_results.append({
            "Stages": combo,
            "Number of turbines": r,
            "Worst month power (kW)": min_month_power,
            "Meets 100 kW every month in drought": meets_every_month
        })

combo_df = pd.DataFrame(combo_results)
combo_df = combo_df.sort_values(
    by=["Number of turbines", "Worst month power (kW)"],
    ascending=[True, False]
)

print("\nDROUGHT-YEAR COMBINATION CHECK")
print(combo_df.to_string(index=False))

valid_drought_df = combo_df[combo_df["Meets 100 kW every month in drought"] == True]

if len(valid_drought_df) > 0:
    best_drought = valid_drought_df.iloc[0]
    print("\nSMALLEST COMBINATION THAT HITS 100 kW IN EVERY MONTH OF DROUGHT YEAR")
    print(best_drought.to_string())
else:
    print("\nNo stage combination meets 100 kW in every month of the drought year.")

# -----------------------------
# 7) Simple plots
# -----------------------------
plt.figure(figsize=(10, 5))
for stage in stages:
    plt.plot(months, power_kW_df[stage], marker="o", label=f"Stage {stage}")
plt.axhline(target_kW, linestyle="--", label="100 kW target")
plt.title("Monthly gross hydraulic power by stage - normal year")
plt.xlabel("Month")
plt.ylabel("Power [kW]")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
if best_typical_combo is not None:
    normal_combo_power = power_kW_df[list(best_typical_combo)].sum(axis=1)
    drought_combo_power = drought_power_kW_df[list(best_typical_combo)].sum(axis=1)

    plt.plot(months, normal_combo_power, marker="o", label="Selected combo - normal year")
    plt.plot(months, drought_combo_power, marker="o", label="Selected combo - drought year")
    plt.axhline(target_kW, linestyle="--", label="100 kW target")
    plt.title(f"Power from selected stage combination {best_typical_combo}")
    plt.xlabel("Month")
    plt.ylabel("Power [kW]")
    plt.legend()
    plt.tight_layout()
    plt.show()

# -----------------------------
# 8) What this model does NOT include
# -----------------------------
# This is only a first-pass screening model.
# A real design must later include:
# - pipe friction losses
# - minor losses
# - turbine/generator efficiency
# - pressure checks against pipe ratings
# - actual intake-to-turbine routing choices