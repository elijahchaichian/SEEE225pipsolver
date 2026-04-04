import math
import numpy as np

# -----------------------------
# constants
# -----------------------------
rho = 1000.0
g = 9.81
nu = 1e-6
months = ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]

# -----------------------------
# project data
# -----------------------------
stages = {
    "I":   {"H": 230.0, "L": 1300.0, "Q": [0.08,0.08,0.14,0.44,0.96,1.26,0.48,0.27,0.18,0.14,0.11,0.09]},
    "II":  {"H": 180.0, "L":  300.0, "Q": [0.15,0.16,0.27,0.77,1.22,1.44,0.72,0.54,0.36,0.27,0.23,0.14]},
    "III": {"H": 210.0, "L":  450.0, "Q": [0.23,0.24,0.45,1.10,1.72,1.84,1.32,0.90,0.60,0.45,0.38,0.21]},
    "IV":  {"H": 210.0, "L":  700.0, "Q": [0.29,0.33,0.54,1.32,1.98,2.16,1.44,1.08,0.72,0.54,0.45,0.34]},
    "V":   {"H":  70.0, "L":  800.0, "Q": [0.50,0.55,0.90,2.20,3.30,3.60,2.40,1.80,1.20,0.90,0.75,0.55]},
}

pipe_data = {
    "PVC": {
        0.05: {"pmax": 1.0,  "cost_pm": 10.0,  "eps": 1e-5},
        0.10: {"pmax": 0.95, "cost_pm": 15.0,  "eps": 1e-5},
        0.15: {"pmax": 0.90, "cost_pm": 20.0,  "eps": 1e-5},
        0.20: {"pmax": 0.85, "cost_pm": 27.5,  "eps": 1e-5},
        0.25: {"pmax": 0.80, "cost_pm": 37.0,  "eps": 1e-5},
        0.30: {"pmax": 0.75, "cost_pm": 50.0,  "eps": 1e-5},
        0.40: {"pmax": 0.65, "cost_pm": 75.0,  "eps": 1e-5},
        0.50: {"pmax": 0.55, "cost_pm": 100.0, "eps": 1e-5},
    },
    "Steel thin": {
        0.05: {"pmax": 13.1, "cost_pm": 25.0,  "eps": 1e-4},
        0.10: {"pmax": 10.7, "cost_pm": 37.5,  "eps": 1e-4},
        0.15: {"pmax": 8.6,  "cost_pm": 50.0,  "eps": 1e-4},
        0.20: {"pmax": 7.6,  "cost_pm": 68.8,  "eps": 1e-4},
        0.25: {"pmax": 3.8,  "cost_pm": 92.5,  "eps": 1e-4},
        0.30: {"pmax": 3.5,  "cost_pm": 125.0, "eps": 1e-4},
        0.40: {"pmax": 3.2,  "cost_pm": 187.5, "eps": 1e-4},
        0.50: {"pmax": 3.0,  "cost_pm": 250.0, "eps": 1e-4},
    },
    "Steel sch40": {
        0.05: {"pmax": 22.0, "cost_pm": 50.0,  "eps": 5e-4},
        0.10: {"pmax": 18.0, "cost_pm": 75.0,  "eps": 5e-4},
        0.15: {"pmax": 14.0, "cost_pm": 100.0, "eps": 5e-4},
        0.20: {"pmax": 12.5, "cost_pm": 137.5, "eps": 5e-4},
        0.25: {"pmax": 11.5, "cost_pm": 185.0, "eps": 5e-4},
        0.30: {"pmax": 10.8, "cost_pm": 250.0, "eps": 5e-4},
        0.40: {"pmax": 10.4, "cost_pm": 375.0, "eps": 5e-4},
        0.50: {"pmax": 10.1, "cost_pm": 500.0, "eps": 5e-4},
    },
}

valves = {
    "Discount": {"KL": 5.0, "pmax": 3.5},
    "Standard": {"KL": 0.5, "pmax": 7.5},
    "Low Loss": {"KL": 0.1, "pmax": 22.0},
}

valve_cost = {
    0.05: {"Discount": 500,  "Standard": 1000, "Low Loss": 1500},
    0.10: {"Discount": 700,  "Standard": 1400, "Low Loss": 2100},
    0.15: {"Discount": 900,  "Standard": 1800, "Low Loss": 2700},
    0.20: {"Discount": 1100, "Standard": 2200, "Low Loss": 3300},
    0.25: {"Discount": 1300, "Standard": 2600, "Low Loss": 3900},
    0.30: {"Discount": 1500, "Standard": 3000, "Low Loss": 4500},
    0.40: {"Discount": 1900, "Standard": 3800, "Low Loss": 5700},
    0.50: {"Discount": 2300, "Standard": 4600, "Low Loss": 6900},
}

turb_cost = {"K1":15000, "K2":25000, "F1":30000, "F2":40000, "P1":45000, "P2":60000, "P3":75000}

# approximate digitized turbine curves from the appendix graphs
turb = {
    "K1": {"H":[10,20,30,40,50,60,70,80],
           "qmin":[0.002,0.005,0.010,0.015,0.020,0.025,0.030,0.035],
           "qmax":[0.010,0.020,0.030,0.040,0.050,0.060,0.070,0.080],
           "eta":[0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.80]},
    "K2": {"H":[10,20,30,40,50,60,70,80],
           "qmin":[0.015,0.030,0.040,0.050,0.060,0.070,0.080,0.090],
           "qmax":[0.020,0.050,0.080,0.110,0.130,0.150,0.170,0.200],
           "eta":[0.70,0.73,0.75,0.78,0.80,0.83,0.85,0.88]},
    "F1": {"H":[40,70,100,130,160,190,220,250],
           "qmin":[0.004,0.006,0.008,0.011,0.014,0.017,0.020,0.022],
           "qmax":[0.012,0.016,0.020,0.025,0.030,0.035,0.040,0.045],
           "eta":[0.60,0.65,0.70,0.75,0.80,0.75,0.65,0.55]},
    "F2": {"H":[40,70,100,130,160,190,220,250],
           "qmin":[0.012,0.020,0.028,0.040,0.052,0.064,0.076,0.085],
           "qmax":[0.020,0.035,0.060,0.085,0.105,0.120,0.140,0.150],
           "eta":[0.65,0.70,0.75,0.80,0.85,0.80,0.70,0.60]},
    "P1": {"H":[100,170,250,320,400,470,550,620,700,770,850],
           "qmin":[0.0020,0.0022,0.0025,0.0030,0.0035,0.0040,0.0045,0.0048,0.0050,0.0053,0.0055],
           "qmax":[0.0030,0.0035,0.0040,0.0045,0.0050,0.0055,0.0062,0.0070,0.0080,0.0090,0.0100],
           "eta":[0.70,0.73,0.75,0.78,0.80,0.83,0.85,0.88,0.90,0.85,0.75]},
    "P2": {"H":[100,170,250,320,400,470,550,620,700,770,850],
           "qmin":[0.0040,0.0040,0.0050,0.0060,0.0080,0.0090,0.0110,0.0120,0.0130,0.0140,0.0150],
           "qmax":[0.0070,0.0080,0.0100,0.0120,0.0150,0.0170,0.0190,0.0210,0.0220,0.0230,0.0250],
           "eta":[0.72,0.74,0.78,0.80,0.82,0.85,0.88,0.90,0.92,0.87,0.77]},
    "P3": {"H":[100,170,250,320,400,470,550,620,700,770,850],
           "qmin":[0.010,0.012,0.015,0.018,0.021,0.024,0.027,0.030,0.032,0.035,0.038],
           "qmax":[0.014,0.016,0.019,0.023,0.028,0.032,0.036,0.040,0.043,0.045,0.043],
           "eta":[0.70,0.75,0.78,0.82,0.84,0.87,0.89,0.91,0.92,0.88,0.80]},
}

# sensible turbine-family filter by stage head
allowed = {
    "I":   ["F1", "F2", "P1", "P2", "P3"],
    "II":  ["F1", "F2", "P1", "P2", "P3"],
    "III": ["F1", "F2", "P1", "P2", "P3"],
    "IV":  ["F1", "F2", "P1", "P2", "P3"],
    "V":   ["K1", "K2", "F1", "F2"],
}

# -----------------------------
# helper functions
# -----------------------------
def interp(x, xp, fp):
    if x < xp[0] or x > xp[-1]:
        return None
    return float(np.interp(x, xp, fp))

def friction_factor(Re, eps, D):
    if Re <= 0:
        return None
    if Re < 2300:
        return 64.0 / Re
    return 0.25 / (math.log10(eps/(3.7*D) + 5.74/(Re**0.9))**2)

def turbine_match(kind, Hnet, Qavail):
    d = turb[kind]
    qmin = interp(Hnet, d["H"], d["qmin"])
    qmax = interp(Hnet, d["H"], d["qmax"])
    eta  = interp(Hnet, d["H"], d["eta"])
    if qmin is None or qmax is None or eta is None:
        return None
    quse = min(Qavail, qmax)
    if quse < qmin:
        return None
    return quse, eta

def stage_option(stage_name, pipe_mat, D, valve_name, turb_name, flow_scale=1.0):
    # evaluate one stage with one hardware combination across all 12 months
    H = stages[stage_name]["H"]
    L = stages[stage_name]["L"]
    pipe = pipe_data[pipe_mat][D]
    valve = valves[valve_name]

    # conservative static pressure check against 90% of rating
    p_static = rho * g * H / 1e6
    if p_static >= 0.9 * pipe["pmax"]:
        return None
    if p_static >= 0.9 * valve["pmax"]:
        return None

    kw = []

    for Qstream in stages[stage_name]["Q"]:
        Qavail = flow_scale * 0.10 * Qstream
        quse = Qavail

        # iterate because net head affects turbine operating window
        for _ in range(8):
            if quse <= 0:
                break

            V = 4.0 * quse / (math.pi * D**2)
            Re = V * D / nu
            f = friction_factor(Re, pipe["eps"], D)
            if f is None:
                quse = 0.0
                break

            hL = (f * L / D + valve["KL"]) * V**2 / (2.0 * g)
            Hnet = H - hL
            if Hnet <= 0:
                quse = 0.0
                break

            m = turbine_match(turb_name, Hnet, Qavail)
            if m is None:
                quse = 0.0
                break

            qnew, eta = m
            if abs(qnew - quse) < 1e-7:
                quse = qnew
                break
            quse = qnew

        if quse <= 0:
            kw.append(0.0)
            continue

        V = 4.0 * quse / (math.pi * D**2)
        Re = V * D / nu
        f = friction_factor(Re, pipe["eps"], D)
        hL = (f * L / D + valve["KL"]) * V**2 / (2.0 * g)
        Hnet = H - hL
        m = turbine_match(turb_name, Hnet, Qavail)

        if Hnet <= 0 or m is None:
            kw.append(0.0)
            continue

        quse, eta = m
        kw.append(rho * g * quse * Hnet * eta / 1000.0)

    cost = pipe["cost_pm"] * L + valve_cost[D][valve_name] + turb_cost[turb_name]

    return {
        "stage": stage_name,
        "pipe": pipe_mat,
        "D": D,
        "valve": valve_name,
        "turb": turb_name,
        "cost": float(cost),
        "kw": np.array(kw, dtype=float),
    }

def pareto_prune(options):
    # remove options that are worse in both cost and monthly power
    keep = []
    for i, a in enumerate(options):
        dominated = False
        for j, b in enumerate(options):
            if i == j:
                continue
            better_cost = b["cost"] <= a["cost"]
            better_kw = np.all(b["kw"] >= a["kw"] - 1e-9)
            strictly = (b["cost"] < a["cost"]) or np.any(b["kw"] > a["kw"] + 1e-9)
            if better_cost and better_kw and strictly:
                dominated = True
                break
        if not dominated:
            keep.append(a)
    return keep

def fixed_stage_options(stage):
    # all valid hardware options for one stage
    opts = []
    for pipe_mat in pipe_data:
        for D in pipe_data[pipe_mat]:
            for valve_name in valves:
                for turb_name in allowed[stage]:
                    o = stage_option(stage, pipe_mat, D, valve_name, turb_name, flow_scale=1.0)
                    if o is not None and np.max(o["kw"]) > 0:
                        opts.append(o)
    return sorted(pareto_prune(opts), key=lambda x: x["cost"])

def build_stage_options(flow_scale=1.0):
    # stage option set for the overall search, including "omit this stage"
    out = {}
    for s in stages:
        opts = [{"stage": s, "pipe": None, "D": None, "valve": None, "turb": None, "cost": 0.0, "kw": np.zeros(12)}]
        for pipe_mat in pipe_data:
            for D in pipe_data[pipe_mat]:
                for valve_name in valves:
                    for turb_name in allowed[s]:
                        o = stage_option(s, pipe_mat, D, valve_name, turb_name, flow_scale)
                        if o is not None and np.max(o["kw"]) > 0:
                            opts.append(o)
        out[s] = sorted(pareto_prune(opts), key=lambda x: x["cost"])
    return out

def score_combo(combo):
    kw = np.sum([o["kw"] for o in combo], axis=0) if combo else np.zeros(12)
    return {
        "cost": float(sum(o["cost"] for o in combo)),
        "kw": kw,
        "jan_kw": float(kw[0]),
        "min_kw": float(np.min(kw)),
        "months_ge_100": int(np.sum(kw >= 100.0)),
    }

def better_power(a, b):
    return (
        b is None
        or a["min_kw"] > b["min_kw"] + 1e-9
        or (abs(a["min_kw"] - b["min_kw"]) < 1e-9 and a["cost"] < b["cost"])
    )

def combo_text(combo):
    if not combo:
        return "-"
    return "; ".join(
        f"{o['stage']}: {o['pipe']}, {o['D']:.2f} m, {o['valve']}, {o['turb']}"
        for o in combo
    )

def search_fixed_subset(stage_list):
    # force every stage in the subset to be used
    opts = {s: fixed_stage_options(s) for s in stage_list}
    best_feasible = None
    best_power = None

    def rec(i, chosen):
        nonlocal best_feasible, best_power

        if i == len(stage_list):
            row = score_combo(chosen)
            row["combo"] = chosen.copy()
            row["subset"] = "+".join(stage_list)

            if better_power(row, best_power):
                best_power = row

            if np.all(row["kw"] >= 100.0):
                if best_feasible is None or row["cost"] < best_feasible["cost"]:
                    best_feasible = row
            return

        for o in opts[stage_list[i]]:
            rec(i + 1, chosen + [o])

    rec(0, [])
    return best_feasible, best_power

def find_best_average_year():
    # full search where each stage can be included or omitted
    order = ["I","II","III","IV","V"]
    opts = build_stage_options(flow_scale=1.0)

    rem_max = [None] * (len(order) + 1)
    rem_max[-1] = np.zeros(12)
    for i in range(len(order) - 1, -1, -1):
        stage_max = np.max(np.array([o["kw"] for o in opts[order[i]]]), axis=0)
        rem_max[i] = rem_max[i + 1] + stage_max

    best_cost = float("inf")
    best_combo = None

    def dfs(i, cost, kw, combo):
        nonlocal best_cost, best_combo

        if cost >= best_cost:
            return

        # prune if even the best remaining stages cannot hit 100 kW every month
        if np.any(kw + rem_max[i] < 100.0):
            return

        if i == len(order):
            if np.all(kw >= 100.0):
                best_cost = cost
                best_combo = combo[:]
            return

        deficit = np.maximum(0.0, 100.0 - kw)
        cand = sorted(
            opts[order[i]],
            key=lambda o: (-(np.minimum(deficit, o["kw"]).sum() / (o["cost"] + 1.0)), o["cost"])
        )

        for o in cand:
            dfs(i + 1, cost + o["cost"], kw + o["kw"], combo + [o])

    dfs(0, 0.0, np.zeros(12), [])
    return best_cost, best_combo

def evaluate_combo(combo, flow_scale):
    # re-evaluate a chosen design under average year or drought year
    total_kw = np.sum(
        [stage_option(o["stage"], o["pipe"], o["D"], o["valve"], o["turb"], flow_scale)["kw"]
         if o["turb"] is not None else np.zeros(12) for o in combo],
        axis=0
    )
    diesel_kw = np.maximum(0.0, 100.0 - total_kw)
    diesel_kwh = diesel_kw * 24.0 * 30.0
    diesel_kg = diesel_kwh * 0.230
    ghg_kg = diesel_kg * 3.2
    return total_kw, diesel_kw, diesel_kg, ghg_kg

def make_row(name, feasible, best_power):
    row = feasible if feasible is not None else best_power
    return {
        "subset": name,
        "status": "least-cost feasible" if feasible is not None else "no feasible 12-month design",
        "cost_$": round(row["cost"], 0),
        "jan_kw": round(row["jan_kw"], 1),
        "min_kw": round(row["min_kw"], 1),
        "months>=100": row["months_ge_100"],
        "config": combo_text(row["combo"]),
    }

# -----------------------------
# run requested subsets
# -----------------------------
subset_specs = [
    ["III", "IV"],
    ["III", "IV", "V"],
    ["II", "III", "IV"],
    ["II", "III", "IV", "V"],
]

summary = []

for ss in subset_specs:
    feas, best = search_fixed_subset(ss)
    summary.append(make_row("+".join(ss), feas, best))

# overall least-cost feasible design
overall_cost, overall_combo = find_best_average_year()
overall_combo = [o for o in overall_combo if o["turb"] is not None]
overall = score_combo(overall_combo)
overall["combo"] = overall_combo

summary.append({
    "subset": "overall",
    "status": "least-cost feasible",
    "cost_$": round(overall["cost"], 0),
    "jan_kw": round(overall["jan_kw"], 1),
    "min_kw": round(overall["min_kw"], 1),
    "months>=100": overall["months_ge_100"],
    "config": combo_text(overall["combo"]),
})

# -----------------------------
# print summary table
# -----------------------------
print(f"{'subset':<16} {'status':<28} {'cost_$':>10} {'jan_kw':>8} {'min_kw':>8} {'m>=100':>8}")
for r in summary:
    print(f"{r['subset']:<16} {r['status']:<28} {r['cost_$']:>10,.0f} {r['jan_kw']:>8.1f} {r['min_kw']:>8.1f} {r['months>=100']:>8}")
    print("  " + r["config"])

# -----------------------------
# drought-year results for overall winner
# -----------------------------
avg_kw, avg_diesel_kw, avg_diesel_kg, avg_ghg_kg = evaluate_combo(overall_combo, flow_scale=1.0)
dry_kw, dry_diesel_kw, dry_diesel_kg, dry_ghg_kg = evaluate_combo(overall_combo, flow_scale=0.6)

print("\noverall average-year hydro power by month [kW]")
print(np.round(avg_kw, 1))

print("\noverall drought-year hydro power by month [kW]")
print(np.round(dry_kw, 1))

print("\noverall drought-year diesel shortfall by month [kW]")
print(np.round(dry_diesel_kw, 1))

print("\noverall drought-year diesel use by month [kg/month]")
print(np.round(dry_diesel_kg, 1))

print("\noverall drought-year ghg by month [kg CO2e/month]")
print(np.round(dry_ghg_kg, 1))