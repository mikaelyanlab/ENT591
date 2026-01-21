# why_openness_isnt_life_studio.py
# Streamlit App 2 (student-facing): visuals only.
# All questions + answer spaces belong in a Google Doc (not in the app).

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Why Openness Isn’t Life Studio", layout="wide")


# -----------------------------
# Core model (toy but structured)
# -----------------------------
def clamp01(x):
    return np.minimum(1.0, np.maximum(0.0, x))


def simulate(params, initial, perturbation, seed=0):
    """
    State variables (0..1):
      E = energy reserve
      I = integrity (wear vs repair)
      C = core condition (regulated variable; target C*)
      V = viability (composite of E, I, and closeness to target)

    Crossings:
      Matter -> enables throughput (resource inflow)
      Energy -> enables thermal coupling to environment
      Information -> enables sensing-based control (if control mechanism on)

    Mechanisms:
      Self-repair/maintenance -> enables repair flow (capped by repair capacity)
      Feedback control -> enables control effort (capped by actuator strength, delayed)
      Caretaker refuel -> adds extra throughput, but only if caretaker is inside boundary
    """
    rng = np.random.default_rng(seed)

    T = params["steps"]
    dt = 1.0

    # Switches
    matter_on = params["cross_matter"]
    energy_on = params["cross_energy"]
    info_on = params["cross_info"]

    repair_on = params["mech_repair"]
    control_on = params["mech_control"]
    caretaker_refuel_on = params["mech_refuel"]

    duck_type = params["duck_type"]  # "Living" or "Mechanical"
    boundary = params["boundary"]    # "Duck only", "Duck+environment", "Duck+caretaker"

    # Targets + environment coupling
    C_target = params["core_target"]
    env_temp = params["env_temp"]
    env_coupling = params["env_coupling"] if energy_on else 0.0

    # Driving + costs
    throughput = params["throughput"] if matter_on else 0.0
    maintenance_cost = params["maintenance_cost"]

    # Control constraints
    actuator_max = params["actuator_strength"] if (control_on and info_on) else 0.0
    delay_steps = int(params["control_delay"]) if (control_on and info_on) else 0

    # Repair constraints
    repair_cap = params["repair_capacity"] if repair_on else 0.0

    # Noise + disturbance
    noise = params["noise"]
    shock_time = perturbation["shock_time"]
    shock_mag = perturbation["shock_mag"]
    shock_width = perturbation["shock_width"]

    # Caretaker refuel only effective if caretaker is inside boundary
    caretaker_inside = (boundary == "Duck+caretaker")
    refuel_effective = caretaker_refuel_on and caretaker_inside and matter_on
    refuel_boost = params["refuel_boost"] if refuel_effective else 0.0

    # Mechanical duck: weak control/repair, higher overhead
    if duck_type == "Mechanical":
        actuator_max *= params["mechanical_control_discount"]
        repair_cap *= params["mechanical_repair_discount"]
        maintenance_cost *= params["mechanical_maintenance_multiplier"]

    # Arrays
    E = np.zeros(T)
    I = np.zeros(T)
    C = np.zeros(T)
    V = np.zeros(T)

    # Diagnostics
    control_effort = np.zeros(T)
    repair_flow = np.zeros(T)
    maintenance_flow = np.zeros(T)
    throughput_flow = np.zeros(T)
    wear_flow = np.zeros(T)
    error_trace = np.zeros(T)

    # Delay buffer
    err_buffer = [0.0] * (delay_steps + 1)

    # Init
    E[0], I[0], C[0] = initial["E"], initial["I"], initial["C"]

    def viability(e, i, c):
        core_ok = 1.0 - np.abs(c - C_target)
        v = 0.40 * e + 0.35 * i + 0.25 * core_ok
        return clamp01(v)

    V[0] = float(viability(E[0], I[0], C[0]))

    for t in range(1, T):
        # Cold shock: transiently reduces environment temperature
        shock_active = (shock_time <= t < shock_time + shock_width)
        env_temp_t = env_temp - (shock_mag if shock_active else 0.0)
        env_temp_t = float(clamp01(env_temp_t))

        # Throughput (resource inflow)
        effective_throughput = throughput + refuel_boost
        throughput_flow[t] = effective_throughput
        dE_in = params["assimilation"] * effective_throughput * (1.0 - E[t - 1])

        # Maintenance drain (higher when integrity is low)
        maint = maintenance_cost * (params["maint_base"] + params["maint_integrity_weight"] * (1.0 - I[t - 1]))
        maintenance_flow[t] = maint

        # Control (negative feedback): only if info + control
        error = (C_target - C[t - 1])
        error_trace[t] = error
        err_buffer.append(error)
        delayed_error = err_buffer.pop(0)

        u_raw = params["control_gain"] * delayed_error
        u = float(np.clip(u_raw, -actuator_max, actuator_max))
        control_effort[t] = u

        # Core condition dynamics: environment coupling + control + energy penalty + noise
        dC_env = env_coupling * (env_temp_t - C[t - 1])
        dC_ctrl = u
        dC_energy_penalty = -params["core_energy_drag"] * max(0.0, (0.3 - E[t - 1]))
        dC_noise = noise * rng.normal(0.0, 1.0)

        C[t] = float(clamp01(C[t - 1] + dt * (dC_env + dC_ctrl + dC_energy_penalty) + dC_noise))

        # Integrity dynamics: wear vs repair
        wear = params["wear_rate"] * (params["wear_base"] + params["wear_control_weight"] * abs(u))
        wear_flow[t] = wear

        demand = (1.0 - I[t - 1]) * params["repair_demand_gain"]
        potential_repair = min(repair_cap, demand)
        actual_repair = potential_repair * min(1.0, E[t - 1] / params["repair_energy_scale"])
        repair_flow[t] = actual_repair

        I[t] = float(clamp01(I[t - 1] + dt * (actual_repair - wear)))

        # Energy update: inflow - maintenance - costs of control + repair
        control_cost = params["control_energy_cost"] * abs(u)
        repair_cost = params["repair_energy_cost"] * actual_repair
        E[t] = float(clamp01(E[t - 1] + dt * (dE_in - maint - control_cost - repair_cost)))

        # Viability
        V[t] = float(viability(E[t], I[t], C[t]))

    # Summary diagnostics
    minV = float(np.min(V))
    finalV = float(V[-1])

    # Regime label
    v_std = float(np.std(V[int(0.4 * T):])) if T > 5 else 0.0
    if finalV < params["collapse_threshold"] or minV < params["collapse_threshold"]:
        regime = "Collapse / drift"
    elif v_std > params["osc_threshold"]:
        regime = "Oscillatory"
    else:
        regime = "Regulated steady"

    # Binding constraint heuristic (coarse)
    sat_frac = float(np.mean(np.isclose(np.abs(control_effort), actuator_max, atol=1e-6))) if actuator_max > 0 else 0.0
    low_I = float(np.mean(I < 0.4))
    repair_cap_hit = float(np.mean(repair_flow > 0.9 * repair_cap)) if repair_cap > 0 else 0.0
    energy_stress = float(np.mean(E < 0.3))
    maint_frac = float(np.mean(maintenance_flow > (throughput_flow * params["maint_vs_throughput_scale"])))

    if sat_frac > 0.15 and actuator_max > 0:
        binding = "Actuator saturation"
    elif (repair_cap_hit > 0.20 and low_I > 0.20 and repair_cap > 0):
        binding = "Repair capacity"
    elif energy_stress > 0.25 or maint_frac > 0.25:
        binding = "Energy / maintenance budget"
    elif delay_steps >= 3 and (regime == "Oscillatory"):
        binding = "Control delay"
    else:
        binding = "No single dominant constraint"

    return {
        "E": E, "I": I, "C": C, "V": V,
        "control_effort": control_effort,
        "repair_flow": repair_flow,
        "maintenance_flow": maintenance_flow,
        "throughput_flow": throughput_flow,
        "wear_flow": wear_flow,
        "error_trace": error_trace,
        "minV": minV,
        "finalV": finalV,
        "regime": regime,
        "binding": binding,
        "actuator_max": actuator_max,
    }


# -----------------------------
# Plot helpers
# -----------------------------
def plot_timeseries(out, steps, title):
    x = np.arange(steps)
    fig = plt.figure(figsize=(9, 4.0))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, out["V"], label="Viability")
    ax.plot(x, out["C"], label="Core")
    ax.plot(x, out["E"], label="Energy")
    ax.plot(x, out["I"], label="Integrity")
    ax.set_xlabel("Step")
    ax.set_ylabel("0–1")
    ax.set_title(title)
    ax.legend(loc="best")
    return fig


def plot_budget(out, steps, title):
    x = np.arange(steps)
    fig = plt.figure(figsize=(9, 4.0))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, out["throughput_flow"], label="Throughput (in)")
    ax.plot(x, out["maintenance_flow"], label="Maintenance (cost)")
    ax.plot(x, np.abs(out["control_effort"]), label="|Control effort|")
    ax.plot(x, out["repair_flow"], label="Repair flow")
    ax.set_xlabel("Step")
    ax.set_ylabel("arb. units")
    ax.set_title(title)
    ax.legend(loc="best")
    return fig


def plot_control(out, steps, title):
    x = np.arange(steps)
    fig = plt.figure(figsize=(9, 4.0))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, out["error_trace"], label="Error (target − core)")
    ax.plot(x, out["control_effort"], label="Control effort (u)")
    if out["actuator_max"] > 0:
        ax.axhline(out["actuator_max"], linestyle="--", linewidth=1, label="Actuator cap (+)")
        ax.axhline(-out["actuator_max"], linestyle="--", linewidth=1, label="Actuator cap (−)")
    ax.set_xlabel("Step")
    ax.set_ylabel("signal")
    ax.set_title(title)
    ax.legend(loc="best")
    return fig


def plot_equifinality_overlay(params, perturbation, seed=0):
    initials = [
        {"name": "IC-A", "E": 0.25, "I": 0.85, "C": params["core_target"]},
        {"name": "IC-B", "E": 0.75, "I": 0.35, "C": params["core_target"]},
        {"name": "IC-C", "E": 0.75, "I": 0.85, "C": 0.35},
    ]
    outs = [simulate(params, ic, perturbation, seed=seed + i) for i, ic in enumerate(initials)]

    fig = plt.figure(figsize=(9, 4.0))
    ax = fig.add_subplot(1, 1, 1)
    for ic, out in zip(initials, outs):
        ax.plot(np.arange(params["steps"]), out["V"], label=ic["name"])
    ax.set_xlabel("Step")
    ax.set_ylabel("Viability (0–1)")
    ax.set_title("Equifinality overlay (Viability)")
    ax.legend(loc="best")

    finals = np.array([o["finalV"] for o in outs])
    eq_present = (np.max(finals) - np.min(finals)) < 0.08
    return fig, eq_present


def plot_regime_map(base_params, perturbation, seed=0):
    thr_vals = np.linspace(0.0, 1.0, 21)
    maint_vals = np.linspace(0.0, 1.0, 21)
    regime_code = np.zeros((len(maint_vals), len(thr_vals)))  # 0 steady, 1 osc, 2 collapse

    ic = {"E": 0.75, "I": 0.85, "C": base_params["core_target"]}

    for i, m in enumerate(maint_vals):
        for j, th in enumerate(thr_vals):
            p = dict(base_params)
            p["throughput"] = float(th)
            p["maintenance_cost"] = float(m)
            out = simulate(p, ic, perturbation, seed=seed)
            if out["regime"] == "Regulated steady":
                regime_code[i, j] = 0
            elif out["regime"] == "Oscillatory":
                regime_code[i, j] = 1
            else:
                regime_code[i, j] = 2

    fig = plt.figure(figsize=(7.2, 5.2))
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(
        regime_code,
        origin="lower",
        aspect="auto",
        extent=[thr_vals[0], thr_vals[-1], maint_vals[0], maint_vals[-1]]
    )
    ax.set_xlabel("Throughput")
    ax.set_ylabel("Maintenance cost")
    ax.set_title("Regime map")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_ticks([0, 1, 2])
    cbar.set_ticklabels(["Steady", "Osc", "Collapse"])
    return fig


# -----------------------------
# App UI (minimal; App-1 style)
# -----------------------------
st.title("Why Openness Isn’t Life Studio")

# Sidebar: controls only (no instructions)
with st.sidebar:
    st.header("Scenario")
    st.write("Mechanical duck vs living duck")

    st.divider()
    st.header("System")

    duck_type = st.radio("Duck type", ["Living", "Mechanical"], index=0)

    boundary = st.selectbox(
        "Boundary",
        ["Duck only", "Duck+environment", "Duck+caretaker"],
        index=0
    )

    st.subheader("Crossings")
    cross_matter = st.checkbox("Matter", value=True)
    cross_energy = st.checkbox("Energy", value=True)
    cross_info = st.checkbox("Information", value=True)

    st.subheader("Mechanisms")
    mech_repair = st.checkbox("Self-repair / maintenance", value=True if duck_type == "Living" else False)
    mech_control = st.checkbox("Feedback control", value=True)
    mech_refuel = st.checkbox("Caretaker refuel", value=False)

    st.divider()
    st.header("Constraints")
    throughput = st.slider("Throughput", 0.0, 1.0, 0.65, 0.01)
    maintenance_cost = st.slider("Maintenance cost", 0.0, 1.0, 0.35, 0.01)
    repair_capacity = st.slider("Repair capacity", 0.0, 1.0, 0.55, 0.01)
    actuator_strength = st.slider("Actuator strength", 0.0, 1.0, 0.50, 0.01)
    control_delay = st.slider("Control delay (steps)", 0, 10, 1, 1)
    noise = st.slider("Noise", 0.0, 0.10, 0.01, 0.005)

    st.divider()
    st.header("Disturbance")
    shock_time_mode = st.radio("Cold shock time", ["Early", "Mid", "Late"], index=1, horizontal=True)
    shock_mag = st.slider("Magnitude", 0.0, 0.60, 0.15, 0.01)
    shock_width = st.slider("Width (steps)", 1, 30, 8, 1)

    st.divider()
    st.header("Simulation")
    steps = st.slider("Steps", 60, 300, 160, 10)
    env_temp = st.slider("Environment temp", 0.0, 1.0, 0.70, 0.01)
    env_coupling = st.slider("Energy coupling", 0.0, 0.25, 0.08, 0.01)
    seed = st.number_input("Seed", min_value=0, max_value=10_000, value=0, step=1)

# Shock timing preset
if shock_time_mode == "Early":
    shock_time = int(0.15 * steps)
elif shock_time_mode == "Mid":
    shock_time = int(0.45 * steps)
else:
    shock_time = int(0.75 * steps)

perturbation = {"shock_time": shock_time, "shock_mag": shock_mag, "shock_width": shock_width}

# Params bundle
params = {
    "duck_type": duck_type,
    "boundary": boundary,

    "cross_matter": cross_matter,
    "cross_energy": cross_energy,
    "cross_info": cross_info,

    "mech_repair": mech_repair,
    "mech_control": mech_control,
    "mech_refuel": mech_refuel,

    "steps": steps,

    # Targets / environment
    "core_target": 0.75,
    "env_temp": env_temp,
    "env_coupling": env_coupling,

    # Driving and costs
    "throughput": throughput,
    "maintenance_cost": maintenance_cost,

    # Constraints
    "repair_capacity": repair_capacity,
    "actuator_strength": actuator_strength,
    "control_delay": control_delay,

    # Process constants
    "assimilation": 0.18,
    "maint_base": 0.20,
    "maint_integrity_weight": 0.45,
    "wear_rate": 0.040,
    "wear_base": 0.30,
    "wear_control_weight": 0.60,
    "repair_demand_gain": 1.10,
    "repair_energy_scale": 0.35,
    "control_gain": 0.55,
    "core_energy_drag": 0.06,
    "control_energy_cost": 0.10,
    "repair_energy_cost": 0.12,

    # Regime thresholds
    "collapse_threshold": 0.30,
    "osc_threshold": 0.06,

    # Helpers for binding constraint heuristic
    "maint_vs_throughput_scale": 0.90,

    # Caretaker refuel
    "refuel_boost": 0.20,

    # Mechanical duck discounts
    "mechanical_control_discount": 0.25,
    "mechanical_repair_discount": 0.10,
    "mechanical_maintenance_multiplier": 1.20,

    # Noise
    "noise": noise,
}

initial_default = {"E": 0.75, "I": 0.85, "C": params["core_target"]}
out = simulate(params, initial_default, perturbation, seed=int(seed))

# Top-row: observables only
m1, m2, m3, m4 = st.columns(4)
m1.metric("Final viability", f"{out['finalV']:.2f}")
m2.metric("Minimum viability", f"{out['minV']:.2f}")
m3.metric("Regime", out["regime"])
m4.metric("Binding constraint", out["binding"])

# Tabs: visuals only
tab_bert, tab_prig, tab_wien = st.tabs(["Bertalanffy", "Prigogine", "Wiener"])

with tab_bert:
    left, right = st.columns([1.25, 1.0])
    with left:
        fig_eq, eq_present = plot_equifinality_overlay(params, perturbation, seed=int(seed))
        st.pyplot(fig_eq, clear_figure=True)
    with right:
        st.pyplot(plot_timeseries(out, steps, title="Time series (current setting)"), clear_figure=True)
        st.metric("Equifinality indicator", "Present" if eq_present else "Absent")

with tab_prig:
    left, right = st.columns([1.25, 1.0])
    with left:
        st.pyplot(plot_timeseries(out, steps, title="Time series (current setting)"), clear_figure=True)
        st.pyplot(plot_budget(out, steps, title="Budget / flows (current setting)"), clear_figure=True)
    with right:
        st.pyplot(plot_regime_map(params, perturbation, seed=int(seed)), clear_figure=True)

with tab_wien:
    left, right = st.columns([1.25, 1.0])
    with left:
        st.pyplot(plot_timeseries(out, steps, title="Time series (current setting)"), clear_figure=True)
        st.pyplot(plot_control(out, steps, title="Control signals (current setting)"), clear_figure=True)
    with right:
        st.pyplot(plot_budget(out, steps, title="Budget / flows (current setting)"), clear_figure=True)
