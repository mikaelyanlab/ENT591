# app2_three_upgrades_studio.py
# Streamlit App 2: "Three Upgrades Studio" (von Bertalanffy, Prigogine, Wiener)
#
# Design goals (per your constraints):
# - No text-input answers from students (no free-response boxes).
# - Interaction only via toggles/menus/sliders that update in-app visuals.
# - Claim-testing is implicit: students infer from plots + graph + auto-badges.
# - Builds directly on Duck App 1 (same scenario, same boundary/crossings/mechanisms framing),
#   but moves the conceptual load to: steady state/equifinality (Bertalanffy),
#   dissipative regimes/thresholds (Prigogine), and constrained feedback (Wiener).

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Three Upgrades Studio", layout="wide")


# -----------------------------
# Core model (toy but structured)
# -----------------------------
def clamp01(x):
    return np.minimum(1.0, np.maximum(0.0, x))


def simulate(params, initial, perturbation, seed=0):
    """
    State variables:
      E = energy reserve (0..1)
      I = integrity (0..1)  [wear vs repair]
      C = core condition (0..1)  [regulated variable; target C*]
      V = viability (0..1)  [composite of E, I, and closeness to target C*]

    Key idea:
      - Crossings (matter/energy/info) enable specific causal routes.
      - Mechanisms (repair/control/refuel) add internal capacities.
      - Constraints (maintenance cost, actuator cap, repair cap, delay) decide what *works*.

    Returns:
      dict with time series arrays and diagnostic traces.
    """
    rng = np.random.default_rng(seed)

    T = params["steps"]
    dt = 1.0

    # Unpack switches
    matter_on = params["cross_matter"]
    energy_on = params["cross_energy"]
    info_on = params["cross_info"]

    repair_on = params["mech_repair"]
    control_on = params["mech_control"]
    caretaker_refuel_on = params["mech_refuel"]

    duck_type = params["duck_type"]  # "Living" or "Mechanical"
    boundary = params["boundary"]    # "Duck only", "Duck+environment", "Duck+caretaker"

    # Core targets and couplings
    C_target = params["core_target"]
    env_temp = params["env_temp"]  # in "normalized" 0..1 for this toy model
    env_coupling = params["env_coupling"] if energy_on else 0.0

    # Throughput and costs
    throughput = params["throughput"] if matter_on else 0.0
    maintenance_cost = params["maintenance_cost"]

    # Control + constraint knobs
    actuator_max = params["actuator_strength"] if (control_on and info_on) else 0.0
    delay_steps = int(params["control_delay"]) if (control_on and info_on) else 0

    # Repair knobs
    repair_cap = params["repair_capacity"] if repair_on else 0.0

    # Noise and disturbance
    noise = params["noise"]
    shock_time = perturbation["shock_time"]
    shock_mag = perturbation["shock_mag"]
    shock_width = perturbation["shock_width"]

    # Caretaker refuel only matters if caretaker is inside boundary
    caretaker_inside = (boundary == "Duck+caretaker")
    refuel_effective = caretaker_refuel_on and caretaker_inside and matter_on
    refuel_boost = params["refuel_boost"] if refuel_effective else 0.0

    # Mechanical duck differences:
    # - It can "have" toggled mechanisms, but we intentionally weaken them to show the contrast.
    #   This makes the app robust to the case you observed where control slightly changes viability.
    if duck_type == "Mechanical":
        # Crude, low-gain “pseudo-control”: e.g., friction-triggered throttling (weak).
        actuator_max *= params["mechanical_control_discount"]
        # Repair is minimal: e.g., no self-repair; at best, slowing wear slightly.
        repair_cap *= params["mechanical_repair_discount"]
        # Mechanical maintenance is "wear-like" (can rise with actuation).
        maintenance_cost *= params["mechanical_maintenance_multiplier"]

    # Allocate arrays
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

    # Delay buffer for control
    err_buffer = [0.0] * (delay_steps + 1)

    # Initialize
    E[0], I[0], C[0] = initial["E"], initial["I"], initial["C"]

    # Helper: viability from states
    def viability(e, i, c):
        core_ok = 1.0 - np.abs(c - C_target)  # 1 when at target, 0 when far
        # Weighted composite (no need for sigmoid; keep interpretable)
        v = 0.40 * e + 0.35 * i + 0.25 * core_ok
        return clamp01(v)

    V[0] = viability(E[0], I[0], C[0])

    for t in range(1, T):
        # --- disturbance: cold shock acts as a transient downward push on env temp OR core condition
        shock_active = (shock_time <= t < shock_time + shock_width)
        env_temp_t = env_temp - (shock_mag if shock_active else 0.0)
        env_temp_t = clamp01(env_temp_t)

        # --- Throughput supplies energy (matter crossing)
        # Think: food/oxygen allows free-energy production; simplified as an inflow.
        # Refuel adds extra throughput if caretaker is inside boundary.
        effective_throughput = throughput + refuel_boost
        throughput_flow[t] = effective_throughput

        dE_in = params["assimilation"] * effective_throughput * (1.0 - E[t-1])

        # --- Maintenance drains energy (baseline + more when integrity is low)
        maint = maintenance_cost * (params["maint_base"] + params["maint_integrity_weight"] * (1.0 - I[t-1]))
        maintenance_flow[t] = maint

        # --- Control: negative feedback on core condition (if info + control)
        error = (C_target - C[t-1])
        error_trace[t] = error
        err_buffer.append(error)
        delayed_error = err_buffer.pop(0)  # oldest

        # Control effort capped by actuator_max (saturation)
        u_raw = params["control_gain"] * delayed_error
        u = float(np.clip(u_raw, -actuator_max, actuator_max))
        control_effort[t] = u

        # --- Core condition dynamics:
        # Drift toward environment when energy crossing is on.
        dC_env = env_coupling * (env_temp_t - C[t-1])

        # Control pushes core back to target (if any actuation exists)
        dC_ctrl = u

        # Small intrinsic drift toward "wear" if energy is low (optional realism)
        dC_energy_penalty = -params["core_energy_drag"] * max(0.0, (0.3 - E[t-1]))

        dC_noise = noise * rng.normal(0.0, 1.0)

        C_t = C[t-1] + dt * (dC_env + dC_ctrl + dC_energy_penalty) + dC_noise
        C[t] = clamp01(C_t)

        # --- Integrity dynamics: wear vs repair
        wear = params["wear_rate"] * (params["wear_base"] + params["wear_control_weight"] * abs(u))
        wear_flow[t] = wear

        # Repair demand rises as integrity falls
        demand = (1.0 - I[t-1]) * params["repair_demand_gain"]
        # Repair flow capped (constraint) and requires energy to pay
        potential_repair = min(repair_cap, demand)
        # Energy-limited repair
        actual_repair = potential_repair * min(1.0, E[t-1] / params["repair_energy_scale"])
        repair_flow[t] = actual_repair

        I_t = I[t-1] + dt * (actual_repair - wear)
        I[t] = clamp01(I_t)

        # --- Energy update: inflow - maintenance - cost of control - cost of repair
        control_cost = params["control_energy_cost"] * abs(u)
        repair_cost = params["repair_energy_cost"] * actual_repair

        E_t = E[t-1] + dt * (dE_in - maint - control_cost - repair_cost)
        E[t] = clamp01(E_t)

        # --- Viability
        V[t] = viability(E[t], I[t], C[t])

    # Diagnostics summary
    minV = float(np.min(V))
    finalV = float(V[-1])

    # Simple regime classification
    # (You can tune thresholds; these are meant to be stable teaching bins.)
    v_std = float(np.std(V[int(0.4*T):]))  # variability after initial transient
    if finalV < params["collapse_threshold"] or minV < params["collapse_threshold"]:
        regime = "Collapse / drift"
    elif v_std > params["osc_threshold"]:
        regime = "Oscillatory"
    else:
        regime = "Regulated steady"

    # Binding constraint heuristic
    # Decide what "bottleneck" dominated:
    # - If actuator saturates frequently -> actuator constraint
    # - If repair demand >> cap while integrity low -> repair constraint
    # - If maintenance dominates energy budget -> energy constraint
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
        "repair_cap": repair_cap,
        "delay_steps": delay_steps
    }


def plot_timeseries(out, params, title_suffix=""):
    T = len(out["V"])
    x = np.arange(T)

    fig = plt.figure(figsize=(9, 4.2))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, out["V"], label="Viability")
    ax.plot(x, out["C"], label="Core condition")
    ax.plot(x, out["E"], label="Energy reserve")
    ax.plot(x, out["I"], label="Integrity")
    ax.set_xlabel("Step")
    ax.set_ylabel("State (0–1)")
    ax.set_title(f"Time series {title_suffix}".strip())
    ax.legend(loc="best")
    return fig


def plot_constraint_dashboard(out, params, title_suffix=""):
    T = len(out["V"])
    x = np.arange(T)

    fig = plt.figure(figsize=(9, 4.2))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, out["throughput_flow"], label="Throughput (in)")
    ax.plot(x, out["maintenance_flow"], label="Maintenance (cost)")
    ax.plot(x, np.abs(out["control_effort"]), label="|Control effort|")
    ax.plot(x, out["repair_flow"], label="Repair flow")
    ax.set_xlabel("Step")
    ax.set_ylabel("Flow (arbitrary units)")
    ax.set_title(f"Constraint dashboard {title_suffix}".strip())
    ax.legend(loc="best")
    return fig


def plot_equifinality_overlay(params, perturbation, seed=0):
    initials = [
        {"name": "IC-A (low E)", "E": 0.25, "I": 0.85, "C": params["core_target"]},
        {"name": "IC-B (low I)", "E": 0.75, "I": 0.35, "C": params["core_target"]},
        {"name": "IC-C (low C)", "E": 0.75, "I": 0.85, "C": 0.35},
    ]
    outs = [simulate(params, ic, perturbation, seed=seed+i) for i, ic in enumerate(initials)]

    fig = plt.figure(figsize=(9, 4.2))
    ax = fig.add_subplot(1, 1, 1)
    for ic, out in zip(initials, outs):
        ax.plot(np.arange(params["steps"]), out["V"], label=ic["name"])
    ax.set_xlabel("Step")
    ax.set_ylabel("Viability (0–1)")
    ax.set_title("Equifinality test: do different starts converge?")
    ax.legend(loc="best")

    # Equifinality badge: end values close
    finals = np.array([o["finalV"] for o in outs])
    eq = (np.max(finals) - np.min(finals)) < 0.08  # tolerance
    return fig, eq, outs


def plot_regime_map(base_params, perturbation, seed=0):
    # 2D grid: throughput vs maintenance cost -> regime classification
    thr_vals = np.linspace(0.0, 1.0, 21)
    maint_vals = np.linspace(0.0, 1.0, 21)

    regime_code = np.zeros((len(maint_vals), len(thr_vals)))  # 0 steady, 1 osc, 2 collapse

    # Use a fixed initial condition representative of "alive"
    ic = {"E": 0.75, "I": 0.85, "C": base_params["core_target"]}

    for i, m in enumerate(maint_vals):
        for j, th in enumerate(thr_vals):
            p = dict(base_params)
            p["throughput"] = th
            p["maintenance_cost"] = m
            out = simulate(p, ic, perturbation, seed=seed)
            if out["regime"] == "Regulated steady":
                regime_code[i, j] = 0
            elif out["regime"] == "Oscillatory":
                regime_code[i, j] = 1
            else:
                regime_code[i, j] = 2

    fig = plt.figure(figsize=(7.0, 5.4))
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(
        regime_code,
        origin="lower",
        aspect="auto",
        extent=[thr_vals[0], thr_vals[-1], maint_vals[0], maint_vals[-1]]
    )
    ax.set_xlabel("Throughput (driving)")
    ax.set_ylabel("Maintenance cost (overhead)")
    ax.set_title("Prigogine regime map: where does order persist?")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_ticks([0, 1, 2])
    cbar.set_ticklabels(["Steady", "Oscillatory", "Collapse"])
    return fig


# -----------------------------
# UI: global controls
# -----------------------------
st.title("Three Upgrades Studio: Bertalanffy • Prigogine • Wiener")

with st.sidebar:
    st.header("System definition (builds on App 1)")

    duck_type = st.radio("Duck type", ["Living", "Mechanical"], index=0)

    boundary = st.selectbox(
        "Boundary",
        ["Duck only", "Duck+environment", "Duck+caretaker"],
        index=0
    )

    st.subheader("Crossings (what can pass the boundary)")
    cross_matter = st.checkbox("Matter", value=True)
    cross_energy = st.checkbox("Energy", value=True)
    cross_info = st.checkbox("Information", value=True)

    st.subheader("Mechanisms (what exists inside the boundary)")
    mech_repair = st.checkbox("Self-repair / maintenance", value=True if duck_type == "Living" else False)
    mech_control = st.checkbox("Feedback control", value=True)
    mech_refuel = st.checkbox("Caretaker refuel", value=False)

    st.divider()
    st.header("Constraint knobs (the Thursday focus)")

    throughput = st.slider("Throughput (driving)", 0.0, 1.0, 0.65, 0.01)
    maintenance_cost = st.slider("Maintenance cost (overhead)", 0.0, 1.0, 0.35, 0.01)

    repair_capacity = st.slider("Repair capacity (cap)", 0.0, 1.0, 0.55, 0.01)
    actuator_strength = st.slider("Actuator strength (cap)", 0.0, 1.0, 0.50, 0.01)
    control_delay = st.slider("Control delay (steps)", 0, 10, 1, 1)

    noise = st.slider("Noise", 0.0, 0.10, 0.01, 0.005)

    st.divider()
    st.header("Disturbance (cold shock)")
    shock_time_mode = st.radio("Shock timing", ["Early", "Mid", "Late"], index=1, horizontal=True)
    shock_mag = st.slider("Shock magnitude", 0.0, 0.60, 0.15, 0.01)
    shock_width = st.slider("Shock width (steps)", 1, 30, 8, 1)

    st.divider()
    st.header("Simulation")
    steps = st.slider("Steps", 60, 300, 160, 10)

    env_temp = st.slider("Environment temperature (normalized)", 0.0, 1.0, 0.70, 0.01)
    env_coupling = st.slider("Energy coupling to environment", 0.0, 0.25, 0.08, 0.01)

    seed = st.number_input("Random seed", min_value=0, max_value=10_000, value=0, step=1)

# Perturbation preset
if shock_time_mode == "Early":
    shock_time = int(0.15 * steps)
elif shock_time_mode == "Mid":
    shock_time = int(0.45 * steps)
else:
    shock_time = int(0.75 * steps)

perturbation = {"shock_time": shock_time, "shock_mag": shock_mag, "shock_width": shock_width}

# Global parameters bundle
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

    # Mechanical duck discounts (so “mechanical control” looks like weak pseudo-sensing)
    "mechanical_control_discount": 0.25,
    "mechanical_repair_discount": 0.10,
    "mechanical_maintenance_multiplier": 1.20,

    # Noise
    "noise": noise,
}

# Default initial condition (used by Wiener & Prigogine tabs; Bertalanffy tab uses 3 ICs)
initial_default = {"E": 0.75, "I": 0.85, "C": params["core_target"]}

# Run model once for current settings
out = simulate(params, initial_default, perturbation, seed=int(seed))


# -----------------------------
# Top-row diagnostics (always visible)
# -----------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Final viability", f"{out['finalV']:.2f}")
c2.metric("Minimum viability", f"{out['minV']:.2f}")
c3.metric("Regime", out["regime"])
c4.metric("Binding constraint", out["binding"])


# -----------------------------
# Main content: 3 lenses
# -----------------------------
tab_bert, tab_prig, tab_wien = st.tabs(
    ["1) von Bertalanffy: Steady state & equifinality",
     "2) Prigogine: Dissipative regimes",
     "3) Wiener: Feedback under constraints"]
)

with tab_bert:
    st.subheader("Bertalanffy lens: does the system converge to the same steady behavior from different starts?")
    st.caption("Equifinality = same functional outcome from different initial conditions (when organization + redundancy can compensate).")

    left, right = st.columns([1.2, 1.0])

    with left:
        fig_eq, eq_present, outs = plot_equifinality_overlay(params, perturbation, seed=int(seed))
        st.pyplot(fig_eq, clear_figure=True)

    with right:
        st.markdown("### Equifinality badge")
        st.write("**Present**" if eq_present else "**Absent**")
        st.markdown("### What to look for (student-facing logic)")
        st.markdown("- If curves **converge**, the system reaches a similar steady regime despite different starts.\n"
                    "- If curves **stay separated**, a constraint (energy/repair/control) prevents convergence.")
        st.markdown("### Quick instructor knob")
        st.markdown("Try lowering **Repair capacity** or raising **Maintenance cost** to make equifinality fail.")

        # Mini-table of final outcomes (not downloadable; in-app only)
        st.markdown("### Final viability by initial condition")
        for i, o in enumerate(outs):
            st.write(f"- IC-{i+1}: final={o['finalV']:.2f}, min={o['minV']:.2f}, regime={o['regime']}")

with tab_prig:
    st.subheader("Prigogine lens: where does ordered persistence exist as a dissipative regime?")
    st.caption("A dissipative structure persists only while throughput is sufficient to pay maintenance + control + repair costs.")

    left, right = st.columns([1.05, 1.15])

    with left:
        st.pyplot(plot_timeseries(out, params, title_suffix="(current settings)"), clear_figure=True)
        st.pyplot(plot_constraint_dashboard(out, params, title_suffix="(current settings)"), clear_figure=True)

    with right:
        st.markdown("### Regime map (throughput vs maintenance)")
        st.pyplot(plot_regime_map(params, perturbation, seed=int(seed)), clear_figure=True)
        st.markdown("### How students should interpret this")
        st.markdown("- **Steady** region: the system can pay the ongoing costs → maintained pattern.\n"
                    "- **Collapse** region: driving is insufficient → gradients/organization cannot persist.\n"
                    "- **Oscillatory** region: feedback interacts with delays/constraints → overshoot and correction cycles.\n")
        st.markdown("### Teaching move")
        st.markdown("Ask: *Is collapse here about missing ‘feedback’—or about not having enough throughput to fund it?*")

with tab_wien:
    st.subheader("Wiener lens: feedback exists, but can it win under actuator limits and delays?")
    st.caption("Wiener is not ‘feedback ON/OFF’—it’s feedback under noise, delay, and saturation.")

    left, right = st.columns([1.15, 1.05])

    with left:
        st.pyplot(plot_timeseries(out, params, title_suffix="(current settings)"), clear_figure=True)

        # Control-focused plot
        fig = plt.figure(figsize=(9, 4.2))
        ax = fig.add_subplot(1, 1, 1)
        x = np.arange(params["steps"])
        ax.plot(x, out["error_trace"], label="Error (target - core)")
        ax.plot(x, out["control_effort"], label="Control effort (u)")
        if out["actuator_max"] > 0:
            ax.axhline(out["actuator_max"], linestyle="--", linewidth=1, label="Actuator cap (+)")
            ax.axhline(-out["actuator_max"], linestyle="--", linewidth=1, label="Actuator cap (-)")
        ax.set_xlabel("Step")
        ax.set_ylabel("Signal / effort")
        ax.set_title("Control under constraints: error, effort, and saturation")
        ax.legend(loc="best")
        st.pyplot(fig, clear_figure=True)

    with right:
        st.markdown("### Control status badges")
        # Simple badges driven by diagnostics we already computed
        if out["binding"] == "Actuator saturation":
            st.write("**Control status:** Saturating (effort hits cap frequently)")
        elif out["binding"] == "Control delay":
            st.write("**Control status:** Delay-driven instability (oscillation)")
        elif (params["mech_control"] and params["cross_info"]):
            st.write("**Control status:** Active (not obviously saturated)")
        else:
            st.write("**Control status:** Inactive / unavailable (no info or no control)")

        st.markdown("### Quick experiments (instructor scripts)")
        st.markdown("1) Increase **Control delay** to 6–8 and watch oscillations appear.\n"
                    "2) Lower **Actuator strength** and watch recovery fail even though ‘feedback’ is enabled.\n"
                    "3) Raise **Noise** slightly and watch regulation become brittle at the margins.\n")

        st.markdown("### Interpretation rule (for students)")
        st.markdown("A feedback explanation is **licensed** only if Information+Control are on, "
                    "but it is **effective** only if the effort can correct error fast/strong enough "
                    "without exhausting energy or saturating.")


# -----------------------------
# Optional: short, visible “model legend” (no student writing)
# -----------------------------
with st.expander("Model legend (for instructor / Q&A)"):
    st.markdown(
        "**States:** Energy reserve (E), Integrity (I), Core condition (C), Viability (V).\n\n"
        "**Cold shock:** transiently reduces environment temperature (and thus pushes core condition down if Energy crossing is ON).\n\n"
        "**Viability:** weighted composite of E, I, and closeness of C to its target.\n\n"
        "**Equifinality test (Bertalanffy):** overlay V(t) from different initial states.\n\n"
        "**Regime map (Prigogine):** classify long-run behavior as steady/oscillatory/collapse across throughput × maintenance.\n\n"
        "**Control plot (Wiener):** shows error and control effort relative to actuator caps; delay can induce oscillation."
    )
