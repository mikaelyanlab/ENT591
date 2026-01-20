import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

st.set_page_config(page_title="Boundary & Crossings Studio", layout="wide")

# -----------------------------
# Helpers
# -----------------------------
def simulate_duck(
    duck_type: str,
    T: int,
    dt: float,
    boundary: str,
    crossings: dict,
    mechanisms: dict,
    env_temp: float,
    cold_shock_time: int,
    cold_shock_mag: float,
    noise: float,
):
    """
    Toy simulation meant to be *qualitatively* diagnostic:
    - Mechanical duck winds down; can only be kept going if caretaker/refuel is allowed inside boundary.
    - Living duck can maintain viability if it has throughput + maintenance and/or control.
    """

    t = np.arange(0, T, dt)
    n = len(t)

    # State variables (dimensionless 0..1-ish)
    E = np.zeros(n)  # energy reserve
    I = np.zeros(n)  # integrity (wear/repair)
    C = np.zeros(n)  # core condition (proxy for defended variable)

    # Initialize
    if duck_type == "Mechanical Duck":
        E[0] = 1.0
        I[0] = 1.0
        C[0] = 0.85
    else:
        E[0] = 0.7
        I[0] = 0.85
        C[0] = 0.85

    # Parameters (chosen for stable qualitative behavior)
    # Energy dynamics
    base_metabolic_use = 0.010 if duck_type == "Living Duck" else 0.012
    activity_cost = 0.006 if duck_type == "Living Duck" else 0.007

    # Throughput (if matter+energy crossings enabled)
    intake_rate = 0.020 if duck_type == "Living Duck" else 0.0  # mechanical has no endogenous "intake"
    # External caretaker refuel (only if boundary allows caretaker mechanism)
    caretaker_refuel_rate = 0.035

    # Wear and repair
    wear_rate = 0.010 if duck_type == "Living Duck" else 0.014
    repair_rate = 0.020  # max possible, paid with energy

    # Control loop
    setpoint = 0.85
    control_gain = 0.9
    control_cost = 0.010

    # Environment coupling
    # If Energy crossing enabled, C is pulled toward env_temp (scaled), else weaker coupling
    env_pull = 0.020 if crossings["Energy"] else 0.006

    # Normalize env_temp into [0,1] "comfort" scale for this toy model
    # 22C ~ 0.85, colder ~ lower
    env_norm = np.clip(0.85 + 0.02 * (env_temp - 22.0), 0.0, 1.0)

    # Disturbance schedule (cold shock)
    shock = np.zeros(n)
    if 0 <= cold_shock_time < n:
        shock[int(cold_shock_time)] = cold_shock_mag

    for k in range(1, n):
        # noise/disturbance
        eps = np.random.normal(0, noise)

        # -----------------------
        # INPUTS across boundary
        # -----------------------
        # Throughput intake requires Matter crossing (food/oxygen) and (implicitly) Energy usefulness.
        # We gate it with Matter crossing; Energy crossing mainly matters via env coupling + costs.
        intake = 0.0
        if duck_type == "Living Duck" and crossings["Matter"]:
            intake = intake_rate * (1.0 - E[k - 1])  # saturating as E fills

        # Caretaker refuel only if included in boundary AND mechanism turned on
        caretaker_refuel = 0.0
        if mechanisms["caretaker_refuel"] and boundary == "Duck + caretaker":
            caretaker_refuel = caretaker_refuel_rate * (1.0 - E[k - 1])

        # -----------------------
        # MAINTENANCE / REPAIR
        # -----------------------
        repair = 0.0
        if mechanisms["repair"]:
            # repair is limited by energy; it increases integrity, costs energy
            repair = repair_rate * min(E[k - 1], 1.0)

        # -----------------------
        # CYBERNETIC CONTROL
        # -----------------------
        control = 0.0
        if mechanisms["control"]:
            # control requires Information crossing OR caretaker inside boundary (external controller)
            if crossings["Information"] or boundary == "Duck + caretaker":
                error = setpoint - C[k - 1]
                control = control_gain * error
            else:
                control = 0.0

        # -----------------------
        # ENERGY UPDATE
        # -----------------------
        # Activity depends on integrity and core condition (if the "duck" is broken/cold, it "struggles")
        activity = np.clip(0.3 + 0.7 * I[k - 1] * C[k - 1], 0.0, 1.0)

        E_use = base_metabolic_use + activity_cost * activity
        E_costs = 0.0
        if mechanisms["repair"]:
            E_costs += 0.5 * repair  # repair consumes energy
        if mechanisms["control"] and (crossings["Information"] or boundary == "Duck + caretaker"):
            E_costs += control_cost * abs(control)

        dE = intake + caretaker_refuel - (E_use + E_costs)
        E[k] = np.clip(E[k - 1] + dE + eps * 0.002, 0.0, 1.2)

        # -----------------------
        # INTEGRITY UPDATE
        # -----------------------
        # wear rises with activity and disturbances; repair counters it if enabled
        dI = -(wear_rate * activity) - shock[k] * 0.30 + (0.8 * repair)
        I[k] = np.clip(I[k - 1] + dI + eps * 0.002, 0.0, 1.0)

        # -----------------------
        # CORE CONDITION UPDATE
        # -----------------------
        # pulled toward environment if Energy crossing; control pushes toward setpoint (if available)
        dC = -env_pull * (C[k - 1] - env_norm) - shock[k] * 0.45 + (0.15 * control)
        C[k] = np.clip(C[k - 1] + dC + eps * 0.004, 0.0, 1.0)

    # "Viability" as a banded function (students infer defended range concept from the band)
    # Penalize low energy, low integrity, and core drifting away from setpoint.
    V = np.clip(
        1.0
        - 1.1 * np.maximum(0, 0.25 - E)
        - 1.3 * np.maximum(0, 0.45 - I)
        - 0.9 * np.abs(C - setpoint),
        0.0,
        1.0,
    )

    return t, E, I, C, V, setpoint, env_norm


def build_causal_graph(boundary, crossings, mechanisms, duck_type):
    """
    Build a directed causal graph that updates with boundary/crossings/mechanisms.
    Students learn: boundary choices and crossings *license* causal edges.
    """
    G = nx.DiGraph()

    # Core nodes
    nodes_internal = ["Energy reserve (E)", "Integrity (I)", "Core condition (C)", "Viability (V)"]
    for n in nodes_internal:
        G.add_node(n, kind="internal")

    # External nodes
    G.add_node("Environment", kind="external")
    G.add_node("Disturbance", kind="external")

    # Crossings as interface nodes (conceptual)
    if crossings["Matter"]:
        G.add_node("Matter in/out", kind="crossing")
    if crossings["Energy"]:
        G.add_node("Energy exchange", kind="crossing")
    if crossings["Information"]:
        G.add_node("Information signals", kind="crossing")

    # Caretaker node only if boundary includes it
    if boundary == "Duck + caretaker":
        G.add_node("Caretaker", kind="boundary_included")

    # Baseline internal couplings
    G.add_edge("Energy reserve (E)", "Integrity (I)", label="powers repair")
    G.add_edge("Energy reserve (E)", "Core condition (C)", label="powers regulation")
    G.add_edge("Integrity (I)", "Viability (V)", label="enables function")
    G.add_edge("Core condition (C)", "Viability (V)", label="within bounds")
    G.add_edge("Energy reserve (E)", "Viability (V)", label="avoids depletion")

    # Disturbance always hits internal variables (as an external driver)
    G.add_edge("Disturbance", "Integrity (I)", label="damage/wear")
    G.add_edge("Disturbance", "Core condition (C)", label="shock")

    # Environment coupling depends on Energy crossing (strong) vs weak background (still shown)
    if crossings["Energy"]:
        G.add_edge("Environment", "Energy exchange", label="heat/work")
        G.add_edge("Energy exchange", "Core condition (C)", label="pull toward env")
    else:
        # show weak coupling as dashed-style by label hint
        G.add_edge("Environment", "Core condition (C)", label="weak coupling (no energy crossing)")

    # Matter crossing enables intake affecting energy reserve
    if crossings["Matter"]:
        G.add_edge("Matter in/out", "Energy reserve (E)", label="intake supports E")

    # Repair mechanism
    if mechanisms["repair"]:
        G.add_edge("Energy reserve (E)", "Integrity (I)", label="repair (costly)")

    # Control mechanism requires info OR caretaker inside boundary
    if mechanisms["control"]:
        if crossings["Information"]:
            G.add_edge("Information signals", "Core condition (C)", label="feedback control")
        elif boundary == "Duck + caretaker":
            G.add_edge("Caretaker", "Core condition (C)", label="external control")
        else:
            # control selected but not licensed; graph shows "missing link"
            G.add_node("⚠ missing sensing", kind="warning")
            G.add_edge("⚠ missing sensing", "Core condition (C)", label="control not licensed")

    # Caretaker refuel
    if mechanisms["caretaker_refuel"]:
        if boundary == "Duck + caretaker":
            G.add_edge("Caretaker", "Energy reserve (E)", label="refuel")
        else:
            G.add_node("⚠ caretaker outside", kind="warning")
            G.add_edge("⚠ caretaker outside", "Energy reserve (E)", label="refuel not licensed")

    return G


def draw_graph(G):
    # Node color by kind (no seaborn; matplotlib only)
    kind_to_color = {
        "internal": None,
        "external": None,
        "crossing": None,
        "boundary_included": None,
        "warning": None,
    }

    pos = nx.spring_layout(G, seed=7, k=0.85)

    fig = plt.figure(figsize=(8.8, 5.6))
    ax = plt.gca()
    ax.set_axis_off()

    # Draw nodes by kind to allow different alpha/edge emphasis without explicit colors
    for kind in ["external", "crossing", "boundary_included", "internal", "warning"]:
        nodelist = [n for n, d in G.nodes(data=True) if d.get("kind") == kind]
        if not nodelist:
            continue
        nx.draw_networkx_nodes(G, pos, nodelist=nodelist, node_size=1050, alpha=0.9, ax=ax)

    nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=18, width=1.4, alpha=0.9, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=9, ax=ax)

    # Edge labels (kept short)
    edge_labels = {(u, v): d.get("label", "") for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, ax=ax, rotate=False)

    plt.tight_layout()
    return fig


# -----------------------------
# Sidebar controls (all interaction)
# -----------------------------
st.sidebar.header("Controls (no typing)")

duck_type = st.sidebar.radio("Duck", ["Mechanical Duck", "Living Duck"], index=1)

boundary = st.sidebar.selectbox(
    "Boundary",
    ["Duck only", "Duck + environment", "Duck + caretaker"],
    index=0
)
boundary_map = {
    "Duck only": "Duck body only",
    "Duck + environment": "Duck + immediate environment",
    "Duck + caretaker": "Duck + caretaker"
}

st.sidebar.subheader("Crossings (what crosses the boundary)")
crossings = {
    "Matter": st.sidebar.checkbox("Matter", value=True),
    "Energy": st.sidebar.checkbox("Energy", value=True),
    "Information": st.sidebar.checkbox("Information", value=(duck_type == "Living Duck")),
}

st.sidebar.subheader("Mechanisms (what exists *inside* the boundary)")
# These are model toggles; boundary/crossings will determine whether they actually have effect.
mechanisms = {
    "repair": st.sidebar.checkbox("Self-repair / maintenance", value=(duck_type == "Living Duck")),
    "control": st.sidebar.checkbox("Feedback control", value=(duck_type == "Living Duck")),
    "caretaker_refuel": st.sidebar.checkbox("Caretaker refuel", value=False),
}

st.sidebar.subheader("Disturbance + environment")
env_temp = st.sidebar.slider("Environment temperature (°C)", min_value=0.0, max_value=35.0, value=22.0, step=0.5)
cold_shock_time = st.sidebar.slider("Cold shock time (step)", 0, 120, 40, 1)
cold_shock_mag = st.sidebar.slider("Cold shock magnitude", 0.0, 1.0, 0.35, 0.05)
noise = st.sidebar.slider("Noise", 0.0, 0.15, 0.04, 0.01)

st.sidebar.subheader("Time")
T = st.sidebar.slider("Simulation length (steps)", 60, 240, 120, 10)

# -----------------------------
# Main layout
# -----------------------------
st.title("App 1 — Boundary & Crossings Studio")
st.caption("Flip boundary/crossing/mechanism switches and infer concepts from **graphs**, not text answers.")

left, right = st.columns([1.05, 1.0], gap="large")

# -----------------------------
# Build causal graph
# -----------------------------
boundary_full = boundary_map[boundary]
G = build_causal_graph(boundary_full, crossings, mechanisms, duck_type)

with left:
    st.subheader("A) Causal structure implied by your modeling choices")
    fig_g = draw_graph(G)
    st.pyplot(fig_g, clear_figure=True)

    # Quick, non-graded in-app diagnostics (still a display, not a worksheet)
    st.subheader("B) What changed when you toggled settings?")
    # This is intentionally descriptive and short; the inference happens from the graph + plots.
    msgs = []
    if mechanisms["control"] and not (crossings["Information"] or boundary_full == "Duck + caretaker"):
        msgs.append("Feedback control is selected but **no sensing route exists** (no Information crossing and no caretaker inside boundary).")
    if mechanisms["caretaker_refuel"] and boundary_full != "Duck + caretaker":
        msgs.append("Caretaker refuel is selected but the caretaker is **outside the boundary**.")
    if msgs:
        st.warning(" \n\n".join(msgs))
    else:
        st.info("Use the causal graph to justify which explanations are licensed by your boundary + crossings.")

# -----------------------------
# Simulate and plot
# -----------------------------
t, E, I, C, V, setpoint, env_norm = simulate_duck(
    duck_type=duck_type,
    T=T,
    dt=1.0,
    boundary=boundary_full,
    crossings=crossings,
    mechanisms=mechanisms,
    env_temp=env_temp,
    cold_shock_time=cold_shock_time,
    cold_shock_mag=cold_shock_mag,
    noise=noise,
)

with right:
    st.subheader("C) Observable behavior over time (infer concepts here)")

    fig = plt.figure(figsize=(8.8, 5.6))
    ax = plt.gca()

    # Plot viability and internal states (no explicit colors set)
    ax.plot(t, V, linewidth=2.2, label="Viability (V)")
    ax.plot(t, E, linewidth=1.6, label="Energy reserve (E)")
    ax.plot(t, I, linewidth=1.6, label="Integrity (I)")
    ax.plot(t, C, linewidth=1.6, label="Core condition (C)")

    # Defended band for "viability" (students infer defended range idea)
    ax.axhspan(0.70, 1.00, alpha=0.12, label="Viable band (0.70–1.00)")
    ax.axvline(cold_shock_time, linestyle="--", linewidth=1.2, label="Cold shock")

    ax.set_ylim(-0.05, 1.25)
    ax.set_xlabel("Time (steps)")
    ax.set_ylabel("State (scaled)")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()

    st.pyplot(fig, clear_figure=True)

    st.subheader("D) Minimal readout")
    st.metric("Final viability", f"{V[-1]:.2f}")
    st.metric("Min viability", f"{V.min():.2f}")
    st.caption(
        "Try: (1) make mechanical duck survive shocks without caretaker, "
        "(2) turn off Information and see whether ‘control’ still works, "
        "(3) move caretaker inside boundary and see which edges and trajectories change."
    )

