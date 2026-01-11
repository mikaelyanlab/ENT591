# vaucanson_duck_app.py
# Streamlit app: a simple “functional Vaucanson duck” stock–flow model
# Shows stocks, flows, and modifiers; sliders update both plots and a live diagram.

import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Vaucanson Duck: Stocks, Flows, Modifiers", layout="wide")

# -----------------------------
# Model
# -----------------------------
def simulate(
    T=200.0,
    dt=0.5,
    E0=80.0,         # Stored energy (mechanical battery)
    G0=40.0,         # Gut contents (seed+water mixture)
    W0=0.0,          # Wear
    work_in=0.6,     # Work input rate (energy/time)
    seed_in=0.35,    # Seed+water input rate (mass/time)
    P_max=2.2,       # Max deliverable power (power limit of mechanism)
    alloc_proc=0.45, # Fraction of available power allocated to processing
    k_proc=0.045,    # Processing constant (how strongly power+substrate drive conversion)
    Kp=0.35,         # Power half-saturation for processing (smooth gating)
    wear_sens=0.015, # Wear accumulation per unit power
    eff_slope=0.08,  # Efficiency loss per wear (modifier)
    heat_base=0.35,  # Baseline fraction of power lost as heat
    heat_wear=0.35,  # Additional heat fraction per unit wear (modifier)
    waste_leak=0.01, # Passive gut leak to waste (small)
):
    n = int(T / dt) + 1
    t = np.linspace(0, T, n)

    E = np.zeros(n)  # Stored energy
    G = np.zeros(n)  # Gut contents
    W = np.zeros(n)  # Wear
    paste_out = np.zeros(n)  # cumulative paste/waste out (from processing)
    waste_out = np.zeros(n)  # instantaneous waste out rate (processing + leak)

    # Diagnostics (flows and modifiers)
    P_avail = np.zeros(n)
    P_move = np.zeros(n)
    P_proc = np.zeros(n)
    F_proc = np.zeros(n)     # processing flow (G -> paste/waste)
    Eff = np.zeros(n)
    Heat_frac = np.zeros(n)
    Motion_out = np.zeros(n) # "useful motion" per time
    Heat_out = np.zeros(n)   # heat per time
    Wear_rate = np.zeros(n)

    E[0], G[0], W[0] = E0, G0, W0

    for i in range(1, n):
        # ---- Modifiers (functions of wear) ----
        # Efficiency: reduces how much of the mechanism's potential power is actually deliverable as "useful work"
        eff = 1.0 / (1.0 + eff_slope * W[i - 1])
        eff = max(0.05, min(1.0, eff))  # keep it bounded for stability in a teaching demo

        # Heat fraction: increases with wear (more friction -> more energy lost as heat)
        heat_frac = heat_base + heat_wear * (W[i - 1] / (1.0 + W[i - 1]))
        heat_frac = max(0.05, min(0.95, heat_frac))

        # ---- Energy availability and allocation ----
        # Potential power is bounded by P_max and reduced by efficiency.
        # Also cannot exceed what's available in stored energy over dt.
        # (Power * dt is energy spent this step.)
        P_potential = P_max * eff
        P_energy_limited = E[i - 1] / dt
        P_av = min(P_potential, P_energy_limited)

        # Allocate available power between motion/display vs processing
        a = max(0.0, min(1.0, alloc_proc))
        Pp = a * P_av
        Pm = (1.0 - a) * P_av

        # ---- Gut processing flow (matter conversion) ----
        # Processing rate depends on substrate (G) and on delivered power to processing (Pp).
        # Smooth gating of power using Michaelis-Menten-like term: Pp/(Pp+Kp).
        power_gate = 0.0 if (Pp + Kp) <= 0 else (Pp / (Pp + Kp))
        f_proc = k_proc * G[i - 1] * power_gate  # mass/time
        # Also ensure we can't process more than is present in G over dt:
        f_proc = min(f_proc, G[i - 1] / dt)

        # Passive leak (not power-driven)
        leak = min(waste_leak * G[i - 1], G[i - 1] / dt)

        # Total waste out rate (instantaneous)
        w_out = f_proc + leak

        # ---- Wear accumulation ----
        wear_rate = wear_sens * P_av  # wear/time
        # ---- Update stocks ----
        dE = (work_in - P_av) * dt
        dG = (seed_in - w_out) * dt
        dW = wear_rate * dt

        E[i] = max(0.0, E[i - 1] + dE)
        G[i] = max(0.0, G[i - 1] + dG)
        W[i] = max(0.0, W[i - 1] + dW)

        paste_out[i] = paste_out[i - 1] + f_proc * dt
        waste_out[i] = w_out

        # ---- Outputs (diagnostic) ----
        # Split motion power into useful motion vs heat loss
        motion_useful = (1.0 - heat_frac) * Pm
        heat_from_motion = heat_frac * Pm

        # Processing power is also mostly dissipated as heat + mechanical losses; we count it as heat out
        heat_from_processing = Pp

        Motion_out[i] = motion_useful
        Heat_out[i] = heat_from_motion + heat_from_processing
        Heat_frac[i] = heat_frac
        Eff[i] = eff
        P_avail[i] = P_av
        P_move[i] = Pm
        P_proc[i] = Pp
        F_proc[i] = f_proc
        Wear_rate[i] = wear_rate

    df = pd.DataFrame(
        {
            "t": t,
            "E": E,
            "G": G,
            "W": W,
            "P_avail": P_avail,
            "P_move": P_move,
            "P_proc": P_proc,
            "F_proc": F_proc,
            "Waste_out_rate": waste_out,
            "Paste_cum": paste_out,
            "Eff": Eff,
            "Heat_frac": Heat_frac,
            "Motion_out": Motion_out,
            "Heat_out": Heat_out,
            "Wear_rate": Wear_rate,
        }
    )
    return df


# -----------------------------
# Diagram builder (Graphviz DOT)
# -----------------------------
def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def penwidth_from(x, x_ref, min_w=1.0, max_w=8.0):
    # x_ref is a typical scale (avoid exploding thickness)
    if x_ref <= 0:
        return min_w
    r = clamp(x / x_ref, 0.0, 2.0)
    return min_w + (max_w - min_w) * (r / 2.0)

def build_dot(snapshot, refs):
    """
    snapshot: dict-like row at a chosen time
    refs: dict of reference magnitudes for scaling penwidth
    """
    E = float(snapshot["E"])
    G = float(snapshot["G"])
    W = float(snapshot["W"])
    P_av = float(snapshot["P_avail"])
    Pm = float(snapshot["P_move"])
    Pp = float(snapshot["P_proc"])
    fproc = float(snapshot["F_proc"])
    win = float(snapshot["Work_in"])
    sin = float(snapshot["Seed_in"])
    eff = float(snapshot["Eff"])
    hfrac = float(snapshot["Heat_frac"])
    wear_rate = float(snapshot["Wear_rate"])
    wout = float(snapshot["Waste_out_rate"])

    # Thickness scaling
    pw_work = penwidth_from(win, refs["work"])
    pw_seed = penwidth_from(sin, refs["seed"])
    pw_pmove = penwidth_from(Pm, refs["power"])
    pw_pproc = penwidth_from(Pp, refs["power"])
    pw_fproc = penwidth_from(fproc, refs["proc"])
    pw_wout = penwidth_from(wout, refs["proc"] + refs["leak"])
    pw_wear = penwidth_from(wear_rate, refs["wear"])

    # DOT graph
    dot = f"""
digraph Duck {{
  rankdir=LR;
  graph [fontsize=12, labelloc="t", label="Vaucanson Duck (Functional): Stocks, Flows, Modifiers"];
  node  [shape=box, fontsize=11];
  edge  [fontsize=10];

  // Stocks
  E [label="Stored Energy [E]\\n{E:0.2f}"];
  G [label="Gut Contents [G]\\n{G:0.2f}"];
  W [label="Wear [W]\\n{W:0.2f}"];

  // Sources/Sinks
  WorkIn [shape=ellipse, label="Work input\\n{win:0.2f}/t"];
  SeedIn [shape=ellipse, label="Seed+Water in\\n{sin:0.2f}/t"];
  Heat [shape=ellipse, label="Heat out"];
  Motion [shape=ellipse, label="Motion out"];
  Waste [shape=ellipse, label="Paste/Waste out"];

  // Processing operator
  Proc [shape=box, label="PROCESSOR\\n(rate depends on G and P_proc)"];

  // Flows
  WorkIn -> E [label="charging", penwidth={pw_work:0.2f}];
  E -> Motion [label="P_move={Pm:0.2f}", penwidth={pw_pmove:0.2f}];
  E -> Proc   [label="P_proc={Pp:0.2f}", penwidth={pw_pproc:0.2f}];
  E -> Heat   [label="dissipation\\n(total P_avail={P_av:0.2f})", style=dotted, penwidth=1.2];

  SeedIn -> G [label="inflow", penwidth={pw_seed:0.2f}];
  G -> Proc   [label="substrate", penwidth=2.0];
  Proc -> Waste [label="F_proc={fproc:0.2f}", penwidth={pw_fproc:0.2f}];
  G -> Waste    [label="leak", style=dashed, penwidth=1.2];

  Waste -> Waste [label="Waste_out={wout:0.2f}", style=invis]; // label anchor

  // Wear accumulation (stock change)
  E -> W [label="wear_rate={wear_rate:0.3f}", penwidth={pw_wear:0.2f}];

  // Modifiers (causal influences)
  node [shape=note, fontsize=10];
  EffNote [label="Modifier: Efficiency\\nEff={eff:0.2f}\\n(reduces deliverable power)"];
  HeatNote [label="Modifier: Heat fraction\\nheat_frac={hfrac:0.2f}\\n(more wear → more heat)"];

  W -> EffNote [style=dashed, arrowhead=normal];
  W -> HeatNote [style=dashed, arrowhead=normal];
  EffNote -> E [style=dashed, arrowhead=normal, label="limits P_max"];
  HeatNote -> Heat [style=dashed, arrowhead=normal, label="more heat\\nless motion"];

}}
"""
    return dot


# -----------------------------
# UI
# -----------------------------
st.title("Vaucanson Duck — A Simple Stock–Flow Model (with Modifiers)")

with st.sidebar:
    st.header("Controls")

    st.subheader("Simulation")
    T = st.slider("Time horizon (T)", 50.0, 500.0, 200.0, 10.0)
    dt = st.select_slider("Time step (dt)", options=[0.1, 0.2, 0.5, 1.0, 2.0], value=0.5)

    st.subheader("Initial stocks")
    E0 = st.slider("Stored Energy E0", 0.0, 200.0, 80.0, 1.0)
    G0 = st.slider("Gut Contents G0", 0.0, 200.0, 40.0, 1.0)
    W0 = st.slider("Wear W0", 0.0, 10.0, 0.0, 0.05)

    st.subheader("Inputs")
    work_in = st.slider("Work input rate (charging)", 0.0, 5.0, 0.6, 0.05)
    seed_in = st.slider("Seed+Water input rate", 0.0, 5.0, 0.35, 0.05)

    st.subheader("Power & allocation")
    P_max = st.slider("Max power capacity (P_max)", 0.2, 10.0, 2.2, 0.1)
    alloc_proc = st.slider("Fraction of power to processing", 0.0, 1.0, 0.45, 0.01)

    st.subheader("Processing dynamics")
    k_proc = st.slider("Processing constant (k_proc)", 0.0, 0.2, 0.045, 0.005)
    Kp = st.slider("Processing power half-sat (Kp)", 0.01, 2.0, 0.35, 0.01)
    waste_leak = st.slider("Passive gut leak", 0.0, 0.1, 0.01, 0.005)

    st.subheader("Wear & modifiers")
    wear_sens = st.slider("Wear sensitivity", 0.0, 0.1, 0.015, 0.001)
    eff_slope = st.slider("Efficiency loss per wear", 0.0, 0.5, 0.08, 0.01)
    heat_base = st.slider("Baseline heat fraction", 0.05, 0.95, 0.35, 0.01)
    heat_wear = st.slider("Extra heat from wear", 0.0, 1.0, 0.35, 0.01)

# Run sim
df = simulate(
    T=T, dt=dt,
    E0=E0, G0=G0, W0=W0,
    work_in=work_in, seed_in=seed_in,
    P_max=P_max, alloc_proc=alloc_proc,
    k_proc=k_proc, Kp=Kp,
    wear_sens=wear_sens, eff_slope=eff_slope,
    heat_base=heat_base, heat_wear=heat_wear,
    waste_leak=waste_leak
)

# Add inputs into df for diagram labels
df["Work_in"] = work_in
df["Seed_in"] = seed_in
df["Leak_ref"] = waste_leak * df["G"].max() if df["G"].max() > 0 else 0.0

# Choose a time point to "inspect" (drives the diagram labels)
inspect_t = st.slider("Inspect time (diagram snapshot)", 0.0, float(df["t"].max()), float(df["t"].max() * 0.35), float(dt))
snap = df.iloc[(df["t"] - inspect_t).abs().argsort().iloc[0]]

# Reference magnitudes for penwidth scaling (keep stable across slider changes)
refs = {
    "work": max(0.2, work_in),
    "seed": max(0.2, seed_in),
    "power": max(0.2, P_max),
    "proc": max(0.2, (df["F_proc"].quantile(0.9) if df["F_proc"].max() > 0 else 0.3)),
    "leak": max(0.05, waste_leak * max(1.0, df["G"].quantile(0.9))),
    "wear": max(0.01, (df["Wear_rate"].quantile(0.9) if df["Wear_rate"].max() > 0 else 0.02)),
}

# Layout
left, right = st.columns([1.1, 1.0], gap="large")

with left:
    st.subheader("Live stock–flow diagram (updates with sliders + snapshot time)")
    dot = build_dot(snap, refs)
    st.graphviz_chart(dot, use_container_width=True)

    st.caption(
        "Interpretation: [E] is the shared energy stock. Some power goes to motion, some to processing. "
        "[G] is substrate that becomes paste/waste via the PROCESSOR. Wear accumulates with total power use, "
        "reducing efficiency and increasing heat losses."
    )

with right:
    st.subheader("Snapshot values (at selected time)")
    cols = st.columns(3)
    cols[0].metric("Stored Energy E", f"{snap['E']:.2f}")
    cols[1].metric("Gut Contents G", f"{snap['G']:.2f}")
    cols[2].metric("Wear W", f"{snap['W']:.2f}")

    cols = st.columns(3)
    cols[0].metric("Power available", f"{snap['P_avail']:.2f}")
    cols[1].metric("Power to motion", f"{snap['P_move']:.2f}")
    cols[2].metric("Power to processing", f"{snap['P_proc']:.2f}")

    cols = st.columns(3)
    cols[0].metric("Processing flow F_proc", f"{snap['F_proc']:.3f}")
    cols[1].metric("Efficiency (modifier)", f"{snap['Eff']:.2f}")
    cols[2].metric("Heat fraction (modifier)", f"{snap['Heat_frac']:.2f}")

# Plots
st.subheader("Time series (stocks, flows, modifiers)")

tab1, tab2, tab3 = st.tabs(["Stocks", "Flows/Outputs", "Modifiers"])

with tab1:
    fig = plt.figure()
    plt.plot(df["t"], df["E"], label="Stored Energy E")
    plt.plot(df["t"], df["G"], label="Gut Contents G")
    plt.plot(df["t"], df["W"], label="Wear W")
    plt.xlabel("time")
    plt.ylabel("stock level")
    plt.legend()
    st.pyplot(fig)

with tab2:
    fig = plt.figure()
    plt.plot(df["t"], df["P_move"], label="P_move (to motion)")
    plt.plot(df["t"], df["P_proc"], label="P_proc (to processing)")
    plt.plot(df["t"], df["F_proc"], label="F_proc (processing flow)")
    plt.plot(df["t"], df["Waste_out_rate"], label="Waste out rate")
    plt.xlabel("time")
    plt.ylabel("rate / power")
    plt.legend()
    st.pyplot(fig)

    fig = plt.figure()
    plt.plot(df["t"], df["Motion_out"], label="Motion_out (useful)")
    plt.plot(df["t"], df["Heat_out"], label="Heat_out (total)")
    plt.xlabel("time")
    plt.ylabel("output rate")
    plt.legend()
    st.pyplot(fig)

with tab3:
    fig = plt.figure()
    plt.plot(df["t"], df["Eff"], label="Efficiency modifier")
    plt.plot(df["t"], df["Heat_frac"], label="Heat fraction modifier")
    plt.plot(df["t"], df["Wear_rate"], label="Wear rate")
    plt.xlabel("time")
    plt.ylabel("modifier / rate")
    plt.legend()
    st.pyplot(fig)

st.subheader("Data table (optional)")
st.dataframe(df, use_container_width=True)
