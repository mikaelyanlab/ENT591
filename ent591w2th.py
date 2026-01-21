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
    caretaker_inside = (boundary ==_
