import streamlit as st

# App Title
st.title("Boundary & Crossings Studio")

# Panel A: Scenario Selector
st.header("Panel A: Scenario Selector")
scenario = "Mechanical duck vs living duck (as systems)"
st.write(f"Selected Scenario: {scenario}")

# Session state for selections
if 'boundary' not in st.session_state:
    st.session_state.boundary = None
if 'crossings' not in st.session_state:
    st.session_state.crossings = {'Matter': False, 'Energy': False, 'Information': False}
if 'duck_type' not in st.session_state:
    st.session_state.duck_type = "Living Duck"
if 'question' not in st.session_state:
    st.session_state.question = None
if 'claim' not in st.session_state:
    st.session_state.claim = ""
if 'mechanism' not in st.session_state:
    st.session_state.mechanism = ""
if 'variables' not in st.session_state:
    st.session_state.variables = []
if 'invoked_crossings' not in st.session_state:
    st.session_state.invoked_crossings = {'Matter': False, 'Energy': False, 'Information': False}
if 'timescale' not in st.session_state:
    st.session_state.timescale = "Minutes"

# Panel B: Boundary Picker
st.header("Panel B: Boundary Picker")
boundaries = [
    "Duck body only",
    "Duck + immediate environment (air + local temperature)",
    "Duck + caretaker (external repair/refuel allowed)"
]
st.session_state.boundary = st.selectbox("Choose Boundary", boundaries)

# Duck Type Selector
st.session_state.duck_type = st.radio("Duck Type", ["Mechanical Duck", "Living Duck"])

# Define allowed variables based on boundary and duck type
internal_vars = {
    "Mechanical Duck": ["spring tension / fuel reserve", "mechanical wear / jam probability", "internal temperature"],
    "Living Duck": ["energy reserve (generalized)", "integrity/repair capacity", "controlled variable (core condition, “viability index”)"],
}
external_drivers = ["environmental temperature", "disturbances"]
crossing_vars = {
    'Matter': ["food/oxygen/waste"],
    'Energy': ["heat/work/free energy"],
    'Information': ["sensed temperature / damage signal"]
}

allowed_vars = internal_vars[st.session_state.duck_type].copy()
if st.session_state.boundary == "Duck + immediate environment (air + local temperature)":
    allowed_vars.append("local air temperature")
if st.session_state.boundary == "Duck + caretaker (external repair/refuel allowed)":
    allowed_vars.extend(["caretaker repair", "external refuel"])

# Panel C: Crossings Checklist
st.header("Panel C: Crossings Checklist")
for crossing in ['Matter', 'Energy', 'Information']:
    st.session_state.crossings[crossing] = st.checkbox(
        f"{crossing}: { 'mass crosses the boundary (food, oxygen, waste).' if crossing == 'Matter' else 'energy crosses as heat/work/free energy.' if crossing == 'Energy' else 'signals crossing that causally change rates/actions.' }",
        value=st.session_state.crossings[crossing]
    )

# Enable crossing vars if checked
for c, vals in crossing_vars.items():
    if st.session_state.crossings[c]:
        allowed_vars.extend(vals)

# Panel D: State Variables
st.header("Panel D: State Variables I am allowed to talk about")
st.subheader("Internal State Variables")
for v in internal_vars[st.session_state.duck_type]:
    st.write(f"- {v}")
st.subheader("External Drivers")
for v in external_drivers:
    st.write(f"- {v}")
st.subheader("Crossing Variables")
for c, vals in crossing_vars.items():
    if st.session_state.crossings[c]:
        for v in vals:
            st.write(f"- {v} ({c})")

# Timescale Slider (Instructor Knob: Can be hidden)
st.session_state.timescale = st.radio("Timescale", ["Minutes", "Months"])

# Panel E: Question Menu
st.header("Panel E: Question Menu")
questions = [
    "Why does the duck stop functioning after a while?",
    "Why can the living duck return to ‘normal’ after cold shock?",
    "Why does the mechanical duck fail under small disturbances?",
    "What makes ‘duckness’ persist despite turnover?",
    "What changes if we move from minutes to months?"
]
st.session_state.question = st.selectbox("Choose Question", questions)

# Panel F: Claim Builder + Inference Checker
st.header("Panel F: Claim Builder + Inference Checker")
st.session_state.claim = st.text_input("Claim (one sentence)", st.session_state.claim)
st.session_state.mechanism = st.text_input("Mechanism Invoked", st.session_state.mechanism)
st.session_state.variables = st.multiselect("Variables Referenced", allowed_vars + external_drivers, st.session_state.variables)

for crossing in ['Matter', 'Energy', 'Information']:
    st.session_state.invoked_crossings[crossing] = st.checkbox(f"Invoke {crossing} in Claim", value=st.session_state.invoked_crossings[crossing])

# Checker Logic
if st.button("Check Claim"):
    feedback = []
    # Out-of-bound: Check if variables are allowed
    for var in st.session_state.variables:
        if var in external_drivers and var not in allowed_vars:
            feedback.append(f"This explanation requires adding {var} inside the boundary or enabling a crossing.")
    
    # Crossing Mismatch
    for c in ['Matter', 'Energy', 'Information']:
        if st.session_state.invoked_crossings[c] and not st.session_state.crossings[c]:
            feedback.append(f"Using “{c.lower()}” as causal input when {c} crossing wasn’t enabled.")
    
    # Timescale Mismatch (simple example)
    if st.session_state.timescale == "Minutes" and "turnover" in st.session_state.claim.lower():
        feedback.append("Invoking slow variables (turnover, adaptation) on a short timescale.")
    
    # Question-specific requirements (simplified)
    q_reqs = {
        questions[1]: ['information', 'control action', 'defended range'],  # Cold shock
    }
    if st.session_state.question in q_reqs:
        for req in q_reqs[st.session_state.question]:
            if req not in ''.join(st.session_state.variables + [st.session_state.mechanism]).lower():
                feedback.append(f"This explanation requires {req} variable/category.")
    
    if feedback:
        st.error("Flags:\n" + "\n".join(feedback))
    else:
        st.success("Claim is licensed.")

# Deliverables Section
st.header("Deliverables")
st.write("1. Boundary choice + justification")
justification = st.text_area("Justification (2–3 sentences)")
st.write("2. Crossings declared: " + ", ".join([c for c in st.session_state.crossings if st.session_state.crossings[c]]))
st.write("3. Two licensed claims: (Paste your passing claims here)")
claim1 = st.text_input("Licensed Claim 1")
claim2 = st.text_input("Licensed Claim 2")
st.write("4. One failed claim with reason + correction")
failed_claim = st.text_input("Failed Claim")
reason = st.text_input("App's Reason")
correction = st.text_input("Correction")

# Instructor Knobs
with st.expander("Instructor Knobs"):
    strict_mode = st.checkbox("Strict Mode (forces explicit variable selection)", value=True)
    hints = st.checkbox("Show Hint Text", value=False)
    if hints:
        st.write("Hints: Remember to include all necessary variables and crossings.")
