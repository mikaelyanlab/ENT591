import json
from datetime import datetime
import streamlit as st

st.set_page_config(page_title="Boundary & Crossings Studio", layout="wide")

# ---------------------------
# App Title
# ---------------------------
st.title("Boundary & Crossings Studio")
st.caption("Use the app to test whether an explanation is *licensed* by your boundary choice and boundary crossings.")

# ---------------------------
# Scenario (fixed for App 1)
# ---------------------------
st.header("Scenario")
scenario = "Mechanical duck vs living duck (as systems)"
st.write(f"Selected Scenario: **{scenario}**")

# ---------------------------
# Session state
# ---------------------------
defaults = {
    "boundary": None,
    "crossings": {"Matter": False, "Energy": False, "Information": False},
    "duck_type": "Living Duck",
    "question": None,
    "claim": "",
    "mechanism": "",
    "variables": [],
    "invoked_crossings": {"Matter": False, "Energy": False, "Information": False},
    "timescale": "Minutes",
    "last_check": {"status": None, "messages": []},  # status: "pass"/"fail"
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------------------------
# Knowledge base (small + explicit)
# ---------------------------
boundaries = [
    "Duck body only",
    "Duck + immediate environment (air + local temperature)",
    "Duck + caretaker (external repair/refuel allowed)",
]

internal_vars = {
    "Mechanical Duck": [
        "spring tension / fuel reserve",
        "mechanical wear / jam probability",
        "internal temperature",
    ],
    "Living Duck": [
        "energy reserve (generalized)",
        "integrity/repair capacity",
        "controlled variable (core condition / viability index)",
    ],
}

# These are *outside* by default unless boundary explicitly includes them
external_drivers = [
    "environmental temperature",
    "disturbance events (cold shock, damage, noise)",
]

crossing_vars = {
    "Matter": ["food", "oxygen", "waste (feces/CO2/etc.)"],
    "Energy": ["free energy input (food)", "heat out"],
    "Information": ["sensed temperature", "damage signal", "state estimate / error signal"],
}

questions = [
    "Why does the duck stop functioning after a while?",
    "Why can the living duck return to ‘normal’ after cold shock?",
    "Why does the mechanical duck fail under small disturbances?",
    "What makes ‘duckness’ persist despite turnover?",
    "What changes if we move from minutes to months?",
]

# Question “requirements” as *categories*, not substring vibes
question_requirements = {
    "Why can the living duck return to ‘normal’ after cold shock?": {
        "requires_information_crossing": True,
        "requires_control_language": True,   # expects mention of sensing/compare/actuate
        "requires_defended_range": True,
    }
}

CONTROL_KEYWORDS = {"sensor", "sense", "detect", "reference", "setpoint", "range", "error", "compare", "actuator", "correct", "feedback", "control"}
DEFENDED_RANGE_KEYWORDS = {"defended range", "viable range", "within bounds", "homeostasis", "setpoint", "range"}
SLOW_KEYWORDS = {"turnover", "adapt", "learning", "evolve", "months", "years"}

# ---------------------------
# Panel B: Boundary picker
# ---------------------------
st.header("Boundary Choice")
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.session_state.boundary = st.selectbox("Choose a boundary", boundaries, index=0)
with col2:
    st.session_state.duck_type = st.radio("Duck type", ["Mechanical Duck", "Living Duck"], horizontal=True)

# Compute allowed variable universe
allowed_vars = list(internal_vars[st.session_state.duck_type])

# Boundary expansions (what becomes “inside”)
boundary_includes_environment = (st.session_state.boundary == "Duck + immediate environment (air + local temperature)")
boundary_includes_caretaker = (st.session_state.boundary == "Duck + caretaker (external repair/refuel allowed)")

if boundary_includes_environment:
    allowed_vars.append("local air temperature (inside boundary)")
if boundary_includes_caretaker:
    allowed_vars.extend(["caretaker repair action", "external refuel action"])

# ---------------------------
# Panel C: Crossings
# ---------------------------
st.header("Boundary Crossings")
st.write("Check what *crosses the boundary* for your chosen system.")

cross_desc = {
    "Matter": "Mass crosses the boundary (food, oxygen, waste).",
    "Energy": "Energy crosses as free energy/heat/work.",
    "Information": "Signals cross that causally change rates/actions (not merely ‘data’).",
}

c1, c2, c3 = st.columns(3)
for col, crossing in zip([c1, c2, c3], ["Matter", "Energy", "Information"]):
    with col:
        st.session_state.crossings[crossing] = st.checkbox(f"{crossing}", value=st.session_state.crossings[crossing])
        st.caption(cross_desc[crossing])

# Add crossing variables if enabled (as referenceable terms)
for c, vals in crossing_vars.items():
    if st.session_state.crossings[c]:
        allowed_vars.extend([f"{v} ({c} crossing)" for v in vals])

# ---------------------------
# Panel D: Variables dashboard
# ---------------------------
st.header("What you are allowed to talk about (given your boundary)")
d1, d2, d3 = st.columns([1, 1, 1], gap="large")

with d1:
    st.subheader("Internal (inside boundary)")
    for v in internal_vars[st.session_state.duck_type]:
        st.write(f"- {v}")
    if boundary_includes_environment:
        st.write("- local air temperature (inside boundary)")
    if boundary_includes_caretaker:
        st.write("- caretaker repair action")
        st.write("- external refuel action")

with d2:
    st.subheader("External drivers (outside unless boundary expanded)")
    for v in external_drivers:
        st.write(f"- {v}")

with d3:
    st.subheader("Crossing variables (only if crossing enabled)")
    any_cross = False
    for c, vals in crossing_vars.items():
        if st.session_state.crossings[c]:
            any_cross = True
            for v in vals:
                st.write(f"- {v}  **[{c}]**")
    if not any_cross:
        st.write("_None enabled_")

# Timescale
st.header("Timescale")
st.session_state.timescale = st.radio("Choose a timescale for your explanation", ["Minutes", "Months"], horizontal=True)

# ---------------------------
# Panel E: Question menu
# ---------------------------
st.header("Question")
st.session_state.question = st.selectbox("Choose a question to explain", questions, index=0)

# ---------------------------
# Panel F: Claim builder + checker
# ---------------------------
st.header("Claim Tester (licensed vs not licensed)")

st.session_state.claim = st.text_input("Claim (one sentence)", st.session_state.claim)
st.session_state.mechanism = st.text_input("Mechanism invoked (short phrase)", st.session_state.mechanism)

# Let them select variables — include externals so they can be flagged if outside
st.session_state.variables = st.multiselect(
    "Variables referenced in your explanation",
    options=sorted(set(allowed_vars + external_drivers)),
    default=st.session_state.variables,
)

st.write("Crossings you are *using as causal inputs* in the claim:")
cc1, cc2, cc3 = st.columns(3)
for col, crossing in zip([cc1, cc2, cc3], ["Matter", "Energy", "Information"]):
    with col:
        st.session_state.invoked_crossings[crossing] = st.checkbox(
            f"Invoke {crossing}",
            value=st.session_state.invoked_crossings[crossing],
            key=f"invoke_{crossing}",
        )

def run_checker():
    feedback = []

    # 1) Out-of-bound inference: referencing external drivers without including them in boundary
    for var in st.session_state.variables:
        if var == "environmental temperature" and not boundary_includes_environment:
            feedback.append("You referenced **environmental temperature**, but your boundary does not include the environment. Either expand the boundary or treat it as an external driver you can’t use as an internal mechanism.")
        if var.startswith("disturbance") and not boundary_includes_environment:
            # disturbances can still be *external* influences, but if they’re used as mechanism, we flag
            feedback.append("You referenced **disturbances**. That’s fine as an external push, but you can’t explain recovery/maintenance using disturbance itself as a mechanism unless you include relevant internal control/repair variables.")

    # 2) Crossing mismatch: invoking a crossing causally when it’s not enabled
    for c in ["Matter", "Energy", "Information"]:
        if st.session_state.invoked_crossings[c] and not st.session_state.crossings[c]:
            feedback.append(f"You invoked **{c}** causally, but **{c} crossing** is not enabled for your system.")

    # 3) Timescale mismatch (simple but effective)
    text_blob = " ".join([st.session_state.claim, st.session_state.mechanism]).lower()
    if st.session_state.timescale == "Minutes":
        if any(k in text_blob for k in SLOW_KEYWORDS):
            feedback.append("You invoked slow processes (turnover/adaptation/evolution) while using a **minutes** timescale. Either switch to months or rewrite with fast variables.")
    if st.session_state.timescale == "Months":
        # not an error, but nudge them to include slow variables if they chose months
        pass

    # 4) Question-specific requirements (kept strict)
    req = question_requirements.get(st.session_state.question)
    if req:
        if req.get("requires_information_crossing") and not st.session_state.crossings["Information"]:
            feedback.append("This question (return to normal after cold shock) typically requires **Information crossing** (sensing/state estimation). Enable Information or expand boundary to include an external controller.")
        if req.get("requires_control_language"):
            if not any(k in text_blob for k in CONTROL_KEYWORDS):
                feedback.append("Your explanation doesn’t yet contain control-loop language (sensor/reference/error/actuator/feedback/correction). Add a control mechanism or revise the boundary to include an external controller.")
        if req.get("requires_defended_range"):
            if not any(k in text_blob for k in DEFENDED_RANGE_KEYWORDS):
                feedback.append("Your explanation doesn’t yet invoke a **defended range/viable bounds/homeostasis** idea. Add it explicitly or revise your mechanism.")

    if feedback:
        st.session_state.last_check = {"status": "fail", "messages": feedback}
    else:
        st.session_state.last_check = {"status": "pass", "messages": ["Claim is licensed under your boundary + crossings."]}

if st.button("Check claim"):
    run_checker()

# Display last check results
if st.session_state.last_check["status"] == "fail":
    st.error("Flags:\n- " + "\n- ".join(st.session_state.last_check["messages"]))
elif st.session_state.last_check["status"] == "pass":
    st.success("\n".join(st.session_state.last_check["messages"]))

# ---------------------------
# Worksheet + Export (no student writing stored in-app)
# ---------------------------
st.header("Student Worksheet + Export")

worksheet_text = """Boundary & Crossings Studio — Worksheet (submit on paper or as a photo/PDF)

1) Boundary choice:
- Which boundary did you choose, and why?

2) Crossings:
- Which crossings (Matter/Energy/Information) did you enable?
- Give one concrete example of each enabled crossing.

3) Two licensed explanations:
- Write two explanations that the app accepts (licensed).
- For each, list: variables referenced + which crossing(s) you used.

4) One failed explanation (the learning moment):
- Write one explanation the app rejected.
- Copy the app’s main flag.
- What did you change (boundary or crossings) to make it licensed?

5) Reflection:
- What did you learn about why “open system” is not the same as “alive”?
"""

st.download_button(
    "Download worksheet (TXT)",
    data=worksheet_text,
    file_name="boundary_crossings_worksheet.txt",
    mime="text/plain",
)

# Export current run summary for receipts (optional)
run_summary = {
    "timestamp": datetime.now().isoformat(timespec="seconds"),
    "scenario": scenario,
    "duck_type": st.session_state.duck_type,
    "boundary": st.session_state.boundary,
    "crossings_enabled": st.session_state.crossings,
    "timescale": st.session_state.timescale,
    "question": st.session_state.question,
    "claim": st.session_state.claim,
    "mechanism": st.session_state.mechanism,
    "variables_selected": st.session_state.variables,
    "invoked_crossings": st.session_state.invoked_crossings,
    "last_check": st.session_state.last_check,
}

st.download_button(
    "Export run summary (JSON)",
    data=json.dumps(run_summary, indent=2),
    file_name="boundary_crossings_run_summary.json",
    mime="application/json",
)

# Instructor knobs (non-persistent; just convenience)
with st.expander("Instructor knobs"):
    st.write("If you want strict/no-hints modes later, wire these into the checker and UI.")
    st.checkbox("Strict mode (future)", value=True, disabled=True)
    st.checkbox("Show hints (future)", value=False, disabled=True)
