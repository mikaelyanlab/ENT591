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
        "controlled variable (core
