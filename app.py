import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="AI Virtual Doctor", layout="centered")

st.title("🩺 AI Virtual Doctor")

# =========================
# LOAD MODEL SAFELY
# =========================
@st.cache_resource
def load_model():
    return joblib.load("disease_predictor_model.pkl")

@st.cache_resource
def load_encoder():
    return joblib.load("label_encoder.pkl")

model = load_model()
encoder = load_encoder()

# =========================
# LOAD DATA (medicine lookup)
# =========================
df = pd.read_csv("training.csv")

# clean column names
df.columns = df.columns.str.strip()

# =========================
# SYMPTOM CATEGORIES
# =========================
categories = {
    "General": ["fatigue", "chills", "weight_loss", "weight_gain", "malaise"],
    "Respiratory": ["cough", "breathlessness", "phlegm", "runny_nose", "congestion"],
    "Digestive": ["stomach_pain", "vomiting", "acidity", "nausea", "abdominal_pain"],
    "Skin": ["itching", "skin_rash", "blister", "red_spots_over_body"],
    "Neurological": ["headache", "dizziness", "loss_of_balance", "unsteadiness"],
    "Urinary": ["burning_micturition", "spotting_ urination", "polyuria"],
    "Musculoskeletal": ["joint_pain", "muscle_pain", "knee_pain", "stiff_neck"],
    "Cardio": ["chest_pain", "palpitations", "fast_heart_rate"]
}

# =========================
# SESSION STATE
# =========================
if "step" not in st.session_state:
    st.session_state.step = 0
    st.session_state.symptoms = []
    st.session_state.age = None
    st.session_state.gender = None

# =========================
# RESET
# =========================
def reset():
    st.session_state.step = 0
    st.session_state.symptoms = []
    st.session_state.age = None
    st.session_state.gender = None

# =========================
# BUILD FEATURE VECTOR
# =========================
def build_vector(selected):
    features = df.columns[:-2]  # last 2 = prognosis, medicine
    vec = np.zeros(len(features))

    for s in selected:
        if s in features:
            idx = list(features).index(s)
            vec[idx] = 1

    return np.array(vec).reshape(1, -1)

# =========================
# PREDICT FUNCTION
# =========================
def predict(symptoms):
    X = build_vector(symptoms)

    pred_index = model.predict(X)[0]
    disease = encoder.inverse_transform([pred_index])[0]

    medicine = df[df["prognosis"] == disease]["medicine"].values
    medicine = medicine[0] if len(medicine) > 0 else "Not Available"

    return disease, medicine

# =========================
# UI FLOW
# =========================

# STEP 0 - AGE
if st.session_state.step == 0:
    st.subheader("What is your age?")
    age = st.number_input("Age", 1, 120)

    if st.button("Next"):
        st.session_state.age = age
        st.session_state.step = 1

# STEP 1 - GENDER
elif st.session_state.step == 1:
    st.subheader("Select Gender")
    gender = st.radio("Gender", ["Male", "Female"])

    if st.button("Next"):
        st.session_state.gender = gender
        st.session_state.step = 2

# STEP 2 - CATEGORY
elif st.session_state.step == 2:
    st.subheader("Select Symptom Category")

    category = st.selectbox("Category", list(categories.keys()))

    symptoms = st.multiselect("Select Symptoms", categories[category])

    if st.button("Next"):
        st.session_state.symptoms.extend(symptoms)
        st.session_state.step = 3

# STEP 3 - RESULT
elif st.session_state.step == 3:
    st.subheader("🩺 Diagnosis")

    if len(st.session_state.symptoms) == 0:
        st.warning("No symptoms selected")
        st.stop()

    disease, medicine = predict(st.session_state.symptoms)

    st.success(f"Predicted Disease: {disease}")
    st.info(f"Recommended Medicine: {medicine}")

    if st.button("Restart"):
        reset()
