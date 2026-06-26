import streamlit as st
import numpy as np
import pandas as pd
import joblib

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="AI Virtual Doctor", layout="centered")

st.title("🩺 AI Virtual Doctor")

# =========================
# LOAD MODEL + ENCODER
# =========================
model = joblib.load("disease_predictor_model.pkl")
encoder = joblib.load("label_encoder.pkl")

# =========================
# LOAD DATASET (for feature mapping + medicine)
# =========================
df = pd.read_csv("training.csv")
df.columns = df.columns.str.strip()

features = df.columns[:-2]  # last 2 = prognosis + medicine

# =========================
# SYMPTOM CATEGORIES (CLEAN)
# =========================
categories = {
    "General": ["fatigue", "lethargy", "malaise", "restlessness", "anxiety", "depression", "loss_of_appetite"],
    "Fever": ["high_fever", "mild_fever", "chills", "shivering", "sweating"],
    "Respiratory": ["cough", "breathlessness", "chest_pain", "runny_nose", "congestion", "phlegm"],
    "Digestive": ["stomach_pain", "abdominal_pain", "nausea", "vomiting", "diarrhoea", "acidity"],
    "Neurological": ["headache", "dizziness", "loss_of_balance", "unsteadiness", "altered_sensorium"],
    "Skin": ["itching", "skin_rash", "blister", "red_spots_over_body", "pus_filled_pimples"],
    "Urinary": ["burning_micturition", "polyuria", "dark_urine", "foul_smell_of urine"],
    "Cardio": ["chest_pain", "palpitations", "fast_heart_rate"],
    "Musculoskeletal": ["joint_pain", "back_pain", "neck_pain", "muscle_pain"],
    "Metabolic": ["obesity", "weight_gain", "weight_loss", "increased_appetite"]
}

# =========================
# BUILD INPUT VECTOR
# =========================
def build_input(selected_symptoms):
    vector = np.zeros(len(features))

    for symptom in selected_symptoms:
        if symptom in features:
            idx = list(features).index(symptom)
            vector[idx] = 1

    return vector.reshape(1, -1)

# =========================
# PREDICT FUNCTION
# =========================
def predict(symptoms):
    X = build_input(symptoms)

    pred_index = model.predict(X)[0]
    disease = encoder.inverse_transform([pred_index])[0]

    medicine = df[df["prognosis"] == disease]["medicine"].values
    medicine = medicine[0] if len(medicine) > 0 else "Not Available"

    return disease, medicine

# =========================
# UI (SINGLE PAGE)
# =========================

st.subheader("Select your symptoms")

category = st.selectbox("Choose category", list(categories.keys()))
symptoms_selected = st.multiselect("Select symptoms", categories[category])

# Store all selections
if "all_symptoms" not in st.session_state:
    st.session_state.all_symptoms = []

if st.button("➕ Add Symptoms"):
    st.session_state.all_symptoms.extend(symptoms_selected)
    st.success("Symptoms added!")

st.write("### Selected Symptoms:")
st.write(st.session_state.all_symptoms)

# =========================
# PREDICTION
# =========================
if st.button("🩺 Predict Disease"):

    if len(st.session_state.all_symptoms) == 0:
        st.error("Please select symptoms first")
    else:
        disease, medicine = predict(st.session_state.all_symptoms)

        st.success(f"🧾 Disease: {disease}")
        st.info(f"💊 Medicine: {medicine}")

# =========================
# RESET
# =========================
if st.button("🔄 Reset"):
    st.session_state.all_symptoms = []
