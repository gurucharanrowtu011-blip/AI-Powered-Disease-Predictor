import streamlit as st
import numpy as np
import pandas as pd
import joblib

# =========================
# LOAD MODEL + ENCODER
# =========================
model = joblib.load("disease_predictor_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# OPTIONAL: dataset for medicine lookup
df = pd.read_csv("training.csv")

# =========================
# SYMPTOM CATEGORIES
# =========================
symptom_groups = {
    "General": [
        "itching", "skin_rash", "fatigue", "lethargy",
        "fever", "chills", "weight_loss", "weight_gain"
    ],
    "Respiratory": [
        "cough", "breathlessness", "chest_pain",
        "phlegm", "throat_irritation", "runny_nose"
    ],
    "Digestive": [
        "stomach_pain", "acidity", "vomiting",
        "nausea", "abdominal_pain", "diarrhoea", "constipation"
    ],
    "Skin": [
        "skin_rash", "itching", "blister",
        "red_spots_over_body", "skin_peeling"
    ],
    "Neurological": [
        "headache", "dizziness", "loss_of_balance",
        "unsteadiness", "slurred_speech"
    ]
}

# flatten all symptoms (for vector creation)
all_symptoms = sorted(set([s for group in symptom_groups.values() for s in group]))

# =========================
# TITLE
# =========================
st.title("🩺 AI Virtual Doctor")

st.write("Answer a few questions and I will predict your disease.")

# =========================
# INPUT FORM
# =========================
with st.form("doctor_form"):
    age = st.number_input("Age", 1, 120, 25)
    gender = st.selectbox("Gender", ["Male", "Female"])

    selected_groups = st.multiselect(
        "Select Symptom Categories",
        list(symptom_groups.keys())
    )

    symptoms_selected = []

    for group in selected_groups:
        st.subheader(f"{group} Symptoms")
        symptoms_selected += st.multiselect(
            f"Select {group} symptoms",
            symptom_groups[group],
            key=group
        )

    submitted = st.form_submit_button("Predict Disease")

# =========================
# PREDICTION FUNCTION
# =========================
def predict(symptoms):
    input_data = [0] * len(all_symptoms)

    for s in symptoms:
        if s in all_symptoms:
            idx = all_symptoms.index(s)
            input_data[idx] = 1

    input_array = np.array([input_data])

    pred = model.predict(input_array)
    disease = label_encoder.inverse_transform(pred)[0]

    return disease

# =========================
# MEDICINE LOOKUP
# =========================
def get_medicine(disease):
    row = df[df["prognosis"] == disease]
    if len(row) > 0:
        return row["medicine"].values[0]
    return "Not available"

# =========================
# RESULT
# =========================
if submitted:
    disease = predict(symptoms_selected)
    medicine = get_medicine(disease)

    st.success(f"🧠 Predicted Disease: {disease}")

    st.info(f"💊 Suggested Medicine: {medicine}")

    st.warning("⚠️ This is an AI prediction, consult a doctor for confirmation.")
