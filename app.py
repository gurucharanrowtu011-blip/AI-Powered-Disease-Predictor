import streamlit as st
import numpy as np
import pandas as pd
import joblib

# =========================
# LOAD FILES
# =========================
model = joblib.load("disease_predictor_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")
feature_columns = joblib.load("features.pkl")

df = pd.read_csv("training.csv")

# =========================
# UI TITLE
# =========================
st.title("🩺 AI Virtual Doctor")
st.write("Select symptoms and get disease prediction")

# =========================
# SYMPTOM CATEGORIES (UI ONLY)
# =========================
symptom_groups = {
    "General": ["itching", "skin_rash", "fatigue", "fever", "chills", "weight_loss"],
    "Respiratory": ["cough", "breathlessness", "chest_pain", "phlegm"],
    "Digestive": ["stomach_pain", "acidity", "vomiting", "nausea", "diarrhoea"],
    "Skin": ["skin_rash", "itching", "blister"],
    "Neurological": ["headache", "dizziness", "loss_of_balance"]
}

# =========================
# FORM UI
# =========================
with st.form("doctor_form"):
    age = st.number_input("Age", 1, 120, 25)
    gender = st.selectbox("Gender", ["Male", "Female"])

    selected_symptoms = []

    for cat in symptom_groups:
        selected_symptoms += st.multiselect(cat, symptom_groups[cat], key=cat)

    submit = st.form_submit_button("Predict Disease")

# =========================
# PREDICTION FUNCTION
# =========================
def predict(symptoms):
    input_data = np.zeros(len(feature_columns))

    for s in symptoms:
        if s in feature_columns:
            idx = list(feature_columns).index(s)
            input_data[idx] = 1

    pred = model.predict([input_data])[0]
    disease = label_encoder.inverse_transform([pred])[0]

    return disease

# =========================
# MEDICINE + INFO
# =========================
def get_info(disease):
    row = df[df["prognosis"] == disease]

    if len(row) > 0:
        medicine = row["medicine"].values[0]
    else:
        medicine = "Not available"

    return medicine

# =========================
# OUTPUT
# =========================
if submit:
    if len(selected_symptoms) == 0:
        st.error("Select at least one symptom")
    else:
        disease = predict(selected_symptoms)
        medicine = get_info(disease)

        st.success(f"🧠 Disease: {disease}")
        st.info(f"💊 Medicine: {medicine}")
        st.warning("⚠️ This is an AI prediction, consult a doctor")
