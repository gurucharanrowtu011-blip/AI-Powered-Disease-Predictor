import streamlit as st
import numpy as np
import pandas as pd
import joblib

model = joblib.load("disease_predictor_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")
feature_columns = joblib.load("features.pkl")

df = pd.read_csv("training.csv")

symptom_groups = {
    "General": ["itching", "skin_rash", "fatigue", "fever", "chills"],
    "Respiratory": ["cough", "breathlessness", "chest_pain"],
    "Digestive": ["stomach_pain", "acidity", "vomiting"],
    "Skin": ["skin_rash", "itching", "blister"],
    "Neurological": ["headache", "dizziness"]
}

st.title("🩺 AI Virtual Doctor")

with st.form("form"):
    age = st.number_input("Age", 1, 120, 25)
    gender = st.selectbox("Gender", ["Male", "Female"])

    selected = []
    for g in symptom_groups:
        selected += st.multiselect(g, symptom_groups[g], key=g)

    submit = st.form_submit_button("Predict")

def predict(symptoms):
    input_data = np.zeros(len(feature_columns))

    for s in symptoms:
        if s in feature_columns:
            input_data[list(feature_columns).index(s)] = 1

    pred = model.predict([input_data])[0]

    # 🔥 FIX HERE
    disease = label_encoder.inverse_transform([pred])[0]

    return disease

if submit:
    disease = predict(selected)
    st.success("🧠 Disease: " + disease)
