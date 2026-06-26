import streamlit as st
import numpy as np
import joblib
import os

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AI Health Assistant",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 AI Health Assistant")
st.write("Select your symptom category and symptoms")

# =========================
# SAFE LOAD MODELS
# =========================
BASE_DIR = os.path.dirname(__file__)

model = joblib.load(os.path.join(BASE_DIR, "disease_predictor_model.pkl"))
encoder = joblib.load(os.path.join(BASE_DIR, "label_encoder.pkl"))
symptom_dict = joblib.load(os.path.join(BASE_DIR, "symptom_dict.pkl"))

# =========================
# SYMPTOM GROUPING (CLEAN UI)
# =========================
symptom_categories = {
    "General": [
        "high_fever", "chills", "fatigue", "weakness_in_limbs", "malaise"
    ],
    "Respiratory": [
        "cough", "breathlessness", "chest_pain", "sore_throat", "runny_nose"
    ],
    "Skin": [
        "itching", "skin_rash", "blister", "red_spots_over_body"
    ],
    "Digestive": [
        "stomach_pain", "vomiting", "nausea", "acidity", "abdominal_pain"
    ],
    "Neurological": [
        "headache", "dizziness", "loss_of_balance", "unsteadiness"
    ],
    "Urinary": [
        "burning_micturition", "urination_frequent"
    ]
}

# =========================
# UI FORM
# =========================
with st.form("form"):

    age = st.number_input("Age", 0, 120, 25)
    gender = st.radio("Gender", ["Male", "Female"])

    category = st.selectbox("Choose Symptom Category", list(symptom_categories.keys()))
    symptoms = st.multiselect("Select Symptoms", symptom_categories[category])

    submit = st.form_submit_button("Predict Disease")

# =========================
# MEDICINE MAP (SIMPLE)
# =========================
medicine_map = {
    "Common Cold": "Paracetamol, Rest, Fluids",
    "Dengue": "Paracetamol, Hydration",
    "Malaria": "Antimalarial drugs (consult doctor)",
    "Fungal infection": "Antifungal cream, Fluconazole"
}

# =========================
# PREDICTION
# =========================
if submit:

    if len(symptoms) == 0:
        st.warning("Please select at least one symptom")
        st.stop()

    # create 134 feature vector
    input_vector = np.zeros(len(symptom_dict))

    for s in symptoms:
        if s in symptom_dict:
            input_vector[symptom_dict[s]] = 1

    pred = model.predict([input_vector])[0]
    disease = encoder.inverse_transform([pred])[0]

    try:
        confidence = np.max(model.predict_proba([input_vector])) * 100
    except:
        confidence = None

    medicine = medicine_map.get(disease, "Consult doctor for proper treatment")

    # =========================
    # OUTPUT
    # =========================
    st.success("Prediction Complete")

    st.markdown("## 🩺 Disease")
    st.markdown(f"### {disease}")

    if confidence:
        st.write(f"Confidence: **{confidence:.2f}%**")

    st.markdown("## 💊 Medicine")
    st.write(medicine)

    st.markdown("## 🛡️ Precautions")
    st.write("""
    - Take rest  
    - Drink plenty of water  
    - Maintain hygiene  
    - Avoid self-medication  
    - Consult doctor if symptoms persist  
    """)
