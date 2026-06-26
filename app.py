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
st.write("Select your symptoms to get disease prediction instantly.")

# =========================
# SAFE FILE LOADING (FIX EOF ERROR ISSUES)
# =========================
BASE_DIR = os.path.dirname(__file__)

model_path = os.path.join(BASE_DIR, "disease_predictor_model.pkl")
encoder_path = os.path.join(BASE_DIR, "label_encoder.pkl")
symptom_path = os.path.join(BASE_DIR, "symptom_dict.pkl")

if not os.path.exists(model_path):
    st.error("❌ Model file not found. Please upload disease_predictor_model.pkl")
    st.stop()

if not os.path.exists(encoder_path):
    st.error("❌ Encoder file not found.")
    st.stop()

if not os.path.exists(symptom_path):
    st.error("❌ Symptom dictionary not found.")
    st.stop()

model = joblib.load(model_path)
encoder = joblib.load(encoder_path)
symptom_dict = joblib.load(symptom_path)

# =========================
# SYMPTOM CATEGORIES (UI ONLY)
# =========================
categories = {
    "General": ["itching", "fatigue", "high_fever", "chills", "weight_loss"],
    "Respiratory": ["cough", "breathlessness", "chest_pain", "sore_throat"],
    "Digestive": ["nausea", "vomiting", "abdominal_pain", "acidity"],
    "Skin": ["skin_rash", "blister", "itching"],
    "Neurological": ["headache", "dizziness", "loss_of_balance"],
    "Urinary": ["burning_micturition", "urination_frequent"]
}

# =========================
# INPUT UI (FAST FORM)
# =========================
with st.form("predict_form"):

    age = st.number_input("Age", 0, 120, 25)
    gender = st.radio("Gender", ["Male", "Female"])

    category = st.selectbox("Choose Symptom Category", list(categories.keys()))
    symptoms = st.multiselect("Select Symptoms", categories[category])

    submit = st.form_submit_button("🔍 Predict Disease")

# =========================
# MEDICINE MAP (OPTIONAL)
# =========================
medicine_map = {
    "Fungal infection": "Antifungal cream, Fluconazole",
    "Common Cold": "Paracetamol, Rest, Fluids",
    "Dengue": "Paracetamol, Hydration",
    "Malaria": "Consult doctor for antimalarials",
    "Pneumonia": "Antibiotics prescribed by doctor"
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

    prediction = model.predict([input_vector])[0]
    disease = encoder.inverse_transform([prediction])[0]

    try:
        confidence = np.max(model.predict_proba([input_vector])) * 100
    except:
        confidence = None

    medicine = medicine_map.get(disease, "Consult a doctor for proper treatment")

    # =========================
    # OUTPUT
    # =========================
    st.success("Prediction Complete")

    st.markdown("## 🩺 Disease Identified")
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
