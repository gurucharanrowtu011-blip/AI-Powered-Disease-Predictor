import streamlit as st
import numpy as np
import joblib

# =========================
# LOAD MODEL
# =========================
model = joblib.load("disease_predictor_model.pkl")
encoder = joblib.load("label_encoder.pkl")
symptom_dict = joblib.load("symptom_dict.pkl")

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AI Health Assistant",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 AI Health Assistant")
st.write("Select symptoms and get instant disease prediction")

# =========================
# SYMPTOM CATEGORIES (UI ONLY)
# =========================
category_map = {
    "General": ["itching", "fatigue", "weight_loss", "high_fever", "chills"],
    "Respiratory": ["cough", "breathlessness", "chest_pain", "sore_throat"],
    "Digestive": ["nausea", "vomiting", "abdominal_pain", "acidity"],
    "Skin": ["skin_rash", "itching", "blister"],
    "Neurological": ["headache", "dizziness", "loss_of_balance"],
    "Urinary": ["burning_micturition", "urination_frequent"]
}

categories = list(category_map.keys())

# =========================
# FORM UI (FAST + CLEAN)
# =========================
with st.form("prediction_form"):

    st.subheader("Patient Information")

    age = st.number_input("Age", 0, 120, 25)
    gender = st.radio("Gender", ["Male", "Female"])

    st.subheader("Step 1: Choose Category")

    selected_category = st.selectbox(
        "Symptom Category",
        categories
    )

    st.subheader("Step 2: Select Symptoms")

    available_symptoms = category_map[selected_category]

    selected_symptoms = st.multiselect(
        "Symptoms",
        available_symptoms
    )

    submit = st.form_submit_button("🔍 Predict Disease")

# =========================
# PREDICTION
# =========================
if submit:

    if len(selected_symptoms) == 0:
        st.warning("Please select at least one symptom")
        st.stop()

    # Build 134 feature vector
    features = np.zeros(len(symptom_dict))

    for sym in selected_symptoms:
        if sym in symptom_dict:
            features[symptom_dict[sym]] = 1

    # Prediction
    pred = model.predict([features])[0]
    disease = encoder.inverse_transform([pred])[0]

    # Confidence (optional)
    try:
        prob = model.predict_proba([features])[0]
        confidence = np.max(prob) * 100
    except:
        confidence = None

    # =========================
    # MEDICINE + ADVICE (STATIC SAFE VERSION)
    # =========================
    medicine_map = {
        "Fungal infection": "Antifungal cream, Fluconazole",
        "Common Cold": "Paracetamol, Rest, Fluids",
        "Dengue": "Paracetamol, Hydration",
        "Malaria": "Antimalarial drugs (consult doctor)",
        "Pneumonia": "Antibiotics (doctor prescribed)"
    }

    medicine = medicine_map.get(disease, "Consult a doctor for proper medication")

    # =========================
    # OUTPUT
    # =========================
    st.success("🩺 Prediction Complete")

    st.subheader("Predicted Disease")
    st.markdown(f"### {disease}")

    if confidence:
        st.write(f"Confidence: **{confidence:.2f}%**")

    st.subheader("💊 Medicine")
    st.write(medicine)

    st.subheader("🛡️ Precautions")
    st.write("""
    - Take rest  
    - Drink plenty of fluids  
    - Maintain hygiene  
    - Avoid self-medication  
    - Consult doctor if symptoms persist  
    """)
