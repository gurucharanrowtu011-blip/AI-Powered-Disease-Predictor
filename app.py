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

st.write("Select a symptom category and describe your condition")

# =========================
# CATEGORY SYSTEM (IMPORTANT)
# =========================
category_map = {
    "General": ["itching", "fatigue", "weight_loss", "chills", "high_fever"],
    "Respiratory": ["cough", "breathlessness", "chest_pain", "sore_throat"],
    "Digestive": ["nausea", "vomiting", "abdominal_pain", "acidity"],
    "Skin": ["skin_rash", "itching", "blister"],
    "Neurological": ["headache", "dizziness", "loss_of_balance"],
    "Urinary": ["burning_micturition", "urination_frequent"]
}

categories = list(category_map.keys())

# =========================
# FORM UI (FAST)
# =========================
with st.form("symptom_form"):

    st.subheader("Step 1: Basic Info")
    age = st.number_input("Age", 0, 120, 25)
    gender = st.radio("Gender", ["Male", "Female"])

    st.subheader("Step 2: Select Symptom Category")

    selected_category = st.selectbox(
        "Choose category",
        categories
    )

    st.subheader("Step 3: Select Symptoms")

    symptoms = category_map[selected_category]

    selected_symptoms = st.multiselect(
        "Choose symptoms you have",
        symptoms
    )

    submit = st.form_submit_button("🔍 Predict Disease")

# =========================
# PREDICTION LOGIC
# =========================
if submit:

    # build feature vector (134)
    features = np.zeros(len(symptom_dict))

    for sym in selected_symptoms:
        if sym in symptom_dict:
            features[symptom_dict[sym]] = 1

    # prediction
    pred = model.predict([features])[0]
    disease = encoder.inverse_transform([pred])[0]

    # confidence (optional)
    try:
        prob = model.predict_proba([features])[0]
        confidence = np.max(prob) * 100
    except:
        confidence = None

    # =========================
    # OUTPUT
    # =========================
    st.success("🩺 Prediction Completed")

    st.subheader("Predicted Disease")
    st.markdown(f"### {disease}")

    if confidence:
        st.write(f"Confidence: **{confidence:.2f}%**")

    st.subheader("🛡️ Basic Advice")
    st.write("""
    - Take rest  
    - Stay hydrated  
    - Maintain hygiene  
    - Consult doctor if symptoms worsen  
    """)
