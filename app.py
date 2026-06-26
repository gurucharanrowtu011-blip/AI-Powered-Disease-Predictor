import streamlit as st
import numpy as np
import joblib

# =========================
# LOAD MODELS
# =========================
model = joblib.load("disease_predictor_model.pkl")
encoder = joblib.load("label_encoder.pkl")
symptom_dict = joblib.load("symptom_dict.pkl")

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AI Powered Disease Predictor",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 AI Powered Disease Predictor")
st.write("Select your symptoms and get instant prediction")

# =========================
# SYMPTOM LIST (FIXED)
# =========================
symptoms_list = sorted(symptom_dict.keys())

# =========================
# FORM UI (FAST - NO NEXT BUTTON LAG)
# =========================
with st.form("disease_form"):

    st.subheader("Enter your details")

    age = st.number_input("Age", 0, 120, 25)
    gender = st.radio("Gender", ["Male", "Female"])

    st.subheader("Select your symptoms")

    selected_symptoms = st.multiselect(
        "Choose symptoms you have",
        symptoms_list
    )

    submitted = st.form_submit_button("🔍 Predict Disease")

# =========================
# PREDICTION LOGIC
# =========================
if submitted:

    if len(selected_symptoms) == 0:
        st.warning("Please select at least one symptom")
    else:

        # Build 134-feature vector
        features = np.zeros(len(symptom_dict))

        for sym in selected_symptoms:
            if sym in symptom_dict:
                features[symptom_dict[sym]] = 1

        # Predict
        pred = model.predict([features])[0]
        disease = encoder.inverse_transform([pred])[0]

        # Confidence (if supported)
        try:
            probs = model.predict_proba([features])[0]
            confidence = np.max(probs) * 100
        except:
            confidence = None

        # =========================
        # OUTPUT
        # =========================
        st.success("🩺 Prediction Complete")

        st.subheader("Predicted Disease")
        st.markdown(f"### {disease}")

        if confidence:
            st.subheader("Confidence")
            st.write(f"{confidence:.2f}%")

        st.subheader("🛡️ Advice")
        st.write("• Maintain hygiene")
        st.write("• Drink plenty of water")
        st.write("• Take rest")
        st.write("• Consult doctor if symptoms persist")
