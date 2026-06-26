import streamlit as st
import numpy as np
import joblib

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="AI Virtual Doctor", layout="wide")
st.title("🩺 AI Virtual Doctor")

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    model = joblib.load("disease_predictor_model.pkl")
    return model

model = load_model()

# =========================
# SYMPTOM LIST (MUST MATCH TRAINING COLUMNS)
# =========================
symptoms = list(model.feature_names_in_)

# =========================
# CATEGORY UI
# =========================
categories = {
    "🔥 General": ["fatigue","weight_gain","weight_loss","loss_of_appetite","dehydration","anxiety","depression"],
    "🤒 Fever/Infection": ["high_fever","mild_fever","chills","shivering","sweating"],
    "🫁 Respiratory": ["cough","phlegm","breathlessness","chest_pain","runny_nose","congestion"],
    "🍽️ Digestive": ["stomach_pain","vomiting","nausea","diarrhoea","acidity","indigestion"],
    "🧠 Neurological": ["headache","dizziness","loss_of_balance","unsteadiness"],
    "🧴 Skin": ["itching","skin_rash","nodal_skin_eruptions"],
    "💧 Urinary": ["burning_micturition","dark_urine"],
    "❤️ Cardio": ["chest_pain","fast_heart_rate"],
    "🦴 Musculoskeletal": ["joint_pain","back_pain","neck_pain","muscle_pain"],
    "🧬 Metabolic": ["obesity","yellowish_skin","yellowing_of_eyes"]
}

# =========================
# INPUT UI
# =========================
st.subheader("Select Symptoms")

selected_symptoms = []

for cat, syms in categories.items():
    with st.expander(cat):
        for i, s in enumerate(syms):
            if s in symptoms:
                if st.checkbox(s, key=f"{cat}_{s}_{i}"):
                    selected_symptoms.append(s)

# =========================
# PREDICTION FUNCTION
# =========================
def predict(symptoms_selected):

    input_vector = np.zeros(len(symptoms))

    for s in symptoms_selected:
        if s in symptoms:
            idx = symptoms.index(s)
            input_vector[idx] = 1

    # prediction
    prediction = model.predict([input_vector])[0]

    # probability
    proba = model.predict_proba([input_vector])[0]

    return prediction, proba

# =========================
# BUTTON
# =========================
if st.button("🩺 Predict Disease"):

    if len(selected_symptoms) == 0:
        st.warning("Please select at least one symptom")

    else:
        disease, proba = predict(selected_symptoms)

        st.success(f"🧾 Disease: {disease}")

        # 🔥 TOP 3 PROBABLE DISEASES
        top_n = 3
        top_indices = np.argsort(proba)[::-1][:top_n]

        st.subheader("📊 Prediction Confidence")

        for i in top_indices:
            st.write(f"{model.classes_[i]} → {round(proba[i]*100, 2)}%")
