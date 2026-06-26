import streamlit as st
import numpy as np
import joblib

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="AI Virtual Doctor", layout="wide")
st.title("🩺 AI Virtual Doctor")

# =========================
# LOAD MODEL + ENCODER
# =========================
@st.cache_resource
def load_model():
    model = joblib.load("disease_predictor_model.pkl")
    encoder = joblib.load("label_encoder.pkl")
    return model, encoder

model, encoder = load_model()

# =========================
# SYMPTOM LIST (MUST MATCH TRAINING ORDER)
# =========================
symptoms = [
    'itching','skin_rash','nodal_skin_eruptions','continuous_sneezing','shivering','chills',
    'joint_pain','stomach_pain','acidity','ulcers_on_tongue','vomiting','fatigue',
    'burning_micturition','spotting_ urination','weight_gain','weight_loss','anxiety',
    'cough','high_fever','headache','nausea','loss_of_appetite','back_pain',
    'constipation','abdominal_pain','diarrhoea','mild_fever','yellowing_of_eyes',
    'breathlessness','sweating','dehydration','indigestion','chest_pain',
    'dizziness','loss_of_balance','unsteadiness','blurred_and_distorted_vision',
    'phlegm','throat_irritation','runny_nose','congestion','fast_heart_rate',
    'weakness_in_limbs','irritability','depression','muscle_pain','obesity',
    'swelling_joints','stiff_neck','dark_urine','yellowish_skin'
]

# =========================
# CATEGORY UI (EXPANDED)
# =========================
categories = {
    "🔥 General": ["fatigue","weight_gain","weight_loss","loss_of_appetite","dehydration","weakness_in_limbs","anxiety","depression","irritability"],
    "🤒 Fever / Infection": ["high_fever","mild_fever","chills","shivering","sweating"],
    "🫁 Respiratory": ["cough","phlegm","breathlessness","chest_pain","runny_nose","congestion","throat_irritation"],
    "🍽️ Digestive": ["stomach_pain","vomiting","nausea","diarrhoea","acidity","indigestion","abdominal_pain","constipation"],
    "🧠 Neurological": ["headache","dizziness","loss_of_balance","unsteadiness","blurred_and_distorted_vision"],
    "🧴 Skin": ["itching","skin_rash","nodal_skin_eruptions"],
    "💧 Urinary": ["burning_micturition","spotting_ urination","dark_urine"],
    "❤️ Cardio": ["chest_pain","fast_heart_rate"],
    "🦴 Musculoskeletal": ["joint_pain","back_pain","neck_pain","muscle_pain","swelling_joints","stiff_neck"],
    "🧬 Metabolic": ["obesity","yellowing_of_eyes","yellowish_skin"]
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
# PREDICTION FUNCTION (FIXED)
# =========================
def predict(symptoms_selected):

    input_vector = np.zeros(model.n_features_in_)

    for s in symptoms_selected:
        if s in symptoms:
            idx = symptoms.index(s)
            if idx < len(input_vector):
                input_vector[idx] = 1

    pred_index = model.predict([input_vector])[0]

    # ✅ FIX: correct decoding
    try:
        return encoder.inverse_transform([pred_index])[0]
    except:
        return encoder.classes_[pred_index]

# =========================
# BUTTON
# =========================
if st.button("🩺 Predict Disease"):

    if len(selected_symptoms) == 0:
        st.warning("Select at least one symptom")
    else:
        disease = predict(selected_symptoms)
        st.success(f"🧾 Disease: {disease}")
