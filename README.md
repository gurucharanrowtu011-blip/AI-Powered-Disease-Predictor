🩺 Multiple Disease Predictor 

A web application that predicts possible diseases based on user-selected symptoms using a trained Machine Learning model.

Built using Python, Streamlit, and Random Forest Classifier.

🚀 Live Features
🧠 Predicts disease based on symptoms
📊 Shows prediction confidence score
👤 Takes patient details (Age, Gender)
🧾 Organized symptom categories for easy selection
⚡ Fast ML inference using trained model
📦 Clean Streamlit web interface
🧪 Machine Learning Model
Algorithm: Random Forest Classifier
Training Dataset: 134 symptom features
Output Classes: 42 diseases
Accuracy: ~99%
Label Encoding used for disease classification
🧾 Project Structure
📁 project-root
 ├── app.py
 ├── disease_predictor_model.pkl
 ├── label_encoder.pkl
 ├── features.pkl
 ├── training.csv
 ├── requirements.txt
 └── README.md
⚙️ How It Works
User selects symptoms
Symptoms are converted into binary feature vector
Model predicts disease
Output is decoded using LabelEncoder
Confidence score is displayed
🧑‍💻 Installation & Run
1. Clone repository
git clone https://github.com/gurucharanrowtu011-blip/AI-Powered-Disease-Predictor/edit/main/README.md
cd disease-predictor
2. Install dependencies
pip install -r requirements.txt
3. Run Streamlit app
streamlit run app.py
📦 Requirements
streamlit
numpy
pandas
scikit-learn
joblib
📊 Model Details
Input: 134 symptoms (binary encoding)
Output: Disease prediction (42 classes)
Training split: 80/20
Model type: Ensemble (Random Forest)
🎯 Key Highlights
Clean UI with categorized symptoms
Real-time prediction
High accuracy ML model
Easy to extend with more symptoms or diseases
📌 Future Improvements
🏥 Add medicine recommendation system
📈 Show top-3 disease probabilities graph
💾 Downloadable medical report (PDF)
🌐 Deploy on cloud (Streamlit Cloud / Render)
⚠️ Disclaimer

This project is for educational purposes only and should not be used as a real medical diagnostic tool.

👨‍💻 Author

Built as a Machine Learning project for academic and learning purposes.
