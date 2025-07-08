import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.datasets import make_classification

# Sidebar Navigation
with st.sidebar:
    selected = option_menu(
        menu_title="Multiple Disease Prediction System",
        options=["Home", "Heart Disease", "Diabetes", "Parkinson’s"],
        icons=["house", "heart", "activity", "person"],
        default_index=0,
    )

# ---------------- Home Page ----------------
if selected == "Home":
    st.title("🧠🩺 Welcome to Multiple Disease Prediction System")
    st.markdown("""
    <div style='text-align: justify; font-size: 18px;'>
    This application predicts the likelihood of three common diseases based on patient health metrics:
    <ul>
        <li>❤️ <b>Heart Disease</b></li>
        <li>💉 <b>Diabetes</b></li>
        <li>🧠 <b>Parkinson’s Disease</b></li>
    </ul>
    Please select a disease from the sidebar to begin your prediction.
    </div>
    """, unsafe_allow_html=True)

# ---------------- Heart Disease Page ----------------
if selected == "Heart Disease":
    st.title("❤️ Heart Disease Prediction")

    # Dummy dataset (replace with your real dataset and model)
    X, y = make_classification(n_samples=500, n_features=13, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    heart_model = LogisticRegression()
    heart_model.fit(X_train, y_train)

    st.subheader("Enter the following patient details:")

    heart_features = [
        "Sex", "Chest Pain Type (cp)", "Resting Blood Pressure (trestbps)", "Cholesterol Level (chol)",
        "Fasting Blood Sugar (fbs)", "Resting ECG (restecg)", "Max Heart Rate (thalach)",
        "Exercise-Induced Angina (exang)", "Oldpeak", "Slope", "Number of Major Vessels (ca)", "Thalassemia (thal)",
        )"
    ]

    inputs = []
    for i in range(13):
        val = st.number_input(f"{heart_features[i]}", key=f"heart_{i}")
        inputs.append(val)

    if st.button("Predict Heart Disease"):
        prediction = heart_model.predict([inputs])
        result = "✅ Positive (High Risk)" if prediction[0] == 1 else "🟢 Negative (Low Risk)"
        st.success(f"Prediction: {result}")

# ---------------- Diabetes Page ----------------
if selected == "Diabetes":
    st.title("💉 Diabetes Prediction")

    X, y = make_classification(n_samples=500, n_features=8, random_state=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    diabetes_model = RandomForestClassifier()
    diabetes_model.fit(X_train, y_train)

    st.subheader("Enter the following patient details:")

    diabetes_features = [
        "Number of Pregnancies", "Glucose Level", "Blood Pressure", "Skin Thickness",
        "Insulin Level", "BMI", "Diabetes Pedigree Function", "Age"
    ]

    inputs = []
    for i in range(8):
        val = st.number_input(f"{diabetes_features[i]}", key=f"diabetes_{i}")
        inputs.append(val)

    if st.button("Predict Diabetes"):
        prediction = diabetes_model.predict([inputs])
        result = "✅ Positive (Diabetic)" if prediction[0] == 1 else "🟢 Negative (Not Diabetic)"
        st.success(f"Prediction: {result}")

# ---------------- Parkinson's Page ----------------
if selected == "Parkinson’s":
    st.title("🧠 Parkinson’s Disease Prediction")

    X, y = make_classification(n_samples=500, n_features=15, random_state=2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    parkinson_model = SVC(probability=True)
    parkinson_model.fit(X_train, y_train)

    st.subheader("Enter the following vocal measurements:")

    parkinson_features = [
        "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)", "MDVP:Jitter(Abs)",
        "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP", "MDVP:Shimmer", "MDVP:Shimmer(dB)",
        "Shimmer:APQ3", "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA", "NHR"
    ]

    inputs = []
    for i in range(15):
        val = st.number_input(f"{parkinson_features[i]}", key=f"parkinson_{i}")
        inputs.append(val)

    if st.button("Predict Parkinson’s Disease"):
        prediction = parkinson_model.predict([inputs])
        result = "✅ Positive (Parkinson’s Detected)" if prediction[0] == 1 else "🟢 Negative (No Parkinson’s)"
        st.success(f"Prediction: {result}")
