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

    # Replace this with your real dataset
    X, y = make_classification(n_samples=500, n_features=13, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    heart_model = LogisticRegression()
    heart_model.fit(X_train, y_train)

    st.subheader("Enter the following details:")
    inputs = []
    for i in range(13):
        val = st.number_input(f"Feature {i+1}", key=f"heart_{i}")
        inputs.append(val)

    if st.button("Predict Heart Disease"):
        prediction = heart_model.predict([inputs])
        result = "✅ Positive (High Risk)" if prediction[0] == 1 else "🟢 Negative (Low Risk)"
        st.success(f"Prediction: {result}")

        # Input Feature Chart
        st.subheader("📊 Your Input Features")
        feature_labels = [f"Feature {i+1}" for i in range(13)]
        input_df = pd.DataFrame({'Feature': feature_labels, 'Value': inputs})
        st.bar_chart(input_df.set_index("Feature"))

        # Prediction Probability
        try:
            prob = heart_model.predict_proba([inputs])
            st.subheader("🧮 Prediction Probabilities")
            st.write(f"Chance of No Heart Disease: `{prob[0][0]:.2f}`")
            st.write(f"Chance of Heart Disease: `{prob[0][1]:.2f}`")
            st.progress(int(prob[0][1] * 100))
        except:
            st.info("Model doesn't support probability scores.")

# ---------------- Diabetes Page ----------------
if selected == "Diabetes":
    st.title("💉 Diabetes Prediction")

    # Replace this with your real dataset
    X, y = make_classification(n_samples=500, n_features=8, random_state=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    diabetes_model = RandomForestClassifier()
    diabetes_model.fit(X_train, y_train)

    st.subheader("Enter the following details:")
    inputs = []
    for i in range(8):
        val = st.number_input(f"Feature {i+1}", key=f"diabetes_{i}")
        inputs.append(val)

    if st.button("Predict Diabetes"):
        prediction = diabetes_model.predict([inputs])
        result = "✅ Positive (Diabetic)" if prediction[0] == 1 else "🟢 Negative (Not Diabetic)"
        st.success(f"Prediction: {result}")

        # Input Feature Chart
        st.subheader("📊 Your Input Features")
        feature_labels = [f"Feature {i+1}" for i in range(8)]
        input_df = pd.DataFrame({'Feature': feature_labels, 'Value': inputs})
        st.bar_chart(input_df.set_index("Feature"))

        # Prediction Probability
        try:
            prob = diabetes_model.predict_proba([inputs])
            st.subheader("🧮 Prediction Probabilities")
            st.write(f"Chance of No Diabetes: `{prob[0][0]:.2f}`")
            st.write(f"Chance of Diabetes: `{prob[0][1]:.2f}`")
            st.progress(int(prob[0][1] * 100))
        except:
            st.info("Model doesn't support probability scores.")

# ---------------- Parkinson's Page ----------------
if selected == "Parkinson’s":
    st.title("🧠 Parkinson’s Disease Prediction")

    # Replace this with your real dataset
    X, y = make_classification(n_samples=500, n_features=15, random_state=2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    parkinson_model = SVC(probability=True)
    parkinson_model.fit(X_train, y_train)

    st.subheader("Enter the following details:")
    inputs = []
    for i in range(15):
        val = st.number_input(f"Feature {i+1}", key=f"parkinson_{i}")
        inputs.append(val)

    if st.button("Predict Parkinson’s"):
        prediction = parkinson_model.predict([inputs])
        result = "✅ Positive (Parkinson’s Detected)" if prediction[0] == 1 else "🟢 Negative (No Parkinson’s)"
        st.success(f"Prediction: {result}")

        # Input Feature Chart
        st.subheader("📊 Your Input Features")
        feature_labels = [f"Feature {i+1}" for i in range(15)]
        input_df = pd.DataFrame({'Feature': feature_labels, 'Value': inputs})
        st.bar_chart(input_df.set_index("Feature"))

        # Prediction Probability
        try:
            prob = parkinson_model.predict_proba([inputs])
            st.subheader("🧮 Prediction Probabilities")
            st.write(f"Chance of No Parkinson's: `{prob[0][0]:.2f}`")
            st.write(f"Chance of Parkinson's: `{prob[0][1]:.2f}`")
            st.progress(int(prob[0][1] * 100))
        except:
            st.info("Model doesn't support probability scores.")
