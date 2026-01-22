import streamlit as st
import pandas as pd
import pickle
import numpy as np

# -----------------------------
# App title and description
# -----------------------------
st.set_page_config(page_title="Bank Churn Predictor",
                   page_icon="🏦", layout="centered")
st.title("🏦 Bank Churn Predictor")
st.write("Enter the customer details below to predict if the customer is likely to churn.")


# -----------------------------
# User Inputs
# -----------------------------
with st.form(key="churn_form"):
    st.subheader("Customer Information")

    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
    credit_score = st.number_input(
        "Credit Score", min_value=350, max_value=850, value=600, step=1)
    tenure = st.number_input(
        "Tenure (Years)", min_value=0, max_value=10, value=3, step=1)
    balance = st.number_input(
        "Account Balance", min_value=0.0, max_value=1000000.0, value=1000.0, step=100.0)
    products = st.number_input(
        "Number of Products", min_value=0, max_value=10, value=1, step=1)
    salary = st.number_input(
        "Estimated Salary", min_value=0.0, value=50000.0, step=100.0)
    credit_card = st.selectbox("Credit Card", ["Yes", "No"])
    active_member = st.selectbox("Active Member", ["Yes", "No"])
    country = st.selectbox("Country", ["Germany", "Spain", "France"])

    submit_button = st.form_submit_button(label="Predict Churn")

# -----------------------------
# Encoding inputs
# -----------------------------
if submit_button:
    # Credit card & active member to 0/1
    cy = 1 if credit_card == "Yes" else 0
    ay = 1 if active_member == "Yes" else 0

    # Load LabelEncoder for gender
    with open("Labelencod.pkl", "rb") as f:
        lb_encod = pickle.load(f)

    gender_encoded = lb_encod.transform([gender])[0]

    # Create input dataframe
    user_input = pd.DataFrame({
        'credit_score': [credit_score],
        'gender': [gender_encoded],
        'age': [age],
        'tenure': [tenure],
        'balance': [balance],
        'products_number': [products],
        'credit_card': [cy],
        'active_member': [ay],
        'estimated_salary': [salary]
    })

    # Load OneHotEncoder for country
    with open("ohe.pkl", "rb") as f:
        ohe = pickle.load(f)

    country_encoded = ohe.transform([[country]])
    country_df = pd.DataFrame(
        country_encoded, columns=ohe.get_feature_names_out())

    # Combine all features
    user_input_final = pd.concat([user_input, country_df], axis=1)

    # Load Gradient Boosting model
    with open("GBC Model.pkl", "rb") as f:
        gb = pickle.load(f)

    # -----------------------------
    # Prediction
    # -----------------------------
    predicted_result = gb.predict(user_input_final)[0]  # 0 or 1
    probability = gb.predict_proba(user_input_final)[
        0][1]  # probability of churn

    # -----------------------------
    # Display results
    # -----------------------------
    st.subheader("Prediction Result")

    if predicted_result == 1:
        st.markdown(
            f"<h2 style='color:red'>Yes 🚨 Customer Likely to Churn</h2>", unsafe_allow_html=True)
    else:
        st.markdown(
            f"<h2 style='color:green'>No ✅ Customer is Safe</h2>", unsafe_allow_html=True)

    st.info(f"Churn Probability: **{probability:.2f}**")
