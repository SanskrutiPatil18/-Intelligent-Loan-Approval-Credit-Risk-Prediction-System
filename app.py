import streamlit as st
import joblib, pickle
import pandas as pd

# Load saved artifacts
model = joblib.load("loan_model.pkl")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("encoder.pkl")
with open("feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

st.title("🏦 Loan Approval & Credit Risk Prediction")

# Input form
age = st.number_input("Age", min_value=18, max_value=70)
income = st.number_input("Income")
credit_score = st.number_input("Credit Score")
employment_type = st.selectbox("Employment Type", ["Salaried", "Self-Employed"])
loan_amount = st.number_input("Loan Amount")
loan_term = st.number_input("Loan Term (months)")
existing_debt = st.number_input("Existing Debt")
dependents = st.number_input("Dependents", min_value=0)
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
property_value = st.number_input("Property Value")

if st.button("Predict Loan Approval"):
    applicant = {
        "Age": age,
        "Income": income,
        "CreditScore": credit_score,
        "EmploymentType": employment_type,
        "LoanAmount": loan_amount,
        "LoanTerm": loan_term,
        "ExistingDebt": existing_debt,
        "Dependents": dependents,
        "Education": education,
        "PropertyValue": property_value
    }

    df = pd.DataFrame([applicant])
    numerical_cols = ["Age","Income","LoanAmount","LoanTerm","ExistingDebt","PropertyValue"]
    categorical_cols = ["EmploymentType","Education","Dependents"]

    scaled = scaler.transform(df[numerical_cols])
    encoded = encoder.transform(df[categorical_cols])

    final_input = pd.DataFrame(scaled, columns=numerical_cols)
    encoded_df = pd.DataFrame(encoded.toarray(), columns=encoder.get_feature_names_out(categorical_cols))
    final_input = pd.concat([final_input, encoded_df], axis=1)
    final_input = final_input[feature_columns]

    prediction = model.predict(final_input)[0]
    probability = model.predict_proba(final_input).max() * 100
    risk_level = "Low" if probability > 80 else "Medium" if probability > 50 else "High"

    st.success(f"Prediction: {'Approved' if prediction == 1 else 'Rejected'}")
    st.info(f"Approval Probability: {round(probability, 2)}%")
    st.warning(f"Risk Level: {risk_level}")
