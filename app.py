import streamlit as st
import joblib, pickle, pandas as pd

st.title("Loan Approval & Risk Prediction System")

model = joblib.load("loan_model.pkl")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("encoder.pkl")
with open("feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

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
    scaled = scaler.transform(df[["Age","Income","LoanAmount","LoanTerm","ExistingDebt","PropertyValue"]])
    encoded = encoder.transform(df[["EmploymentType","Education","Dependents"]])

    final_input = pd.DataFrame(scaled, columns=["Age","Income","LoanAmount","LoanTerm","ExistingDebt","PropertyValue"])
    encoded_df = pd.DataFrame(encoded.toarray(), columns=encoder.get_feature_names_out(["EmploymentType","Education","Dependents"]))
    final_input = pd.concat([final_input, encoded_df], axis=1)
    final_input = final_input[feature_columns]

    prediction = model.predict(final_input)[0]
    probability = model.predict_proba(final_input).max() * 100
    risk_level = "Low" if probability > 80 else "Medium" if probability > 50 else "High"

    st.write("Prediction:", "Approved" if prediction == 1 else "Rejected")
    st.write("Approval Probability:", round(probability, 2), "%")
    st.write("Risk Level:", risk_level)
