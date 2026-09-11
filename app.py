
from fastapi import FastAPI
import joblib, pickle
import pandas as pd

# Load saved artifacts
model = joblib.load("loan_model.pkl")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("encoder.pkl")

with open("feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

app = FastAPI()

@app.get("/health")
def health_check():
    return {"status": "running"}

@app.post("/predict")
def predict(applicant: dict):
    df = pd.DataFrame([applicant])

    numerical_cols = ["Age", "Income", "LoanAmount", "LoanTerm", "ExistingDebt", "PropertyValue"]
    categorical_cols = ["EmploymentType", "Education", "Dependents"]

    scaled = scaler.transform(df[numerical_cols])
    encoded = encoder.transform(df[categorical_cols])

    final_input = pd.DataFrame(scaled, columns=numerical_cols)
    encoded_df = pd.DataFrame(encoded.toarray(), columns=encoder.get_feature_names_out(categorical_cols))
    final_input = pd.concat([final_input, encoded_df], axis=1)
    final_input = final_input[feature_columns]

    prediction = model.predict(final_input)[0]
    probability = model.predict_proba(final_input).max() * 100

    risk_level = "Low" if probability > 80 else "Medium" if probability > 50 else "High"

    return {
        "prediction": "Approved" if prediction == 1 else "Rejected",
        "approval_probability": round(probability, 2),
        "risk_level": risk_level
    }
