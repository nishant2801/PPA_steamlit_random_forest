import streamlit as st
import pandas as pd
import joblib

# Load trained RandomForest model
model = joblib.load("model.pkl")

def predict(input_dict):
    """
    input_dict: dictionary of feature_name:value
    returns prediction and probability
    """
    df = pd.DataFrame([input_dict])
    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0].tolist()
    return pred, prob

st.title("Employee Attrition Prediction")

st.write("Enter the employee features:")

# Example input fields — change as needed
age = st.number_input("Age", 18, 60, 30)
monthly_income = st.number_input("Monthly Income", 1000, 20000, 5000)
distance = st.number_input("Distance From Home", 1, 30, 5)

if st.button("Predict"):
    input_data = {
        "Age": age,
        "MonthlyIncome": monthly_income,
        "DistanceFromHome": distance
    }
    pred, prob = predict(input_data)

    st.write("Prediction:", "Yes" if pred == 1 else "No")
    st.write("Probabilities:", prob)

