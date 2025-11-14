import streamlit as st
import pandas as pd
from model import predict

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
