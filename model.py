
import joblib
import pandas as pd

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
