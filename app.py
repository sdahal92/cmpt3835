# app.py
import streamlit as st
import pandas as pd
import joblib

# Load the trained pipeline (preprocessing + classifier)
model = joblib.load("client_retention_model.pkl")

st.title("🔄 Client Retention Predictor (Top Features Only)")
st.write("Prediction based on top 5 most important features.")

# Input form for top 5 features only
with st.form("prediction_form"):
    season = st.selectbox("Season", ['Spring', 'Summer', 'Fall', 'Winter'])
    month = st.selectbox("Month", [
        'January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December'
    ])
    preferred_language = st.selectbox("Preferred Language", ['english', 'other'])
    distance_km = st.slider("Distance to Location (km)", 0.0, 50.0, 5.0)
    age = st.slider("Age", 18, 100, 35)

    submitted = st.form_submit_button("Predict")

# Prepare input and predict
if submitted:
    input_df = pd.DataFrame([{
        'Season': season,
        'Month': month,
        'preferred_languages': preferred_language,
        'distance_km': distance_km,
        'age': age
    }])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.markdown("---")
    st.subheader("Prediction Result:")
    if prediction == 1:
        st.success(f"✅ Client is likely to return (Probability: {round(probability, 2)})")
    else:
        st.warning(f"⚠️ Client may not return (Probability: {round(probability, 2)})")
