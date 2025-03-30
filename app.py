# app.py
import streamlit as st
import pandas as pd
import joblib

# Load the trained pipeline (preprocessing + model)
model = joblib.load("client_retention_model.pkl")

st.title("🔁 Client Retention Predictor")
st.write("Enter client information to predict if they will return, using the top 5 most important features.")

# Input form
with st.form("input_form"):
    season = st.selectbox("Season", ['Spring', 'Summer', 'Fall', 'Winter'])
    month = st.selectbox("Month", ['January', 'February', 'March', 'April', 'May', 'June',
                                   'July', 'August', 'September', 'October', 'November', 'December'])
    preferred_language = st.selectbox("Preferred Language", ['english', 'other'])
    distance_km = st.number_input("Distance to Location (km)", min_value=0.0, max_value=50.0, value=5.0)
    age = st.slider("Age", min_value=18, max_value=100, value=35)

    submitted = st.form_submit_button("Predict")

# Predict and show results
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
        st.success(f"✅ Client is **likely to return** (Probability: {round(probability, 2)})")
    else:
        st.warning(f"⚠️ Client is **unlikely to return** (Probability: {round(probability, 2)})")
