import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np

# Load the saved pipeline (preprocessing + classifier)
model = joblib.load("client_retention_model.pkl")
preprocessor = model.named_steps['preprocessing']
classifier = model.named_steps['classifier']

st.title("Client Retention Prediction App")

# Sample input form
st.sidebar.header("Input Client Features")
input_data = {
    'contact_method': st.sidebar.selectbox("Contact Method", ['email', 'phone', 'text']),
    'household': st.sidebar.selectbox("Household", ['single', 'family', 'group']),
    'preferred_languages': st.sidebar.selectbox("Preferred Language", ['English', 'Arabic', 'Spanish']),
    'sex_new': st.sidebar.selectbox("Sex", ['Male', 'Female']),
    'status': st.sidebar.selectbox("Status", ['new', 'returning']),
    'Season': st.sidebar.selectbox("Season", ['Winter', 'Spring', 'Summer', 'Fall']),
    'Month': st.sidebar.selectbox("Month", ['January', 'February', 'March', 'April', 'May', 'June',
                                            'July', 'August', 'September', 'October', 'November', 'December']),
    'latest_language_is_english': st.sidebar.selectbox("Latest Lang English?", [0, 1]),
    'age': st.sidebar.slider("Age", 18, 90, 35),
    'dependents_qty': st.sidebar.slider("Dependents Qty", 0, 10, 1),
    'distance_km': st.sidebar.slider("Distance (km)", 0, 100, 10),
    'num_of_contact_methods': st.sidebar.slider("Number of Contact Methods", 1, 5, 2),
}

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# Predict
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0][1]

    st.success(f"Prediction: {'Yes' if prediction == 1 else 'No'}")
    st.write(f"Probability of client returning: {prediction_proba:.2f}")

    # SHAP explanation
    st.subheader("🔍 SHAP Explanation (Why this prediction?)")

    # Preprocess the input
    transformed_input = preprocessor.transform(input_df)

    # Create SHAP explainer (use TreeExplainer for GradientBoosting)
    explainer = shap.Explainer(classifier, feature_names=preprocessor.get_feature_names_out())
    shap_values = explainer(transformed_input)

    # Plot SHAP
    fig, ax = plt.subplots(figsize=(10, 5))
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)
