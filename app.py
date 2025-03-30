import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Load model
model = joblib.load("client_retention_model.pkl")

# Get pipeline components
preprocessor = model.named_steps['preprocessing']
classifier = model.named_steps['classifier']

# Define your categorical and numerical columns again (must match training)
categorical_cols = ['contact_method', 'household', 'preferred_languages', 'sex_new', 'status', 'Season', 'Month', 'latest_language_is_english']
numerical_cols = ['age', 'dependents_qty', 'distance_km', 'num_of_contact_methods']

# Streamlit App
st.title("Client Retention Prediction")
st.write("Fill in the details below to predict if a client will return.")

# Input fields
input_data = {}
for col in categorical_cols:
    options = ['value1', 'value2']  # Replace with your actual unique values
    input_data[col] = st.selectbox(f"{col}:", options)

for col in numerical_cols:
    input_data[col] = st.number_input(f"{col}:", value=0.0)

# When user clicks "Predict"
if st.button("Predict"):
    input_df = pd.DataFrame([input_data])
    
    # Preprocess input
    input_transformed = preprocessor.transform(input_df)

    # Predict
    prediction = classifier.predict(input_transformed)[0]
    prediction_proba = classifier.predict_proba(input_transformed)[0][1]
    
    st.subheader("Prediction Result:")
    st.write("✅ Likely to Return" if prediction == 1 else "❌ Unlikely to Return")
    st.write(f"Confidence: {prediction_proba:.2f}")
    
    # SHAP Explanation
    st.subheader("SHAP Explanation (Why this prediction?)")
    explainer = shap.Explainer(classifier, preprocessor.transform)
    shap_values = explainer(input_df)

    # Plot SHAP
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0], max_display=10, show=False)
    st.pyplot(fig)
