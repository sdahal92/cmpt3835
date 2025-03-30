import streamlit as st
import pandas as pd
import joblib

# Load the saved pipeline (preprocessing + classifier)
model = joblib.load("client_retention_model.pkl")

st.title("Client Retention Prediction App")

# Sidebar for user input
st.sidebar.header("Enter Client Details")

# Collect user input
input_data = {
    'contact_method': st.sidebar.selectbox("Contact Method", ['email', 'phone', 'text']),
    'household': st.sidebar.selectbox("Household", ['single', 'family', 'group']),
    'preferred_languages': st.sidebar.selectbox("Preferred Language", ['English', 'Arabic', 'Spanish']),
    'sex_new': st.sidebar.selectbox("Sex", ['Male', 'Female']),
    'status': st.sidebar.selectbox("Status", ['new', 'returning']),
    'Season': st.sidebar.selectbox("Season", ['Winter', 'Spring', 'Summer', 'Fall']),
    'Month': st.sidebar.selectbox("Month", ['January', 'February', 'March', 'April', 'May', 'June',
                                            'July', 'August', 'September', 'October', 'November', 'December']),
    'latest_language_is_english': st.sidebar.selectbox("Latest Language is English?", [0, 1]),
    'age': st.sidebar.slider("Age", 18, 90, 35),
    'dependents_qty': st.sidebar.slider("Number of Dependents", 0, 10, 1),
    'distance_km': st.sidebar.slider("Distance (in km)", 0, 100, 10),
    'num_of_contact_methods': st.sidebar.slider("Number of Contact Methods", 1, 5, 2),
}

# Convert input to DataFrame
input_df = pd.DataFrame([input_data])

# Predict
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result:")
    st.success(f"Client will return: {'Yes' if prediction == 1 else 'No'}")
    st.info(f"Probability of returning: {prediction_proba:.2f}")
