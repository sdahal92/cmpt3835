import streamlit as st

# Title
st.title("Client Retention Prediction - MVP")

# Input fields
st.header("Enter Client Information")
visit_frequency = st.slider("Visits per Month", 0, 10, 3)
last_visit = st.selectbox("Last Visit", ["Last Week", "Last Month", "Over 3 Months Ago"])
service_used = st.selectbox("Most Used Service", ["Counseling", "Food Aid", "Medical Assistance"])

# Submit button
if st.button("Submit"):
    # Simulated prediction logic
    st.header("Prediction Outcome")
    if visit_frequency > 5 or last_visit == "Last Week":
        st.success("Likely to Return ✅")
    else:
        st.warning("Unlikely to Return ❌")

# Sidebar with Future Enhancements
st.sidebar.title("Future Enhancements")
st.sidebar.write("""
✅ **Add Real Machine Learning Model**  
✅ **Train Model with Client Data**  
✅ **Deploy in Google Cloud**  
""")



