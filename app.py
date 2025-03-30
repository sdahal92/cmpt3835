import shap
import matplotlib.pyplot as plt

# Get model and preprocessor from pipeline
model_only = model.named_steps['classifier']
preprocessor = model.named_steps['preprocessing']

# Preprocess input to match model
input_transformed = preprocessor.transform(input_df)

# Create SHAP explainer (TreeExplainer works with GradientBoosting)
explainer = shap.Explainer(model_only, feature_names=preprocessor.get_feature_names_out())

# Calculate SHAP values for the single instance
shap_values = explainer(input_transformed)

# Visualize SHAP
st.subheader("🧠 SHAP Explanation (Why this prediction?)")

# Plot waterfall
fig, ax = plt.subplots(figsize=(10, 5))
shap.plots.waterfall(shap_values[0], max_display=10, show=False)
st.pyplot(fig)
