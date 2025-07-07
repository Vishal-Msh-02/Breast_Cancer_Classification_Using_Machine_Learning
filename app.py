import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("xgboost_breast_cancer_model.pkl")
scaler = joblib.load("scaler.pkl")

# Title and instructions
st.title("💗 Breast Cancer Prediction App")
st.write("Enter the values below to predict whether the tumor is **Malignant (Cancerous)** or **Benign (Non-Cancerous)**")

# Final 22 features (after dropping highly correlated ones)
feature_names = [
    'texture_mean', 'smoothness_mean', 'compactness_mean', 'concave points_mean', 'symmetry_mean',
    'fractal_dimension_mean', 'texture_se', 'area_se', 'smoothness_se', 'compactness_se',
    'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se', 'texture_worst',
    'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst',
    'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

# Collect inputs
user_inputs = []
for feature in feature_names:
    value = st.number_input(f"Enter {feature}:", min_value=0.0, step=0.01)
    user_inputs.append(value)

# Predict on button click
if st.button("Predict"):
    # Scale and predict
    scaled_inputs = scaler.transform([np.array(user_inputs)])
    prediction = model.predict(scaled_inputs)[0]

    # Display result
    result = "🎯 **Malignant (Cancerous)**" if prediction == 1 else "✅ **Benign (Non-Cancerous)**"
    st.subheader("Prediction Result:")
    st.success(result)