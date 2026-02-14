import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Load the saved components
model = joblib.load('loan_model.pkl')
scaler = joblib.load('scaler.pkl')
features = joblib.load('features.pkl')

st.title("🏦 Loan Repayment Predictor")
st.markdown("Enter customer details below to predict if the loan will be paid back.")

# 2. User Input UI (Adjust these based on your actual column names)
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
    
with col2:
    interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=30.0, value=5.0)
    loan_term = st.selectbox("Loan Term (Months)", [12, 24, 36, 48, 60])

# Add dropdowns for your categorical columns (e.g., marital status)
marital_status = st.selectbox("Marital Status", ["Married", "Single", "Divorced"])

# 3. Prediction Logic
if st.button("Predict Repayment"):
    # Create a raw input dictionary
    input_data = {
        'age': age,
        'credit_score': credit_score,
        'interest_rate': interest_rate / 100, # convert % to decimal
        'loan_term': loan_term,
        'marital_status': marital_status
    }
    
    # Convert input to DataFrame and match the One-Hot Encoding of your training
    input_df = pd.DataFrame([input_data])
    input_df_encoded = pd.get_dummies(input_df).reindex(columns=features, fill_value=0)
    
    # SCALE the input
    input_scaled = scaler.transform(input_df_encoded)
    
    # PREDICT
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction[0] == 1:
        st.success(f"✅ Likely to Pay Back (Confidence: {probability:.2%})")
    else:
        st.error(f"❌ High Risk of Default (Confidence: {1-probability:.2%})")