import streamlit as st
import joblib

# Load model
model = joblib.load("model.pkl")

# Page configuration
st.set_page_config(
    page_title="Fraud Detection",
    page_icon="💳",
    layout="centered"
)

# Title
st.title("💳 Fraud Detection System")
st.write("Check whether a financial transaction is fraudulent or legitimate.")

st.divider()

# Transaction inputs
st.subheader("Enter Transaction Details")

amount = st.number_input(
    "Transaction Amount",
    min_value=0.0,
    value=1000.0
)

oldbalanceOrg = st.number_input(
    "Sender's Old Balance",
    min_value=0.0,
    value=5000.0
)

newbalanceOrig = st.number_input(
    "Sender's New Balance",
    min_value=0.0,
    value=4000.0
)

oldbalanceDest = st.number_input(
    "Receiver's Old Balance",
    min_value=0.0,
    value=2000.0
)

newbalanceDest = st.number_input(
    "Receiver's New Balance",
    min_value=0.0,
    value=3000.0
)

st.divider()

# Prediction
if st.button("🔍 Check Transaction", use_container_width=True):

    transaction = [[
        amount,
        oldbalanceOrg,
        newbalanceOrig,
        oldbalanceDest,
        newbalanceDest
    ]]

    prediction = model.predict(transaction)

    if prediction[0] == 1:
        st.error("🚨 Fraudulent Transaction Detected!")
    else:
        st.success("✅ Transaction is Legitimate")