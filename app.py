import streamlit as st
import joblib

# -----------------------------
# Load Model
# -----------------------------
saved_model = joblib.load("model.pkl")

model = saved_model["model"]
encoder = saved_model["encoder"]

# -----------------------------
# Page
# -----------------------------
st.set_page_config(
    page_title="FraudGuard",
    layout="wide"
)

# -----------------------------
# CSS
# -----------------------------
st.markdown("""
<style>

.block-container {
    max-width: 1100px;
    padding-top: 2rem;
}

.hero {
    padding: 30px;
    border-radius: 20px;
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
    margin-bottom: 25px;
}

.hero h1 {
    font-size: 42px;
    margin-bottom: 5px;
}

.hero p {
    font-size: 17px;
    color: #cbd5e1;
}

.card {
    padding: 25px;
    border-radius: 18px;
    border: 1px solid #e2e8f0;
    background: #ffffff;
    margin-bottom: 20px;
}

.stButton > button {
    width: 100%;
    height: 50px;
    border-radius: 12px;
    font-size: 17px;
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.markdown("""
<div class="hero">
    <h1>🛡️ FraudGuard AI</h1>
    <p>Machine Learning based Financial Transaction Fraud Detection</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Input section
# -----------------------------
st.subheader("💳 Transaction Information")

col1, col2 = st.columns(2)

with col1:

    transaction_type = st.selectbox(
        "Transaction Type",
        ["CASH_OUT", "TRANSFER", "PAYMENT", "DEBIT", "CASH_IN"]
    )

    amount = st.number_input(
        "Transaction Amount",
        min_value=0.0,
        value=1000.0,
        step=100.0
    )

    oldbalanceOrg = st.number_input(
        "Sender Old Balance",
        min_value=0.0,
        value=5000.0,
        step=100.0
    )

with col2:

    newbalanceOrig = st.number_input(
        "Sender New Balance",
        min_value=0.0,
        value=4000.0,
        step=100.0
    )

    oldbalanceDest = st.number_input(
        "Receiver Old Balance",
        min_value=0.0,
        value=2000.0,
        step=100.0
    )

    newbalanceDest = st.number_input(
        "Receiver New Balance",
        min_value=0.0,
        value=3000.0,
        step=100.0
    )

st.write("")

# -----------------------------
# Analyze
# -----------------------------
if st.button("🔍 Analyze Transaction"):

    try:

        # Encode transaction type
        type_encoded = encoder.transform(
            [transaction_type]
        )[0]

        transaction = [[
            type_encoded,
            amount,
            oldbalanceOrg,
            newbalanceOrig,
            oldbalanceDest,
            newbalanceDest
        ]]

        prediction = model.predict(transaction)[0]

        probabilities = model.predict_proba(transaction)[0]

        fraud_probability = probabilities[1] * 100
        safe_probability = probabilities[0] * 100

        st.divider()

        # -------------------------
        # Result
        # -------------------------
        if prediction == 1:

            st.error("🚨 FRAUDULENT TRANSACTION DETECTED")

            st.metric(
                "Fraud Probability",
                f"{fraud_probability:.2f}%"
            )

            st.progress(
                min(int(fraud_probability), 100)
            )

            st.warning(
                "The transaction shows patterns associated "
                "with fraudulent transactions."
            )

        else:

            st.success("✅ TRANSACTION APPEARS LEGITIMATE")

            st.metric(
                "Legitimate Probability",
                f"{safe_probability:.2f}%"
            )

            st.progress(
                min(int(safe_probability), 100)
            )

        # -------------------------
        # Summary
        # -------------------------
        st.subheader("📊 Analysis Summary")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Transaction",
                transaction_type
            )

        with c2:
            st.metric(
                "Amount",
                f"₹{amount:,.2f}"
            )

        with c3:

            if prediction == 1:
                st.metric("Risk", "HIGH")
            else:
                st.metric("Risk", "LOW")

    except Exception as e:

        st.error(f"Prediction error: {e}")