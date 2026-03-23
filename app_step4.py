import streamlit as st
from transaction_generator import TransactionGenerator
from fraud_model import FraudDetector

st.set_page_config(page_title="Fraud Detection", layout="wide")

# Initialize
if 'transactions' not in st.session_state:
    st.session_state.transactions = []
    st.session_state.generator = TransactionGenerator()
    st.session_state.detector = FraudDetector()
    st.session_state.model_trained = False

st.title("🚨 Fraud Detection System - Step 4")

# Sidebar
st.sidebar.header("Controls")
if st.sidebar.button("➕ Generate Transaction"):
    tx = st.session_state.generator.generate_transaction()
    
    # Predict fraud
    is_fraud, fraud_prob = st.session_state.detector.predict(tx)
    tx['is_fraud'] = is_fraud
    tx['fraud_probability'] = fraud_prob
    tx['status'] = '🚨 FRAUD' if is_fraud else '✅ OK'
    
    st.session_state.transactions.append(tx)
    st.rerun()

if st.sidebar.button("🔄 Train Model") and len(st.session_state.transactions) > 10:
    df = pd.DataFrame(st.session_state.transactions)
    st.session_state.detector.train(df)
    st.session_state.model_trained = True
    st.sidebar.success("✅ Model Trained!")

# Main dashboard
col1, col2 = st.columns(2)

with col1:
    st.header("📋 Recent Transactions")
    if st.session_state.transactions:
        df = pd.DataFrame(st.session_state.transactions[-10:])
        st.dataframe(df[['transaction_id', 'amount', 'merchant', 'status']], 
                    use_container_width=True)

with col2:
    st.header("📊 Stats")
    if st.session_state.transactions:
        df = pd.DataFrame(st.session_state.transactions)
        col1, col2, col3 = st.columns(3)
        col1.metric("Total", len(df))
        col2.metric("Frauds", len(df[df['is_fraud']]))
        col3.metric("Fraud Rate", f"{len(df[df['is_fraud']])/len(df)*100:.1f}%")

st.markdown("---")
st.caption("Step 4 Complete ✅ - Run: `streamlit run app_step4.py`")