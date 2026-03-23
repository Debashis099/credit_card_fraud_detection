import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
import threading
from datetime import datetime
import random
from transaction_generator import TransactionGenerator
from fraud_model import FraudDetector
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_model():
    # Load data
    df = pd.read_csv("creditcard.csv")

    # Features & target
    X = df.drop("Class", axis=1)
    y = df["Class"]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Model
    model = RandomForestClassifier(class_weight="balanced")

    # Train
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, "fraud_model.pkl")

    return "Model Trained Successfully ✅"

# Page config
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'transactions' not in st.session_state:
    st.session_state.transactions = []
    st.session_state.generator = TransactionGenerator()
    st.session_state.detector = FraudDetector()
    st.session_state.is_running = False
    st.session_state.model_trained = False

st.title("🚨 Real-Time Fraud Detection System")
st.markdown("**Complete Dashboard with Advanced Visualizations**")

# === SIDEBAR CONTROLS ===
st.sidebar.header("⚙️ Controls")
transactions_per_sec = st.sidebar.slider("Transactions/sec", 1, 5, 2)

# Auto-generate toggle
auto_mode = st.sidebar.checkbox("▶️ Auto Generate Transactions", value=True)

# Manual generate button
if st.sidebar.button("➕ Generate Single Transaction", type="secondary"):
    tx = st.session_state.generator.generate_transaction()
    is_fraud, fraud_prob = st.session_state.detector.predict(tx)
    tx['is_fraud'] = is_fraud
    tx['fraud_probability'] = fraud_prob
    tx['status'] = '🚨 FRAUD' if is_fraud else '✅ OK'
    st.session_state.transactions.append(tx)
    st.rerun()

# Train model
if st.sidebar.button("🔄 Train ML Model", type="primary") and len(st.session_state.transactions) > 20:
    df = pd.DataFrame(st.session_state.transactions)
    st.session_state.detector.train(df)
    st.session_state.model_trained = True
    st.sidebar.success("✅ ML Model Trained!")

# === BACKGROUND AUTO-GENERATOR ===
def auto_generate():
    """Background transaction generator"""
    while st.session_state.is_running:
        tx = st.session_state.generator.generate_transaction()
        is_fraud, fraud_prob = st.session_state.detector.predict(tx)
        tx['is_fraud'] = is_fraud
        tx['fraud_probability'] = fraud_prob
        tx['status'] = '🚨 FRAUD' if is_fraud else '✅ OK'
        st.session_state.transactions.append(tx)
        
        # Keep last 500 transactions
        if len(st.session_state.transactions) > 500:
            st.session_state.transactions = st.session_state.transactions[-500:]
        
        time.sleep(1 / transactions_per_sec)

# Toggle auto mode
if auto_mode and not st.session_state.is_running:
    st.session_state.is_running = True
    thread = threading.Thread(target=auto_generate, daemon=True)
    thread.start()
elif not auto_mode:
    st.session_state.is_running = False

# === MAIN DASHBOARD ===
col1, col2, col3 = st.columns([2, 1, 1])

# Live transaction feed
with col1:
    st.header("📊 Live Transaction Feed")
    if st.session_state.transactions:
        df_live = pd.DataFrame(st.session_state.transactions[-15:])
        df_live['timestamp'] = pd.to_datetime(df_live['timestamp'])
        
        st.dataframe(
            df_live[['transaction_id', 'amount', 'merchant', 'status', 'fraud_probability']],
            use_container_width=True,
            height=350,
            column_config={
                "status": st.column_config.TextColumn("Status"),
                "amount": st.column_config.NumberColumn("Amount", format="$%.2f")
            }
        )

# Fraud alerts
with col2:
    st.header("🚨 Recent Frauds")
    if st.session_state.transactions:
        df = pd.DataFrame(st.session_state.transactions)
        recent_frauds = df[df['is_fraud'] == True].tail(5)
        
        if not recent_frauds.empty:
            for _, fraud in recent_frauds.iterrows():
                with st.expander(f"🚨 {fraud['transaction_id']}"):
                    st.metric("Amount", f"${fraud['amount']:.2f}")
                    st.metric("Merchant", fraud['merchant'])
                    st.metric("Risk Score", f"{fraud['fraud_probability']:.1%}")
        else:
            st.info("✅ No frauds detected")

# Live metrics
with col3:
    st.header("📈 Key Metrics")
    if st.session_state.transactions:
        df = pd.DataFrame(st.session_state.transactions)
        c1, c2, c3 = st.columns(3)
        
        total = len(df)
        frauds = len(df[df['is_fraud']])
        fraud_rate = (frauds / total * 100) if total > 0 else 0
        
        with c1:
            st.metric("Total TXs", total, f"+{transactions_per_sec}/s")
        with c2:
            st.metric("Fradulent", frauds)
        with c3:
            st.metric("Fraud Rate", f"{fraud_rate:.2f}%")

# === ADVANCED CHARTS (Step 6) ===
st.markdown("---")
st.header("📊 Advanced Analytics")

if st.session_state.transactions:
    df = pd.DataFrame(st.session_state.transactions)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    col1, col2 = st.columns(2)
    
    # Chart 1: Fraud Rate Over Time
    with col1:
        df_time = df.tail(100).set_index('timestamp').resample('30s').agg({
            'is_fraud': 'mean',
            'transaction_id': 'count'
        }).reset_index()
        
        fig1 = px.line(df_time, x='timestamp', y='is_fraud',
                      title="🔴 Fraud Rate Trend (Last 100 TXs)",
                      labels={'is_fraud': 'Fraud Probability'})
        fig1.update_traces(line_color='red', line_width=3)
        st.plotly_chart(fig1, use_container_width=True)
    
    # Chart 2: Amount Distribution
    with col2:
        fig2 = px.histogram(df.tail(200), x='amount', color='is_fraud',
                           title="💰 Transaction Amount Distribution",
                           nbins=30,
                           color_discrete_map={True: 'red', False: 'green'})
        st.plotly_chart(fig2, use_container_width=True)
    
    # Chart 3: Risk Score Heatmap
    col3, col4 = st.columns(2)
    
    with col3:
        fig3 = px.scatter(df.tail(100), x='hour', y='amount', 
                         size='fraud_probability', color='is_fraud',
                         title="⏰ Risk by Time & Amount",
                         hover_data=['merchant'])
        st.plotly_chart(fig3, use_container_width=True)
    
    with col4:
        # Merchant fraud heatmap
        merchant_fraud = df.groupby('merchant')['is_fraud'].agg(['count', 'sum']).reset_index()
        merchant_fraud['fraud_rate'] = merchant_fraud['sum'] / merchant_fraud['count']
        
        fig4 = px.treemap(merchant_fraud, path=['merchant'], 
                         values='count', color='fraud_rate',
                         title="🏪 Merchant Fraud Risk",
                         color_continuous_scale='RdYlGn_r')
        st.plotly_chart(fig4, use_container_width=True)

# === STATUS BAR ===
st.markdown("---")
status_col1, status_col2 = st.columns(2)

with status_col1:
    model_status = "✅ READY" if st.session_state.model_trained else "⚠️ Train Model"
    st.metric("ML Model", model_status)

with status_col2:
    sim_status = "🟢 LIVE" if st.session_state.is_running else "🔴 STOPPED"
    st.metric("Simulation", sim_status)

# Auto-rerun for live updates
if st.session_state.is_running:
    time.sleep(0.5)
    st.rerun()

st.markdown("---")
st.caption("**🎉 Step 6 Complete!** Real-time dashboard with 4 advanced charts 🚀")