import time
import threading

# Add to session_state initialization
if 'is_running' not in st.session_state:
    st.session_state.is_running = False

def auto_generate():
    """Background transaction generator"""
    while st.session_state.is_running:
        tx = st.session_state.generator.generate_transaction()
        is_fraud, fraud_prob = st.session_state.detector.predict(tx)
        tx['is_fraud'] = is_fraud
        tx['fraud_probability'] = fraud_prob
        tx['status'] = '🚨 FRAUD' if is_fraud else '✅ OK'
        st.session_state.transactions.append(tx)
        
        # Limit to 500 transactions
        if len(st.session_state.transactions) > 500:
            st.session_state.transactions = st.session_state.transactions[-500:]
        time.sleep(1)  # 1 transaction per second

# Add to sidebar
st.sidebar.checkbox("▶️ Auto-generate (1/sec)", 
                   key="auto_mode", 
                   on_change=lambda: toggle_auto_mode())

def toggle_auto_mode():
    if st.session_state.auto_mode and not st.session_state.is_running:
        st.session_state.is_running = True
        thread = threading.Thread(target=auto_generate, daemon=True)
        thread.start()
    else:
        st.session_state.is_running = False

# Auto-refresh
time.sleep(1)
st.rerun()