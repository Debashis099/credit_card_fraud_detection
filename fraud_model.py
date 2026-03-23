import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib

class FraudDetector:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.features = ['amount', 'hour', 'day_of_week']
    
    def train(self, transactions_df):
        """Train model on transaction data"""
        X = transactions_df[self.features].values
        X_scaled = self.scaler.fit_transform(X)
        
        self.model = IsolationForest(contamination=0.05, random_state=42)
        self.model.fit(X_scaled)
        print(f"✅ Trained on {len(transactions_df)} transactions")
    
    def predict(self, transaction):
        """Predict fraud probability"""
        if self.model is None:
            return False, 0.5
            
        X = np.array([[transaction[f] for f in self.features]])
        X_scaled = self.scaler.transform(X)
        
        is_fraud = self.model.predict(X_scaled)[0] == -1
        score = self.model.decision_function(X_scaled)[0]
        fraud_prob = 1 - (score + 1) / 2
        
        return is_fraud, round(fraud_prob, 3)

# Test it
if __name__ == "__main__":
    detector = FraudDetector()
    
    # Mock data
    data = pd.DataFrame({
        'amount': [50, 1000, 45, 75, 2000],
        'hour': [14, 2, 12, 18, 3],
        'day_of_week': [1, 5, 2, 4, 6]
    })
    
    detector.train(data)
    
    test_tx = {'amount': 1500, 'hour': 3, 'day_of_week': 5}
    is_fraud, prob = detector.predict(test_tx)
    print(f"Fraud: {is_fraud}, Probability: {prob}")