import random
from datetime import datetime
import pandas as pd

class TransactionGenerator:
    def __init__(self):
        self.merchants = [
            'Amazon', 'Walmart', 'Target', 'BestBuy', 'Apple Store',
            'Gas Station', 'Restaurant', 'Hotel', 'Airline', 'Online Shop'
        ]
        self.categories = ['Electronics', 'Groceries', 'Travel', 'Dining', 'Gas']
    
    def generate_transaction(self):
        """Generate single realistic transaction"""
        now = datetime.now()
        transaction = {
            'timestamp': now,
            'transaction_id': f"TX_{random.randint(100000, 999999)}",
            'amount': round(random.gauss(50, 25), 2),
            'merchant': random.choice(self.merchants),
            'category': random.choice(self.categories),
            'user_id': random.randint(1000, 5000),
            'hour': now.hour,
            'day_of_week': now.weekday(),
        }
        
        # 5% chance of fraud
        if random.random() < 0.05:
            self._inject_fraud(transaction)
            
        return transaction
    
    def _inject_fraud(self, transaction):
        """Add fraud patterns"""
        fraud_type = random.choice(['high_amount', 'rapid_fire'])
        if fraud_type == 'high_amount':
            transaction['amount'] = random.uniform(500, 2000)
        elif fraud_type == 'rapid_fire':
            transaction['merchant'] = 'Suspicious Merchant'

# Test it
if __name__ == "__main__":
    gen = TransactionGenerator()
    for _ in range(5):
        print(gen.generate_transaction())