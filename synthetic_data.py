"""
Synthetic data generator for testing the Approximate Query Engine.
Generates realistic-looking sales data with categories, prices, and user IDs.
"""

import pandas as pd
import numpy as np

def generate_sales_data(n: int = 1_000_000) -> pd.DataFrame:
    """Generate synthetic sales data for testing.
    
    Args:
        n (int): Number of records to generate. Defaults to 1,000,000.
    
    Returns:
        pd.DataFrame: DataFrame with columns:
            - id: Unique sale ID
            - category: Product category (Electronics, Clothing, Books, etc.)
            - price: Sale amount (exponential distribution)
            - user_id: Customer ID (uniform distribution)
    """
    np.random.seed(42)
    categories = ['Electronics', 'Clothing', 'Books', 'Home', 'Toys']
    data = {
        "id": np.arange(n),
        "category": np.random.choice(categories, size=n, p=[0.2, 0.3, 0.2, 0.2, 0.1]),
        "price": np.round(np.random.exponential(50, size=n) + 10, 2),
        "user_id": np.random.randint(1, 100_000, size=n)
    }
    return pd.DataFrame(data)
