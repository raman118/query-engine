"""Test configuration and shared fixtures."""

import pytest
import pandas as pd
import numpy as np

@pytest.fixture(scope="session")
def large_dataset():
    """Create a large dataset for performance testing."""
    np.random.seed(42)
    n = 100000
    return pd.DataFrame({
        'value': np.random.normal(100, 15, n),
        'category': np.random.choice(['A', 'B', 'C'], n),
        'region': np.random.choice(['North', 'South', 'East', 'West'], n),
        'score': np.random.uniform(0, 100, n),
        'id': range(n)
    })

@pytest.fixture(scope="session")
def small_dataset():
    """Create a small dataset for basic testing."""
    np.random.seed(42)
    n = 1000
    return pd.DataFrame({
        'value': np.random.normal(100, 15, n),
        'category': np.random.choice(['A', 'B', 'C'], n),
        'id': range(n)
    })

@pytest.fixture
def edge_case_dataset():
    """Create dataset with edge cases."""
    return pd.DataFrame({
        'value': [0, 1, -1, 1e6, -1e6, np.nan, np.inf, -np.inf],
        'category': ['A', 'B', 'A', 'C', 'B', 'C', 'A', 'B'],
        'id': range(8)
    })