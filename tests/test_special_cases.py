"""Tests for special cases and numerical stability in sampling module."""

import pytest
import pandas as pd
import numpy as np
from sampling import SimpleRandomSampler, StratifiedSampler, ReservoirSampler

def test_numerical_stability():
    """Test numerical stability with extreme values."""
    # Create dataset with extreme values
    extreme_df = pd.DataFrame({
        'value': [1e-10, 1e10] * 50 + [1.0] * 100,
        'category': ['A', 'B'] * 50 + ['C'] * 100
    })
    
    sampler = StratifiedSampler()
    
    # Test with extreme values
    sample = sampler.sample(extreme_df, 'category', size=50)
    bounds = sampler.estimate_error_bounds(sample, 'value')
    
    # Check that bounds are finite
    assert np.isfinite(bounds.lower_bound)
    assert np.isfinite(bounds.upper_bound)
    assert np.isfinite(bounds.margin_of_error)
    
    # Test with very small values
    small_df = pd.DataFrame({
        'value': [1e-15, 1e-14, 1e-13] * 100,
        'category': ['A', 'B', 'C'] * 100
    })
    
    sample = sampler.sample(small_df, 'category', size=50)
    bounds = sampler.estimate_error_bounds(sample, 'value')
    assert np.isfinite(bounds.margin_of_error)
    
    # Test with very large values
    large_df = pd.DataFrame({
        'value': [1e15, 1e14, 1e13] * 100,
        'category': ['A', 'B', 'C'] * 100
    })
    
    sample = sampler.sample(large_df, 'category', size=50)
    bounds = sampler.estimate_error_bounds(sample, 'value')
    assert np.isfinite(bounds.margin_of_error)

def test_special_cases():
    """Test handling of special numeric cases."""
    special_df = pd.DataFrame({
        'value': [
            np.inf, -np.inf, np.nan,  # Special values
            1e-10, -1e-10,           # Very small values
            1e10, -1e10,             # Very large values
            0, 1, -1                 # Regular values
        ] * 10,
        'category': ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A'] * 10
    })
    
    sampler = SimpleRandomSampler()
    
    # Test sampling with special values
    sample = sampler.sample(special_df, size=20)
    assert len(sample) == 20
    
    # Test error bounds with special values
    # Should handle them gracefully by excluding them
    bounds = sampler.estimate_error_bounds(sample, 'value')
    assert np.isfinite(bounds.estimate)
    assert np.isfinite(bounds.margin_of_error)
    
    # Test stratified sampling with special values
    strat_sampler = StratifiedSampler()
    strat_sample = strat_sampler.sample(special_df, 'category', size=20)
    assert len(strat_sample) > 0
    
    # Error bounds should handle special values appropriately
    bounds = strat_sampler.estimate_error_bounds(strat_sample, 'value')
    assert np.isfinite(bounds.estimate)
    assert np.isfinite(bounds.margin_of_error)

@pytest.fixture
def edge_case_dataset():
    """Create a dataset with various edge cases."""
    return pd.DataFrame({
        'value': [
            np.nan,              # Missing value
            np.inf,              # Infinity
            -np.inf,             # Negative infinity
            1e-10,              # Very small positive
            -1e-10,             # Very small negative
            1e10,               # Very large positive
            -1e10,              # Very large negative
            0,                  # Zero
            1,                  # One
            -1                  # Negative one
        ],
        'category': ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A'],
        'weight': [1.0, 0.5, 0.0, np.nan, 1.0, 1.0, 0.5, 0.0, 1.0, 1.0]
    })