"""Tests for error bounds estimation methods.

This module contains comprehensive tests for all error estimation methods
including parametric and non-parametric approaches.
"""

import numpy as np
import pandas as pd
import pytest
from sampling import SimpleRandomSampler, ErrorBounds

def test_normal_distribution_bounds():
    """Test error bounds calculation using normal distribution."""
    sampler = SimpleRandomSampler(confidence_level=0.95, error_method='normal')
    data = pd.DataFrame({'value': np.random.normal(100, 10, 1000)})
    bounds = sampler.estimate_error_bounds(data, 'value')
    
    assert isinstance(bounds, ErrorBounds)
    assert bounds.lower_bound < bounds.estimate < bounds.upper_bound
    assert abs(bounds.margin_of_error) > 0
    assert 0 < bounds.relative_error < 1

def test_student_t_bounds():
    """Test error bounds calculation using Student's t-distribution."""
    sampler = SimpleRandomSampler(confidence_level=0.95, error_method='student_t')
    data = pd.DataFrame({'value': np.random.normal(100, 10, 30)})  # Smaller sample
    bounds = sampler.estimate_error_bounds(data, 'value')
    
    assert isinstance(bounds, ErrorBounds)
    assert bounds.lower_bound < bounds.estimate < bounds.upper_bound
    # t-distribution should give wider bounds than normal for small samples
    normal_sampler = SimpleRandomSampler(confidence_level=0.95, error_method='normal')
    normal_bounds = normal_sampler.estimate_error_bounds(data, 'value')
    assert bounds.margin_of_error > normal_bounds.margin_of_error

def test_bootstrap_bounds():
    """Test error bounds calculation using bootstrap method."""
    sampler = SimpleRandomSampler(confidence_level=0.95, error_method='bootstrap',
                                n_bootstrap=1000)
    # Use skewed distribution to test non-parametric capabilities
    data = pd.DataFrame({'value': np.random.lognormal(0, 1, 100)})
    bounds = sampler.estimate_error_bounds(data, 'value')
    
    assert isinstance(bounds, ErrorBounds)
    assert bounds.lower_bound < bounds.estimate < bounds.upper_bound
    
    # Test reproducibility with fixed seed
    np.random.seed(42)
    bounds1 = sampler.estimate_error_bounds(data, 'value')
    np.random.seed(42)
    bounds2 = sampler.estimate_error_bounds(data, 'value')
    assert bounds1.lower_bound == bounds2.lower_bound
    assert bounds1.upper_bound == bounds2.upper_bound

def test_non_parametric_bounds():
    """Test error bounds calculation using non-parametric method."""
    sampler = SimpleRandomSampler(confidence_level=0.95, error_method='non_parametric')
    # Use mixture of distributions to test robustness
    data1 = np.random.normal(0, 1, 50)
    data2 = np.random.exponential(2, 50)
    data = pd.DataFrame({'value': np.concatenate([data1, data2])})
    bounds = sampler.estimate_error_bounds(data, 'value')
    
    assert isinstance(bounds, ErrorBounds)
    assert bounds.lower_bound < bounds.estimate < bounds.upper_bound

def test_error_methods_edge_cases():
    """Test error bound methods with edge cases."""
    sampler = SimpleRandomSampler(confidence_level=0.95)
    
    # Empty data
    empty_data = pd.DataFrame({'value': []})
    for method in ['normal', 'student_t', 'bootstrap', 'non_parametric']:
        sampler.error_method = method
        bounds = sampler.estimate_error_bounds(empty_data, 'value')
        assert bounds.lower_bound == bounds.upper_bound == 0
    
    # Single value
    single_data = pd.DataFrame({'value': [42.0]})
    for method in ['normal', 'student_t', 'bootstrap', 'non_parametric']:
        sampler.error_method = method
        bounds = sampler.estimate_error_bounds(single_data, 'value')
        assert bounds.lower_bound == bounds.upper_bound == 42.0
    
    # Data with NaN/Inf
    bad_data = pd.DataFrame({
        'value': [1.0, np.nan, 2.0, np.inf, 3.0, -np.inf]
    })
    for method in ['normal', 'student_t', 'bootstrap', 'non_parametric']:
        sampler.error_method = method
        bounds = sampler.estimate_error_bounds(bad_data, 'value')
        assert np.isfinite(bounds.lower_bound)
        assert np.isfinite(bounds.upper_bound)
        assert bounds.lower_bound <= bounds.estimate <= bounds.upper_bound

def test_finite_population_correction():
    """Test that finite population correction is properly applied."""
    sampler = SimpleRandomSampler(confidence_level=0.95, error_method='normal')
    data = pd.DataFrame({'value': np.random.normal(100, 10, 100)})
    
    # Without FPC
    bounds1 = sampler.estimate_error_bounds(data, 'value')
    # With FPC
    bounds2 = sampler.estimate_error_bounds(data, 'value', population_size=150)
    
    # FPC should reduce the margin of error
    assert bounds2.margin_of_error < bounds1.margin_of_error
    
    # Full population sample should have no error
    bounds3 = sampler.estimate_error_bounds(data, 'value', population_size=100)
    assert np.isclose(bounds3.margin_of_error, 0, atol=1e-10)

def test_custom_estimator():
    """Test bootstrap with custom estimator function."""
    sampler = SimpleRandomSampler(confidence_level=0.95, error_method='bootstrap')
    data = pd.DataFrame({'value': np.random.normal(100, 10, 100)})
    
    def median_estimator(x):
        return np.median(x)
    
    # Get bootstrap bounds for both mean and median
    mean_bounds = sampler.estimate_error_bounds(data, 'value')
    median_bounds = sampler._bootstrap_estimate(
        data['value'].to_numpy(),
        estimator=median_estimator
    )
    
    # Bounds should be different for different estimators
    assert not np.allclose(mean_bounds.lower_bound, median_bounds[0])
    assert not np.allclose(mean_bounds.upper_bound, median_bounds[1])