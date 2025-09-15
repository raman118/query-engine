"""Tests for parameter validation and configuration handling.

This module contains tests that verify proper validation of parameters
and configuration options, ensuring robust error handling.
"""

import numpy as np
import pandas as pd
import pytest
from numpy.random import RandomState, Generator
from sampling import (
    SamplingConfig, DistributionType, SimpleRandomSampler,
    DEFAULT_CONFIDENCE, DEFAULT_BOOTSTRAP_ITERATIONS, DEFAULT_BATCH_SIZE
)

def test_distribution_type_conversion():
    """Test string to DistributionType conversion."""
    assert DistributionType.from_str('normal') == DistributionType.NORMAL
    assert DistributionType.from_str('STUDENT_T') == DistributionType.STUDENT_T
    assert DistributionType.from_str('bootstrap') == DistributionType.BOOTSTRAP
    assert DistributionType.from_str('NON_PARAMETRIC') == DistributionType.NON_PARAMETRIC
    
    with pytest.raises(ValueError):
        DistributionType.from_str('invalid')

def test_sampling_config_defaults():
    """Test default values in SamplingConfig."""
    config = SamplingConfig()
    assert config.confidence_level == DEFAULT_CONFIDENCE
    assert config.error_method == 'normal'
    assert config.n_bootstrap == DEFAULT_BOOTSTRAP_ITERATIONS
    assert config.batch_size == DEFAULT_BATCH_SIZE
    assert config.random_state is None

def test_sampling_config_validation():
    """Test validation of SamplingConfig parameters."""
    # Invalid confidence level
    with pytest.raises(ValueError):
        SamplingConfig(confidence_level=0)
    with pytest.raises(ValueError):
        SamplingConfig(confidence_level=1)
    with pytest.raises(ValueError):
        SamplingConfig(confidence_level=-0.5)
        
    # Invalid error method
    with pytest.raises(ValueError):
        SamplingConfig(error_method='invalid')
        
    # Invalid bootstrap iterations
    with pytest.raises(ValueError):
        SamplingConfig(n_bootstrap=0)
    with pytest.raises(ValueError):
        SamplingConfig(n_bootstrap=-100)
        
    # Invalid batch size
    with pytest.raises(ValueError):
        SamplingConfig(batch_size=0)
    with pytest.raises(ValueError):
        SamplingConfig(batch_size=-1000)

def test_random_state_handling():
    """Test different types of random state initialization."""
    # Test with integer seed
    config1 = SamplingConfig(random_state=42)
    sampler1 = SimpleRandomSampler(config1)
    
    # Test with numpy RandomState
    rs = RandomState(42)
    config2 = SamplingConfig(random_state=rs)
    sampler2 = SimpleRandomSampler(config2)
    
    # Test with numpy Generator
    rg = np.random.default_rng(42)
    config3 = SamplingConfig(random_state=rg)
    sampler3 = SimpleRandomSampler(config3)
    
    # Generate samples and verify they're different (random state works)
    data = pd.DataFrame({'value': range(1000)})
    sample1 = sampler1.sample(data, size=100)
    sample2 = sampler2.sample(data, size=100)
    sample3 = sampler3.sample(data, size=100)
    
    assert not sample1.equals(sample2)
    assert not sample1.equals(sample3)
    assert not sample2.equals(sample3)

def test_error_bounds_validation():
    """Test validation of error bounds parameters."""
    sampler = SimpleRandomSampler()
    data = pd.DataFrame({'value': range(100)})
    sample = sampler.sample(data, size=10)
    
    # Test with different error methods
    for method in ['normal', 'student_t', 'bootstrap', 'non_parametric']:
        sampler.error_method = method
        bounds = sampler.estimate_error_bounds(sample, 'value')
        assert bounds.method == method
        assert bounds.lower_bound <= bounds.estimate <= bounds.upper_bound
        
    # Test with invalid error method
    with pytest.raises(ValueError):
        sampler.error_method = 'invalid'
        sampler.estimate_error_bounds(sample, 'value')
        
    # Test with invalid confidence level
    with pytest.raises(ValueError):
        sampler = SimpleRandomSampler(confidence_level=2.0)
        
    # Test with invalid population size
    with pytest.raises(ValueError):
        sampler = SimpleRandomSampler()
        sampler.estimate_error_bounds(sample, 'value', population_size=5)  # < sample size

def test_sampling_reproducibility():
    """Test that sampling is reproducible with fixed random state."""
    config = SamplingConfig(random_state=42)
    sampler1 = SimpleRandomSampler(config)
    sampler2 = SimpleRandomSampler(config)
    
    data = pd.DataFrame({'value': range(1000)})
    sample1 = sampler1.sample(data, size=100)
    sample2 = sampler2.sample(data, size=100)
    
    # Same random state should produce identical samples
    pd.testing.assert_frame_equal(sample1, sample2)
    
    # Different random states should produce different samples
    config_different = SamplingConfig(random_state=43)
    sampler3 = SimpleRandomSampler(config_different)
    sample3 = sampler3.sample(data, size=100)
    
    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(sample1, sample3)