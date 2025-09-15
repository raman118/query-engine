"""Tests for sampling parameter validation and configuration.

This module contains tests that verify proper validation of parameters
and configuration options, ensuring robust error handling.
"""

import numpy as np
import pandas as pd
import pytest
from numpy.random import RandomState, Generator
from sampling import (
    SamplingConfig, DistributionType, SimpleRandomSampler,
    ErrorBounds, DEFAULT_CONFIDENCE, DEFAULT_BOOTSTRAP_ITERATIONS,
    DEFAULT_BATCH_SIZE
)

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

def test_error_bounds_validation():
    """Test validation of error bounds parameters."""
    # Valid parameters
    bounds = ErrorBounds(
        estimate=100.0,
        lower_bound=95.0,
        upper_bound=105.0,
        confidence_level=0.95,
        sample_size=100,
        method='normal'
    )
    assert bounds.margin_of_error == 5.0
    assert bounds.relative_error == 0.05
    
    # Invalid confidence level
    with pytest.raises(ValueError):
        ErrorBounds(
            estimate=100.0,
            lower_bound=95.0,
            upper_bound=105.0,
            confidence_level=2.0,
            sample_size=100
        )
    
    # Invalid bounds (estimate outside bounds)
    with pytest.raises(ValueError):
        ErrorBounds(
            estimate=90.0,
            lower_bound=95.0,
            upper_bound=105.0,
            confidence_level=0.95,
            sample_size=100
        )
    
    # Invalid sample size
    with pytest.raises(ValueError):
        ErrorBounds(
            estimate=100.0,
            lower_bound=95.0,
            upper_bound=105.0,
            confidence_level=0.95,
            sample_size=-10
        )
    
    # Invalid population size
    with pytest.raises(ValueError):
        ErrorBounds(
            estimate=100.0,
            lower_bound=95.0,
            upper_bound=105.0,
            confidence_level=0.95,
            sample_size=100,
            population_size=50  # Less than sample size
        )

def test_distribution_type_conversion():
    """Test conversion between strings and DistributionType enum."""
    assert DistributionType.from_str('normal') == DistributionType.NORMAL
    assert DistributionType.from_str('student_t') == DistributionType.STUDENT_T
    assert DistributionType.from_str('bootstrap') == DistributionType.BOOTSTRAP
    assert DistributionType.from_str('non_parametric') == DistributionType.NON_PARAMETRIC
    
    with pytest.raises(ValueError):
        DistributionType.from_str('invalid_method')

def test_random_state_initialization():
    """Test random state initialization with different types."""
    # Test with int seed
    config1 = SamplingConfig(random_state=42)
    sampler1 = SimpleRandomSampler(config1)
    
    # Test with RandomState
    rs = RandomState(42)
    config2 = SamplingConfig(random_state=rs)
    sampler2 = SimpleRandomSampler(config2)
    
    # Test with Generator
    rg = np.random.default_rng(42)
    config3 = SamplingConfig(random_state=rg)
    sampler3 = SimpleRandomSampler(config3)
    
    # Generate samples
    data = pd.DataFrame({'value': range(1000)})
    sample1 = sampler1.sample(data, size=100)
    sample2 = sampler2.sample(data, size=100)
    sample3 = sampler3.sample(data, size=100)
    
    # Each sampler should give different results even with same seed
    # due to different RNG implementations
    assert not sample1.equals(sample2)
    assert not sample1.equals(sample3)
    assert not sample2.equals(sample3)

def test_sampling_reproducibility():
    """Test that sampling is reproducible with same config."""
    config = SamplingConfig(random_state=42)
    sampler1 = SimpleRandomSampler(config)
    sampler2 = SimpleRandomSampler(config)
    
    data = pd.DataFrame({'value': range(1000)})
    sample1 = sampler1.sample(data, size=100)
    sample2 = sampler2.sample(data, size=100)
    
    # Same configuration should produce identical samples
    pd.testing.assert_frame_equal(sample1, sample2)

def test_batch_processing():
    """Test that batch processing works correctly."""
    config = SamplingConfig(batch_size=100)
    sampler = SimpleRandomSampler(config)
    
    # Create data larger than batch size
    data = pd.DataFrame({'value': range(1000)})
    sample = sampler.sample(data, size=250)
    
    assert len(sample) == 250
    assert len(sample['value'].unique()) == 250  # No duplicates