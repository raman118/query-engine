"""Tests for the sampling module.

This module contains comprehensive tests for the sampling module, including:
- Error bounds validation
- Simple random sampling
- Stratified sampling with different allocation methods
- Reservoir sampling for streaming data
- Edge cases and numerical stability
"""

import pytest
import pandas as pd
import numpy as np
from sampling import (
    SimpleRandomSampler,
    StratifiedSampler,
    ReservoirSampler,
    ErrorBounds
)

def test_error_bounds_properties():
    """Test ErrorBounds class properties."""
    bounds = ErrorBounds(
        estimate=100.0,
        lower_bound=95.0,
        upper_bound=105.0,
        confidence_level=0.95,
        sample_size=1000,
        population_size=10000
    )
    
    assert bounds.margin_of_error == 5.0
    assert bounds.relative_error == 0.05
    
    # Test zero estimate case
    zero_bounds = ErrorBounds(
        estimate=0.0,
        lower_bound=-1.0,
        upper_bound=1.0,
        confidence_level=0.95,
        sample_size=100
    )
    assert zero_bounds.relative_error == float('inf')

def test_error_bounds_validation():
    """Test error bounds validation."""
    # Test invalid confidence level
    with pytest.raises(ValueError, match="Confidence level must be between 0 and 1"):
        SimpleRandomSampler(confidence_level=1.5)
    
    with pytest.raises(ValueError, match="Confidence level must be between 0 and 1"):
        SimpleRandomSampler(confidence_level=0)

def test_simple_random_sampler(large_dataset):
    """Test simple random sampling with error bounds."""
    sampler = SimpleRandomSampler(confidence_level=0.95)
    
    # Test sample size calculation
    size = sampler.required_sample_size(error_margin=0.05, population_size=len(large_dataset))
    assert 0 < size < len(large_dataset)
    
    # Test sampling with fixed size
    sample = sampler.sample(large_dataset, size=1000)
    assert len(sample) == 1000
    assert not sample.empty
    assert set(sample.columns) == set(large_dataset.columns)
    
    # Test sampling with error margin
    sample = sampler.sample(large_dataset, error_margin=0.05)
    assert len(sample) > 0
    
    # Test error bounds calculation
    bounds = sampler.estimate_error_bounds(sample, 'value')
    assert bounds.lower_bound <= bounds.estimate <= bounds.upper_bound
    assert 0.9 <= bounds.confidence_level <= 1.0
    assert len(sample) == size
    
    # Test error bounds
    bounds = sampler.estimate_error_bounds(sample, 'value', len(large_dataset))
    assert isinstance(bounds, ErrorBounds)
    assert bounds.lower_bound < bounds.estimate < bounds.upper_bound
    assert abs(bounds.estimate - large_dataset['value'].mean()) < 10

def test_stratified_sampler(large_dataset):
    """Test stratified sampling with different allocation methods."""
    sampler = StratifiedSampler(confidence_level=0.95)
    
    for allocation in ['proportional', 'equal', 'neyman']:
        sample = sampler.sample(
            large_dataset,
            strata_column='category',
            size=1000,
            allocation=allocation
        )
        
        # Check sample size
        assert len(sample) <= 1000
        
        # Check all strata are represented
        assert set(sample['category']) == set(large_dataset['category'])
        
        # For proportional allocation, check stratum proportions
        if allocation == 'proportional':
            orig_props = large_dataset.groupby('category').size() / len(large_dataset)
            sample_props = sample.groupby('category').size() / len(sample)
            
            # Allow for small random variations
            assert all(abs(orig_props - sample_props) < 0.1)

def test_reservoir_sampling_streaming(large_dataset):
    """Test reservoir sampling with streaming data."""
    sample_size = 1000
    sampler = ReservoirSampler(size=sample_size, confidence_level=0.95)
    
    # Split data into chunks to simulate streaming
    chunk_size = 1000
    for i in range(0, len(large_dataset), chunk_size):
        chunk = large_dataset.iloc[i:i+chunk_size]
        sampler.add(chunk)
    
    # Get final sample
    final_sample = sampler.get_sample()
    assert len(final_sample) <= sample_size
    
    # Test error bounds
    bounds = sampler.estimate_error_bounds(final_sample, 'value')
    assert isinstance(bounds, ErrorBounds)
    assert bounds.lower_bound < bounds.estimate < bounds.upper_bound
    
    # The sample mean should be close to population mean
    assert abs(bounds.estimate - large_dataset['value'].mean()) < 10

def test_edge_cases(edge_case_dataset):
    """Test sampling with edge case data."""
    sampler = SimpleRandomSampler(confidence_level=0.95)
    
    # Test with small dataset
    sample = sampler.sample(edge_case_dataset, size=5)
    assert len(sample) == 5
    
    # Test error bounds with filtered data
    clean_data = edge_case_dataset.dropna()
    clean_data = clean_data[~np.isinf(clean_data['value'])]
    
    bounds = sampler.estimate_error_bounds(
        clean_data,
        'value',
        population_size=len(edge_case_dataset)
    )
    assert isinstance(bounds, ErrorBounds)
    assert np.isfinite(bounds.estimate)
    assert np.isfinite(bounds.margin_of_error)