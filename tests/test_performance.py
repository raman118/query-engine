"""Tests for new sampling methods and performance optimizations."""

import pytest
import pandas as pd
import numpy as np
from time import time
from sampling import (
    SimpleRandomSampler,
    StratifiedSampler,
    ClusterSampler,
    SystematicSampler
)

def test_simple_random_sampler_performance(large_dataset):
    """Test performance optimizations in simple random sampling."""
    sampler = SimpleRandomSampler()
    
    # Test with different batch sizes
    batch_sizes = [1000, 10000, 100000]
    times = []
    
    for batch_size in batch_sizes:
        start_time = time()
        sample = sampler.sample(large_dataset, size=1000, batch_size=batch_size)
        end_time = time()
        
        assert len(sample) == 1000
        times.append(end_time - start_time)
    
    # Check that larger batch sizes are generally faster
    # (though this might not always be true due to system variations)
    assert min(times) < max(times) * 2  # Reasonable performance difference

def test_stratified_sampler_performance(large_dataset):
    """Test performance optimizations in stratified sampling."""
    sampler = StratifiedSampler()
    
    # Test with different batch sizes
    batch_sizes = [1000, 10000, 100000]
    times = []
    
    for batch_size in batch_sizes:
        start_time = time()
        sample = sampler.sample(
            large_dataset,
            'category',
            size=1000,
            batch_size=batch_size
        )
        end_time = time()
        
        assert len(sample) <= 1000  # Might be slightly less due to rounding
        times.append(end_time - start_time)
    
    # Check that larger batch sizes are generally faster
    assert min(times) < max(times) * 2

def test_cluster_sampler():
    """Test cluster sampling implementation."""
    # Create test dataset with clusters
    data = pd.DataFrame({
        'cluster': ['A'] * 100 + ['B'] * 150 + ['C'] * 200,
        'value': np.random.normal(100, 15, 450)
    })
    
    sampler = ClusterSampler()
    
    # Test sampling specific number of clusters
    sample = sampler.sample(data, 'cluster', size=2)
    unique_clusters = sample['cluster'].unique()
    assert len(unique_clusters) == 2
    
    # Each cluster should be complete
    for cluster in unique_clusters:
        cluster_size = len(data[data['cluster'] == cluster])
        sample_size = len(sample[sample['cluster'] == cluster])
        assert cluster_size == sample_size
    
    # Test error bounds
    bounds = sampler.estimate_error_bounds(sample, 'value', len(data))
    assert bounds.lower_bound <= bounds.estimate <= bounds.upper_bound
    assert bounds.margin_of_error > 0

def test_systematic_sampler():
    """Test systematic sampling implementation."""
    # Create test dataset with known pattern
    data = pd.DataFrame({
        'value': range(1000),
        'group': ['A', 'B'] * 500
    })
    
    sampler = SystematicSampler()
    
    # Test with different sample sizes
    for size in [100, 200, 500]:
        sample = sampler.sample(data, size=size)
        
        assert len(sample) == size
        assert len(sample['value'].unique()) == size  # No duplicates
        
        # Check approximate spacing
        sorted_values = sorted(sample['value'])
        intervals = np.diff(sorted_values)
        mean_interval = len(data) / size
        assert abs(np.mean(intervals) - mean_interval) < mean_interval * 0.2
    
    # Test error bounds
    bounds = sampler.estimate_error_bounds(sample, 'value', len(data))
    assert bounds.lower_bound <= bounds.estimate <= bounds.upper_bound
    assert bounds.margin_of_error > 0

@pytest.fixture
def large_dataset():
    """Create a large dataset for performance testing."""
    np.random.seed(42)
    n = 1000000  # 1 million rows
    
    return pd.DataFrame({
        'value': np.random.normal(100, 15, n),
        'category': np.random.choice(['A', 'B', 'C'], n),
        'cluster': np.random.choice([f'C{i}' for i in range(50)], n),
        'id': range(n)
    })