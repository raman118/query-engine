"""Demo script showing how to use the sampling module."""

from sampling import SimpleRandomSampler, StratifiedSampler, SamplingConfig
import pandas as pd
import numpy as np

def demo_simple_random():
    """Demonstrate simple random sampling."""
    print("\n=== Simple Random Sampling Demo ===")
    
    # create example data
    np.random.seed(42)
    data = pd.DataFrame({
        'value': np.random.normal(100, 15, 10000),
        'category': np.random.choice(['A', 'B', 'C'], 10000)
    })
    
    # create sampler with bootstrap error estimation
    config = SamplingConfig(
        confidence_level=0.95,
        error_method='bootstrap',
        n_bootstrap=1000
    )
    sampler = SimpleRandomSampler(config)
    
    # Take sample and calculate error bounds
    sample = sampler.sample(data, size=1000)
    bounds = sampler.estimate_error_bounds(sample, 'value')
    
    # Print results
    print(f"\nTrue population mean: {data['value'].mean():.2f}")
    print(f"Sample mean: {bounds.estimate:.2f}")
    print(f"95% CI: [{bounds.lower_bound:.2f}, {bounds.upper_bound:.2f}]")
    print(f"Margin of error: ±{bounds.margin_of_error:.2f}")
    print(f"Relative error: {bounds.relative_error:.2%}")

def demo_stratified():
    """Demonstrate stratified sampling."""
    print("\n=== Stratified Sampling Demo ===")
    
    # Create example data with different distributions per category
    np.random.seed(42)
    data = pd.DataFrame({
        'category': np.repeat(['A', 'B', 'C'], [5000, 3000, 2000]),
        'value': np.concatenate([
            np.random.normal(100, 10, 5000),  # Category A
            np.random.normal(150, 20, 3000),  # Category B
            np.random.normal(80, 5, 2000)     # Category C
        ])
    })
    
    # Create stratified sampler
    config = SamplingConfig(
        confidence_level=0.95,
        error_method='student_t'
    )
    sampler = StratifiedSampler(config)
    
    # Take stratified sample
    sample = sampler.sample(
        data,
        strata_column='category',
        size=1000,
        allocation='optimal'
    )
    bounds = sampler.estimate_error_bounds(
        sample, 'value', strata_column='category'
    )
    
    # Print results per stratum
    print("\nPopulation statistics per category:")
    print(data.groupby('category')['value'].agg(['count', 'mean', 'std']))
    
    print("\nSample statistics per category:")
    print(sample.groupby('category')['value'].agg(['count', 'mean', 'std']))
    
    print("\nOverall estimates:")
    print(f"Sample mean: {bounds.estimate:.2f}")
    print(f"95% CI: [{bounds.lower_bound:.2f}, {bounds.upper_bound:.2f}]")
    print(f"Margin of error: ±{bounds.margin_of_error:.2f}")
    print(f"Relative error: {bounds.relative_error:.2%}")

if __name__ == "__main__":
    demo_simple_random()
    demo_stratified()