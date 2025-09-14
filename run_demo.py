from approx_query_engine import ApproxQueryEngine
from synthetic_data import generate_sales_data     
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Union, List, Dict, Any

def calculate_error(exact_result: Union[pd.DataFrame, float, int], 
                   approx_result: Union[pd.DataFrame, float, int]) -> float:
    """Calculate error rate between exact and approximate results."""
    if isinstance(exact_result, (int, float)):
        if not isinstance(approx_result, (int, float)):
            return 100.0  # Maximum error if types don't match
        return float(abs(exact_result - approx_result) / exact_result * 100)
    elif isinstance(exact_result, pd.DataFrame):
        if not isinstance(approx_result, pd.DataFrame):
            return 100.0  # Maximum error if types don't match
        errors: List[float] = []
        for col in exact_result.select_dtypes(include=['number']).columns:
            exact_col = exact_result[col].to_numpy()
            approx_col = approx_result[col].to_numpy()
            with np.errstate(divide='ignore', invalid='ignore'):
                col_errors = abs(exact_col - approx_col) / exact_col * 100
            col_error = float(np.nanmean(col_errors))  # Handle division by zero
            errors.append(col_error)
        return float(np.mean(errors)) if errors else 0.0
    return 0.0

def run_benchmark(streaming: bool = False) -> None:
    """Run benchmarks comparing exact and approximate query execution."""
    print("=== e6data Approximate Query Engine Demo ===")
    print("=" * 70)
    
    # Test data sizes
    sizes = [100_000, 1_000_000]
    results: List[Dict[str, Any]] = []
    
    for size in sizes:
        print(f"\nTesting with {size:,} rows")
        print("-" * 70)
        
        # Generate and load data
        df = generate_sales_data(size)
        
        if streaming:
            # In streaming mode, process data in chunks
            engine = ApproxQueryEngine()  # Start with empty engine
            chunk_size = size // 10
            
            # Process data in chunks, using concat since we don't have true streaming
            for i in range(0, size, chunk_size):
                chunk = df.iloc[i:i+chunk_size]
                if i == 0:
                    engine = ApproxQueryEngine(chunk)  # Initialize with first chunk
                else:
                    # Update data and cache
                    engine.df = pd.concat([engine.df, chunk], ignore_index=True)
                    engine._setup_cache()
                print(f"Processed {min(i+chunk_size, size):,} records...")
        else:
            # In batch mode, load all data at once
            engine = ApproxQueryEngine(df)
        
        # Test queries with varying complexity
        test_queries = [
            ("Simple Count", "SELECT COUNT(*) FROM sales;"),
            ("Distinct Count", "SELECT COUNT(DISTINCT user_id) FROM sales;"),
            ("Sum Aggregation", "SELECT SUM(price) FROM sales;"),
            ("Average", "SELECT AVG(price) FROM sales;"),
            ("Group By", "SELECT category, COUNT(*) FROM sales GROUP BY category;"),
            ("Complex", "SELECT category, COUNT(*), AVG(price), SUM(price) FROM sales GROUP BY category;")
        ]
        
        print("\nRunning benchmarks...")
        print("-" * 70)
        
        for name, query in test_queries:
            print(f"\nTest: {name}")
            print(f"Query: {query}")
            
            # Time exact query
            exact_result = None
            exact_time = 0
            try:
                start = time.time()
                exact_result = engine.execute_sql(query, approximate=False)
                exact_time = time.time() - start
                print(f"\nExact Results:")
                print(f"Time: {exact_time:.4f} seconds")
                print(exact_result)
            except Exception as e:
                print(f"❌ Error in exact query: {str(e)}")
                continue

            # Skip approximate query if exact query failed
            if exact_result is None:
                continue
            
            # Run and time approximate query
            approx_result = None
            approx_time = 0
            try:
                start = time.time()
                approx_result = engine.execute_sql(query, approximate=True)
                approx_time = time.time() - start
                
                speedup = exact_time / approx_time if approx_time > 0 else 0
                error = calculate_error(exact_result, approx_result)
                
                result = {
                    'size': size,
                    'query': name,
                    'exact_time': exact_time,
                    'approx_time': approx_time,
                    'speedup': speedup,
                    'error': error
                }
                results.append(result)
                
                print(f"\nApproximate Results:")
                print(f"Time: {approx_time:.4f} seconds ({speedup:.1f}x faster)")
                print(f"Error Rate: {error:.1f}%")
                print(approx_result)
                
                # Performance assessment
                if speedup >= 3:
                    print("✅ Exceeds 3x speed target")
                elif speedup > 1:
                    print("⚠️ Faster but below 3x target")
                else:
                    print("❌ Slower than exact query")
            except Exception as e:
                print(f"❌ Error in approximate query: {str(e)}")
                continue
    
    # Convert results to DataFrame for analysis
    results_df = pd.DataFrame(results)
    
    # Create results directory
    output_dir = Path('benchmark_results')
    output_dir.mkdir(exist_ok=True)
    
    # Plot performance comparison
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=results_df, x='query', y='speedup')
    plt.title('Query Performance Comparison')
    plt.xticks(rotation=45)
    plt.ylabel('Speedup Factor (vs Exact Query)')
    plt.axhline(y=3, color='r', linestyle='--', label='3x Target')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_comparison.png')
    plt.close()
    
    # Plot error rates
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=results_df, x='query', y='error')
    plt.title('Query Error Rates')
    plt.xticks(rotation=45)
    plt.ylabel('Error Rate (%)')
    plt.axhline(y=5, color='r', linestyle='--', label='5% Target')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'error_rates.png')
    plt.close()
    
    # Save detailed results
    results_df.to_csv(output_dir / 'benchmark_results.csv', index=False)
    
    # Summary statistics
    print("\n📊 Performance Summary:")
    print("-" * 70)
    summary = results_df.groupby('query').agg({
        'speedup': ['mean', 'min', 'max'],
        'error': ['mean', 'min', 'max']
    }).round(2)
    
    print("\nSpeedup Factors:")
    print(summary['speedup'])
    print("\nError Rates:")
    print(summary['error'])

def run_demo() -> None:
    """Run interactive demo of the query engine."""
    print("=== e6data Approximate Query Engine Demo ===\n")
    
    # Generate large dataset
    print("Generating sample dataset (1M rows)...")
    df = generate_sales_data(1_000_000)
    
    # Initialize engine
    print("Initializing query engine...")
    engine = ApproxQueryEngine(df)
    
    # Demo queries
    queries = [
        ("Simple COUNT", "SELECT COUNT(*) FROM sales;"),
        ("DISTINCT COUNT", "SELECT COUNT(DISTINCT user_id) FROM sales;"),
        ("Simple SUM", "SELECT SUM(price) FROM sales;"),
        ("Simple AVG", "SELECT AVG(price) FROM sales;"),
        ("Basic GROUP BY", "SELECT category, COUNT(*) FROM sales GROUP BY category;"),
        ("Complex GROUP BY", "SELECT category, COUNT(*) as count, AVG(price) as avg_price, SUM(price) as total_sales FROM sales GROUP BY category;")
    ]
    
    # Run each query and show results
    for name, query in queries:
        print(f"\n=== {name} Demo ===")
        print(f"Query: {query}\n")
        
        # Run exact query
        exact_result = None
        exact_time = 0
        try:
            start = time.time()
            exact_result = engine.execute_sql(query, approximate=False)
            exact_time = time.time() - start
            print(f"Exact Results:")
            print(exact_result)
        except Exception as e:
            print(f"❌ Error in exact query: {str(e)}")
            continue

        # Skip approximate query if exact query failed
        if exact_result is None:
            continue
        
        # Run approximate query
        approx_result = None
        approx_time = 0
        try:
            start = time.time()
            approx_result = engine.execute_sql(query, approximate=True)
            approx_time = time.time() - start
            
            # Calculate metrics
            speedup = exact_time / approx_time if approx_time > 0 else 0
            error = calculate_error(exact_result, approx_result)
            
            # Print results
            print("\nResults:")
            print("-" * 50)
            print(f"Exact Result:")
            print(exact_result)
            print(f"\nApproximate Result:")
            print(approx_result)
            print("\nPerformance:")
            print(f"Exact time:   {exact_time*1000:.2f}ms")
            print(f"Approx time:  {approx_time*1000:.2f}ms")
            print(f"Speedup:      {speedup:.1f}x")
            print(f"Error rate:   {error:.2f}%")
            
            # Performance assessment
            if speedup >= 3:
                print("✅ Exceeds 3x speed target")
            elif speedup > 1:
                print("⚠️ Faster but below 3x target")
            else:
                print("❌ Slower than exact query")
        except Exception as e:
            print(f"❌ Error in approximate query: {str(e)}")
            continue

if __name__ == "__main__":
    # Run both demo and benchmarks
    print("\n🎯 Interactive Demo")
    run_demo()
    
    print("\n📊 Full Benchmark Suite")
    print("\n🔄 Batch Mode Test")
    run_benchmark(streaming=False)
    
    print("\n🔄 Streaming Mode Test")
    run_benchmark(streaming=True)