"""
Comprehensive benchmark suite for the Approximate Query Engine.
Generates reproducible benchmarks with detailed metrics and visualizations.
"""

import pandas as pd
import numpy as np
import time
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Tuple

from approx_query_engine import ApproxQueryEngine
from synthetic_data import generate_sales_data

class BenchmarkRunner:
    def __init__(self, 
                 data_sizes: List[int] = [10_000, 100_000, 1_000_000, 10_000_000],
                 confidence_levels: List[float] = [0.90, 0.95, 0.99],
                 runs_per_test: int = 5):
        self.data_sizes = data_sizes
        self.confidence_levels = confidence_levels
        self.runs_per_test = runs_per_test
        self.results_dir = Path("benchmarks/results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def run_all_benchmarks(self):
        """Run complete benchmark suite and generate reports."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results = {
            "performance": self.run_performance_benchmarks(),
            "accuracy": self.run_accuracy_benchmarks(),
            "scalability": self.run_scalability_benchmarks()
        }
        
        # Save raw results
        for benchmark_type, data in results.items():
            df = pd.DataFrame(data)
            df.to_csv(self.results_dir / f"{benchmark_type}_{timestamp}.csv", index=False)
        
        # Generate visualizations
        self.generate_plots(results, timestamp)
        
        return results
    
    def run_performance_benchmarks(self) -> List[Dict]:
        """Compare exact vs approximate query performance."""
        results = []
        test_queries = {
            "simple_count": "SELECT COUNT(*) FROM sales",
            "distinct_count": "SELECT COUNT(DISTINCT user_id) FROM sales",
            "group_by": "SELECT category, AVG(price) FROM sales GROUP BY category"
        }
        
        for size in self.data_sizes:
            df = generate_sales_data(size)
            engine = ApproxQueryEngine(df)
            
            for query_name, query in test_queries.items():
                # Exact query timing
                start = time.time()
                exact_result = engine.run_query(query)
                exact_time = time.time() - start
                
                # Approximate query timing (95% confidence)
                approx_query = f"APPROXIMATE {query} WITH ACCURACY 95%"
                start = time.time()
                approx_result = engine.run_query(approx_query)
                approx_time = time.time() - start
                
                results.append({
                    "dataset_size": size,
                    "query_type": query_name,
                    "exact_time": exact_time,
                    "approx_time": approx_time,
                    "speedup": exact_time / approx_time
                })
        
        return results
    
    def run_accuracy_benchmarks(self) -> List[Dict]:
        """Measure error rates and confidence interval accuracy."""
        results = []
        size = 1_000_000  # Fixed size for accuracy tests
        df = generate_sales_data(size)
        engine = ApproxQueryEngine(df)
        
        test_query = "SELECT AVG(price) FROM sales"
        exact_result = engine.run_query(test_query)
        
        for confidence in self.confidence_levels:
            for _ in range(self.runs_per_test):
                approx_query = f"APPROXIMATE {test_query} WITH ACCURACY {confidence*100}%"
                result = engine.run_query(approx_query)
                
                error = abs(result - exact_result) / exact_result
                results.append({
                    "confidence_level": confidence,
                    "relative_error": error,
                    "within_bounds": error <= (1 - confidence)
                })
        
        return results
    
    def run_scalability_benchmarks(self) -> List[Dict]:
        """Measure how performance scales with dataset size."""
        results = []
        query = "SELECT category, AVG(price) FROM sales GROUP BY category"
        
        for size in self.data_sizes:
            df = generate_sales_data(size)
            engine = ApproxQueryEngine(df)
            
            # Memory usage before query
            start_mem = engine.get_memory_usage()
            
            # Time both exact and approximate
            start = time.time()
            engine.run_query(query)
            exact_time = time.time() - start
            
            start = time.time()
            engine.run_query(f"APPROXIMATE {query} WITH ACCURACY 95%")
            approx_time = time.time() - start
            
            # Memory usage after query
            end_mem = engine.get_memory_usage()
            
            results.append({
                "dataset_size": size,
                "exact_time": exact_time,
                "approx_time": approx_time,
                "memory_used": end_mem - start_mem
            })
        
        return results
    
    def generate_plots(self, results: Dict, timestamp: str):
        """Generate visualization plots from benchmark results."""
        # Performance comparison plot
        plt.figure(figsize=(10, 6))
        df = pd.DataFrame(results["performance"])
        sns.barplot(data=df, x="query_type", y="speedup", hue="dataset_size")
        plt.title("Performance Speedup: Approximate vs Exact Queries")
        plt.ylabel("Speedup Factor (x times faster)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.results_dir / f"performance_comparison_{timestamp}.png")
        
        # Error distribution plot
        plt.figure(figsize=(10, 6))
        df = pd.DataFrame(results["accuracy"])
        sns.boxplot(data=df, x="confidence_level", y="relative_error")
        plt.title("Error Distribution by Confidence Level")
        plt.ylabel("Relative Error")
        plt.tight_layout()
        plt.savefig(self.results_dir / f"error_distribution_{timestamp}.png")
        
        # Scalability plot
        plt.figure(figsize=(10, 6))
        df = pd.DataFrame(results["scalability"])
        plt.plot(df["dataset_size"], df["exact_time"], label="Exact Query", marker="o")
        plt.plot(df["dataset_size"], df["approx_time"], label="Approximate Query", marker="o")
        plt.xscale("log")
        plt.yscale("log")
        plt.title("Query Time vs Dataset Size")
        plt.xlabel("Dataset Size (rows)")
        plt.ylabel("Query Time (seconds)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.results_dir / f"scalability_{timestamp}.png")

if __name__ == "__main__":
    runner = BenchmarkRunner()
    results = runner.run_all_benchmarks()
    print("Benchmarks completed. Results saved in benchmarks/results/")