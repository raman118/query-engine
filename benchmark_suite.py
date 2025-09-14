import pandas as pd
import numpy as np
from approx_query_engine import ApproxQueryEngine
from synthetic_data import generate_sales_data
import time
import json
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class BenchmarkSuite:
    def __init__(self, data_size: int = 1_000_000):
        self.data_size = data_size
        self.results = {}
        self.df = generate_sales_data(data_size)
        self.engine = ApproxQueryEngine(self.df)
        
    def run_benchmarks(self):
        """Run comprehensive benchmarks on all supported query types."""
        test_queries = {
            'Simple COUNT': "SELECT COUNT(*) FROM sales",
            'DISTINCT COUNT': "SELECT COUNT(DISTINCT user_id) FROM sales",
            'SUM': "SELECT SUM(price) FROM sales",
            'AVG': "SELECT AVG(price) FROM sales",
            'GROUP BY': "SELECT category, COUNT(*) FROM sales GROUP BY category",
            'Complex': "SELECT category, COUNT(*), AVG(price), SUM(price) FROM sales GROUP BY category"
        }
        
        print("Running benchmarks...")
        for name, query in test_queries.items():
            print(f"\nTesting: {name}")
            result = self._benchmark_query(query)
            self.results[name] = result
            
        self._save_results()
        self._plot_results()
        
    def _benchmark_query(self, query: str, runs: int = 5) -> dict:
        """Benchmark a single query with multiple runs."""
        
        # Exact query timing
        exact_times = []
        for _ in range(runs):
            start = time.time()
            exact_result = self.engine.execute_sql(query, approximate=False)
            exact_times.append(time.time() - start)
        
        # Approximate query timing
        approx_times = []
        approx_results = []
        for _ in range(runs):
            start = time.time()
            result = self.engine.execute_sql(query, approximate=True)
            approx_times.append(time.time() - start)
            approx_results.append(result)
        
        # Calculate error rate
        error_rate = self._calculate_error(exact_result, approx_results[-1])
        
        return {
            'exact_time': np.mean(exact_times),
            'approx_time': np.mean(approx_times),
            'speedup': np.mean(exact_times) / np.mean(approx_times),
            'error_rate': error_rate
        }
    
    def _calculate_error(self, exact, approx) -> float:
        """Calculate error rate between exact and approximate results."""
        if isinstance(exact, pd.DataFrame):
            # For GROUP BY queries
            error_rates = []
            for col in exact.select_dtypes(include=[np.number]).columns:
                error = np.abs(exact[col] - approx[col]) / exact[col]
                error_rates.append(error.mean())
            return np.mean(error_rates) * 100
        else:
            # For simple aggregations
            return abs(exact - approx) / exact * 100
    
    def _save_results(self):
        """Save benchmark results to file."""
        output_dir = Path('benchmark_results')
        output_dir.mkdir(exist_ok=True)
        
        # Save detailed JSON results
        with open(output_dir / 'benchmark_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Create markdown table
        table_data = []
        for query_type, result in self.results.items():
            table_data.append([
                query_type,
                f"{result['speedup']:.1f}x",
                f"{result['error_rate']:.1f}%",
                f"{result['approx_time']*1000:.1f}ms"
            ])
        
        # Save markdown table
        table_headers = ['Query Type', 'Speedup', 'Error Rate', 'Response Time']
        markdown_table = tabulate(table_data, headers=table_headers, tablefmt='pipe')
        with open(output_dir / 'benchmark_results.md', 'w') as f:
            f.write("# Benchmark Results\n\n")
            f.write(markdown_table)
    
    def _plot_results(self):
        """Generate visualization of benchmark results."""
        plt.figure(figsize=(12, 6))
        
        # Prepare data
        query_types = list(self.results.keys())
        speedups = [r['speedup'] for r in self.results.values()]
        error_rates = [r['error_rate'] for r in self.results.values()]
        
        # Plot speedup bars
        ax1 = plt.gca()
        bars = ax1.bar(query_types, speedups, color='skyblue')
        ax1.set_ylabel('Speedup Factor (x)', color='skyblue')
        ax1.tick_params(axis='y', labelcolor='skyblue')
        
        # Add error rate line
        ax2 = ax1.twinx()
        line = ax2.plot(query_types, error_rates, color='red', marker='o')
        ax2.set_ylabel('Error Rate (%)', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        # Customize plot
        plt.title('Query Performance vs Accuracy')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        plt.savefig('benchmark_results/performance_comparison.png')
        plt.close()

if __name__ == '__main__':
    # Run benchmarks
    suite = BenchmarkSuite()
    suite.run_benchmarks()