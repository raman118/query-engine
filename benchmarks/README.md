# Benchmarks

This directory contains benchmarking scripts and results for the Approximate Query Engine.

## Running Benchmarks

To run all benchmarks:
```bash
python -m benchmarks.run_all
```

This will:
1. Generate synthetic datasets of various sizes
2. Run exact and approximate queries
3. Measure performance metrics
4. Generate plots and CSV results

## Benchmark Types

### 1. Performance Comparison
Compares execution time between exact and approximate queries across different query types:
- Simple aggregations (COUNT, SUM, AVG)
- Distinct count operations
- Group by queries
- Complex multi-aggregation queries

### 2. Error Analysis
Measures approximate query accuracy:
- Error rates vs confidence intervals
- Distribution of errors
- Impact of sample size on accuracy

### 3. Scalability Tests
Evaluates performance scaling with:
- Dataset size (10K to 10M rows)
- Query complexity
- Memory usage

## Results

See [benchmark_results.md](./benchmark_results.md) for detailed results and analysis.

### Key Findings
- Approximate queries are 10-100x faster than exact queries
- 95% of results fall within specified error bounds
- Linear memory scaling with dataset size
- Sub-linear time scaling for approximate queries