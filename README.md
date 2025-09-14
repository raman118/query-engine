# Approximate Query Engine

A high-performance query engine that supports both exact and approximate query processing for large datasets. Uses probabilistic data structures and sampling techniques to provide fast, approximate results with guaranteed error bounds.

## Features

- **Fast Approximate Queries**: Get quick results with controllable accuracy-speed tradeoff
- **Confidence Intervals**: Every approximate result comes with error bounds
- **Streaming Support**: Process data in chunks for memory efficiency
- **Rich Query Support**: SQL-like syntax with aggregations and grouping
- **Data Sketches**: Uses HyperLogLog for efficient distinct counting

## Setup Instructions

1. Install requirements:
```bash
pip install pandas numpy
```

2. Project structure:
- `approx_query_engine.py` - Main query engine implementation
- `synthetic_data.py` - Test data generator
- `run_demo.py` - Demo and benchmarks
- `sketches.py` - Probabilistic data structures

## Running the Demo

1. Basic example:
```bash
python run_demo.py
```

2. Custom queries:
```python
from approx_query_engine import ApproxQueryEngine
from synthetic_data import generate_sales_data

# Create engine with test data
df = generate_sales_data(1_000_000)
engine = ApproxQueryEngine(df)

# Run queries
exact = engine.run_query("SELECT category, AVG(price) FROM sales GROUP BY category;")
approx = engine.run_query(
    "APPROXIMATE SELECT category, AVG(price) FROM sales GROUP BY category WITH ACCURACY 95%;"
)

# Print results
engine.print_result(exact)
engine.print_result(approx)
```

## Supported Query Syntax

- Basic aggregations: `COUNT(*), SUM(col), AVG(col)`
- Group by: `GROUP BY column1, column2, ...`
- Approximate queries: Add `APPROXIMATE` prefix and `WITH ACCURACY x%`

## Query Types Supported

1. **Exact Queries**
   ```sql
   SELECT COUNT(*) FROM sales;
   SELECT category, AVG(price) FROM sales GROUP BY category;
   SELECT COUNT(DISTINCT user_id) FROM sales;
   ```

2. **Approximate Queries**
   ```sql
   APPROXIMATE SELECT COUNT(*) FROM sales WITH ACCURACY 95%;
   APPROXIMATE SELECT category, SUM(price) FROM sales GROUP BY category WITH ACCURACY 90%;
   APPROXIMATE SELECT COUNT(DISTINCT user_id) FROM sales WITH ACCURACY 95% WITHIN 95% CONFIDENCE;
   ```

## Supported Operations

- **Aggregations**
  - `COUNT(*)`
  - `COUNT(column)`
  - `COUNT(DISTINCT column)`
  - `SUM(column)`
  - `AVG(column)`

- **Grouping**
  - `GROUP BY column1, column2, ...`

- **Approximation Controls**
  - `WITH ACCURACY x%` - Target accuracy level (90%, 95%, 99%)
  - `WITHIN y% CONFIDENCE` - Confidence interval (default: 95%)

## Implementation Details

1. **Query Processing**
   - SQL parsing with regex for simplicity
   - Support for basic SQL syntax and aggregations
   - Case-insensitive column names

2. **Approximation Techniques**
   - Adaptive sampling based on accuracy requirements
   - HyperLogLog for distinct counting
   - Reservoir sampling for continuous data streams

3. **Error Estimation**
   - Statistical confidence intervals
   - Error bounds based on sample size
   - Actual vs. estimated error reporting in benchmarks
