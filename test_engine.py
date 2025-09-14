import pytest
import pandas as pd
import numpy as np
from approx_query_engine import ApproxQueryEngine
from synthetic_data import generate_sales_data
from sql_parser import SQLParser

@pytest.fixture
def sample_df():
    """Create sample DataFrame for testing."""
    return generate_sales_data(10000)

@pytest.fixture
def engine(sample_df):
    """Create query engine instance with sample data."""
    return ApproxQueryEngine(sample_df)

@pytest.fixture
def sql_parser():
    """Create SQL parser instance."""
    return SQLParser()

class TestSQLParser:
    def test_simple_count(self, sql_parser):
        query = "SELECT COUNT(*) FROM sales"
        result = sql_parser.parse_query(query)
        assert result['select_cols'] == ['COUNT(*)']
        assert result['table'] == 'SALES'
        assert not result['group_by']
        
    def test_group_by(self, sql_parser):
        query = "SELECT category, COUNT(*) FROM sales GROUP BY category"
        result = sql_parser.parse_query(query)
        assert 'category' in result['select_cols']
        assert 'COUNT(*)' in result['select_cols']
        assert result['group_by'] == ['CATEGORY']

class TestApproxQueryEngine:
    def test_simple_count(self, engine):
        query = "SELECT COUNT(*) FROM sales"
        exact = engine.execute_sql(query, approximate=False)
        approx = engine.execute_sql(query, approximate=True)
        error = abs(exact - approx) / exact * 100
        assert error < 5  # Max 5% error
        
    def test_distinct_count(self, engine):
        query = "SELECT COUNT(DISTINCT user_id) FROM sales"
        exact = engine.execute_sql(query, approximate=False)
        approx = engine.execute_sql(query, approximate=True)
        error = abs(exact - approx) / exact * 100
        assert error < 5
        
    def test_sum_aggregation(self, engine):
        query = "SELECT SUM(price) FROM sales"
        exact = engine.execute_sql(query, approximate=False)
        approx = engine.execute_sql(query, approximate=True)
        error = abs(exact - approx) / exact * 100
        assert error < 5
        
    def test_avg_calculation(self, engine):
        query = "SELECT AVG(price) FROM sales"
        exact = engine.execute_sql(query, approximate=False)
        approx = engine.execute_sql(query, approximate=True)
        error = abs(exact - approx) / exact * 100
        assert error < 5
        
    def test_group_by(self, engine):
        query = "SELECT category, COUNT(*) FROM sales GROUP BY category"
        exact = engine.execute_sql(query, approximate=False)
        approx = engine.execute_sql(query, approximate=True)
        
        # Check each group's error
        for category in exact.index:
            exact_count = exact.loc[category, 'count']
            approx_count = approx.loc[category, 'count']
            error = abs(exact_count - approx_count) / exact_count * 100
            assert error < 5
            
    def test_complex_query(self, engine):
        query = """
        SELECT 
            category,
            COUNT(*) as count,
            AVG(price) as avg_price,
            SUM(price) as total_sales
        FROM sales 
        GROUP BY category
        """
        exact = engine.execute_sql(query, approximate=False)
        approx = engine.execute_sql(query, approximate=True)
        
        # Check errors for each metric
        for col in ['count', 'avg_price', 'total_sales']:
            errors = abs(exact[col] - approx[col]) / exact[col] * 100
            assert errors.mean() < 5

class TestStreamingSupport:
    def test_streaming_updates(self):
        # Initialize empty engine
        engine = ApproxQueryEngine()
        
        # Add data in chunks
        chunk_size = 1000
        total_rows = 10000
        
        for i in range(0, total_rows, chunk_size):
            chunk = generate_sales_data(chunk_size)
            engine.add_streaming_data(chunk)
            
            # Verify count after each chunk
            count = engine.execute_sql("SELECT COUNT(*) FROM sales", approximate=False)
            assert count == i + chunk_size
            
    def test_sketch_maintenance(self):
        # Test that sketches are properly maintained in streaming mode
        engine = ApproxQueryEngine()
        
        # Add initial chunk
        chunk1 = generate_sales_data(5000)
        engine.add_streaming_data(chunk1)
        
        # Get initial distinct count
        distinct1 = engine.execute_sql(
            "SELECT COUNT(DISTINCT user_id) FROM sales",
            approximate=True
        )
        
        # Add more data
        chunk2 = generate_sales_data(5000)
        engine.add_streaming_data(chunk2)
        
        # Get updated distinct count
        distinct2 = engine.execute_sql(
            "SELECT COUNT(DISTINCT user_id) FROM sales",
            approximate=True
        )
        
        # Verify count increased
        assert distinct2 > distinct1

if __name__ == '__main__':
    pytest.main([__file__])