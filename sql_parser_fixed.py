"""
SQL Parser for Approximate Query Engine.

This module provides a SQL parsing implementation that supports:
- SELECT, WHERE, GROUP BY clauses
- Aggregation functions (COUNT, SUM, AVG)
- DISTINCT keyword
- Approximate query syntax with error bounds
"""

import re
from typing import Dict, List, Optional, Union
from dataclasses import dataclass

@dataclass
class AggregateInfo:
    """Information about an aggregation function in a query."""
    function: str  # e.g., 'COUNT', 'AVG'
    column: str    # Column being aggregated
    alias: Optional[str] = None  # Optional AS clause
    is_distinct: bool = False

@dataclass
class QueryInfo:
    """Parsed information about a SQL query."""
    select_columns: List[Union[str, AggregateInfo]]
    table: str
    where_clause: Optional[str] = None
    group_by: Optional[List[str]] = None
    error_bound: Optional[float] = None
    confidence_level: Optional[float] = None

    def __post_init__(self):
        if self.group_by is None:
            self.group_by = []

class SQLParser:
    """
    A SQL parser that handles both exact and approximate queries.
    
    Supports:
    - Basic SELECT, WHERE, GROUP BY
    - Aggregations: COUNT, SUM, AVG
    - DISTINCT keyword
    - Error bound specifications
    """
    
    def __init__(self):
        """Initialize the SQL parser."""
        self.query_types = {
            'select': r'SELECT\s+(.+?)\s+FROM',
            'from': r'FROM\s+(\w+)',
            'where': r'WHERE\s+(.+?)(?:\s+GROUP\s+BY|\s+ORDER\s+BY|$)',
            'group_by': r'GROUP\s+BY\s+(.+?)(?:\s+ORDER\s+BY|$)',
            'error_bound': r'WITH\s+ACCURACY\s+(\d+(?:\.\d+)?)%',
            'confidence': r'WITHIN\s+(\d+(?:\.\d+)?)%\s+CONFIDENCE'
        }
    
    def parse_query(self, sql: str) -> QueryInfo:
        """Parse SQL query into structured format."""
        sql = sql.strip().rstrip(';')
        
        # Extract basic components
        table = self._extract_table(sql)
        select_columns = self._extract_select_columns(sql)
        where_clause = self._extract_where(sql)
        group_by = self._extract_group_by(sql)
        error_bound = self._extract_error_bound(sql)
        confidence_level = self._extract_confidence_level(sql)
        
        return QueryInfo(
            select_columns=select_columns,
            table=table,
            where_clause=where_clause,
            group_by=group_by,
            error_bound=error_bound,
            confidence_level=confidence_level
        )
    
    def _extract_match(self, pattern: str, sql: str) -> Optional[str]:
        """Extract the first match for a given pattern."""
        match = re.search(pattern, sql, re.IGNORECASE | re.DOTALL)
        return match.group(1).strip() if match else None
    
    def _extract_table(self, sql: str) -> str:
        """Extract table name from SQL."""
        table = self._extract_match(self.query_types['from'], sql)
        if not table:
            raise ValueError("Missing FROM clause in query")
        return table.lower()
    
    def _extract_select_columns(self, sql: str) -> List[Union[str, AggregateInfo]]:
        """Extract and parse SELECT columns."""
        select_part = self._extract_match(self.query_types['select'], sql)
        if not select_part:
            raise ValueError("Missing SELECT clause")
        
        columns = []
        for col in select_part.split(','):
            col = col.strip()
            
            # Check for aggregation functions
            agg_info = self._parse_aggregation(col)
            if agg_info:
                columns.append(AggregateInfo(
                    function=agg_info['function'],
                    column=agg_info['column'],
                    is_distinct=agg_info.get('distinct', False)
                ))
            else:
                columns.append(col.lower())
        
        return columns
    
    def _extract_where(self, sql: str) -> Optional[str]:
        """Extract WHERE clause."""
        return self._extract_match(self.query_types['where'], sql)
    
    def _extract_group_by(self, sql: str) -> List[str]:
        """Extract GROUP BY columns."""
        group_by = self._extract_match(self.query_types['group_by'], sql)
        if not group_by:
            return []
        return [col.strip().lower() for col in group_by.split(',')]
    
    def _extract_error_bound(self, sql: str) -> Optional[float]:
        """Extract error bound percentage."""
        bound = self._extract_match(self.query_types['error_bound'], sql)
        return float(bound) / 100 if bound else None
    
    def _extract_confidence_level(self, sql: str) -> Optional[float]:
        """Extract confidence level."""
        confidence = self._extract_match(self.query_types['confidence'], sql)
        return float(confidence) / 100 if confidence else None
    
    def _parse_aggregation(self, col: str) -> Optional[Dict]:
        """Parse aggregation functions from a column expression."""
        # Check for COUNT DISTINCT first
        distinct_pattern = r'COUNT\s*\(\s*DISTINCT\s+([^)]+)\)'
        match = re.match(distinct_pattern, col, re.IGNORECASE)
        if match:
            return {
                'function': 'COUNT',
                'distinct': True,
                'column': match.group(1).strip().lower()
            }
        
        # Then check for other aggregations
        agg_pattern = r'(COUNT|SUM|AVG|MIN|MAX)\s*\(([^)]+)\)'
        match = re.match(agg_pattern, col, re.IGNORECASE)
        if match:
            function = match.group(1).upper()
            column = match.group(2).strip()
            
            # Handle COUNT(*)
            if column == '*' and function == 'COUNT':
                return {
                    'function': function,
                    'distinct': False,
                    'column': '*'
                }
                
            # For other aggregations, use lowercase column names
            return {
                'function': function,
                'distinct': False,
                'column': column.lower()
            }
            
        return None