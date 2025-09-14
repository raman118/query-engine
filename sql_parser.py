from typing import List, Dict, Tuple, Optional
import re

class SQLParser:
    """SQL Parser for Approximate Query Engine"""
    
    def __init__(self):
        self.query_types = {
            'select': r'SELECT\s+(.+?)\s+FROM',
            'from': r'FROM\s+(\w+)',
            'group_by': r'GROUP\s+BY\s+(.+?)(?:;|$)',
            'where': r'WHERE\s+(.+?)(?:GROUP BY|;|$)'
        }
        
    def parse_query(self, sql: str) -> Dict:
        """Parse SQL query into components."""
        # Store original query for preserving case in AS clauses
        original_sql = sql
        sql = sql.strip().upper()
        
        # Extract basic components
        result = {
            'select_cols': self._extract_select_columns(sql, original_sql),
            'table': self._extract_match(self.query_types['from'], sql),
            'group_by': self._extract_group_by(sql),
            'where': self._extract_match(self.query_types['where'], sql),
            'aggregations': []
        }
        
        # Parse aggregations
        for col in result['select_cols']:
            agg_info = self._parse_aggregation(col)
            if agg_info:
                result['aggregations'].append(agg_info)
        
        return result
    
    def _extract_match(self, pattern: str, sql: str) -> Optional[str]:
        """Extract first match from SQL using regex pattern."""
        match = re.search(pattern, sql, re.IGNORECASE)
        return match.group(1).strip() if match else None
    
    def _extract_select_columns(self, sql: str, original_sql: str) -> List[str]:
        """Extract and clean columns from SELECT clause, preserving AS clause case."""
        select_part = self._extract_match(self.query_types['select'], sql)
        if not select_part:
            return []
            
        # Find all AS clauses in original query to preserve their case
        as_clauses = {}
        for match in re.finditer(r'(\w+)\s+as\s+(\w+)', original_sql, re.IGNORECASE):
            as_clauses[match.group(1).upper()] = match.group(2)
        
        # Process each column
        columns = []
        for col in select_part.split(','):
            col = col.strip()
            # Check if this is an AS clause
            as_match = re.match(r'(.+?)\s+AS\s+(\w+)', col)
            if as_match:
                expr = as_match.group(1).strip()
                alias = as_clauses.get(expr, as_match.group(2).lower())
                columns.append(f"{expr} AS {alias}")
            else:
                # Convert simple column names to lowercase
                if not any(agg in col.upper() for agg in ['COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'DISTINCT']):
                    col = col.lower()
                columns.append(col)
        return columns
    
    def _extract_group_by(self, sql: str) -> List[str]:
        """Extract and clean columns from GROUP BY clause."""
        group_by = self._extract_match(self.query_types['group_by'], sql)
        if not group_by:
            return []
        # Convert GROUP BY columns to lowercase since they're direct column references
        return [col.strip().lower() for col in group_by.split(',')]
    
    def _parse_aggregation(self, col: str) -> Optional[Dict]:
        """Parse aggregation functions from a column expression."""
        # Match standard aggregation functions
        agg_pattern = r'(COUNT|SUM|AVG|MIN|MAX)\s*\(([^)]+)\)'
        distinct_pattern = r'COUNT\s*\(\s*DISTINCT\s+([^)]+)\)'
        
        # Check for COUNT DISTINCT first
        match = re.match(distinct_pattern, col, re.IGNORECASE)
        if match:
            return {
                'function': 'COUNT',
                'distinct': True,
                'column': match.group(1).strip().lower()  # Column names should be lowercase
            }
        
        # Then check for other aggregations
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
    
    def validate_query(self, query: Dict) -> bool:
        """Validate parsed query structure."""
        if not query.get('table'):
            raise ValueError("Missing FROM clause")
        
        if not query.get('select_cols'):
            raise ValueError("Missing SELECT columns")
            
        return True