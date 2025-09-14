from typing import Dict, Any, List, Union, Optional
import pandas as pd
import numpy as np
import re
import time
from sketches import HyperLogLog, CountMinSketch, ReservoirSampler
from collections import defaultdict

class ApproxQueryEngine:
    def __init__(self, df: Optional[pd.DataFrame] = None):
        """Initialize the query engine with optional initial DataFrame."""
        self.df = df.copy() if df is not None else pd.DataFrame()
        self.streaming_mode = df is None
        self._cache = {
            'total_rows': len(self.df) if df is not None else 0,
            'category_stats': None,
            'sketches': {},
            'samples': {}
        }
        self._setup_cache()

    def _setup_cache(self):
        """Initialize cache with sketches and samples for key columns."""
        if not self.streaming_mode and len(self.df) > 0:
            # Initialize category-based statistics
            self._cache['category_stats'] = (
                self.df.groupby('category', as_index=False)
                .agg({'price': ['count', 'mean', 'sum']})
            )
            self._cache['category_stats'].columns = ['category', 'count', 'avg_price', 'sum_price']

            # Initialize sketches for each numeric column
            for col in self.df.select_dtypes(include=[np.number]).columns:
                self._cache['sketches'][f'count_min_{col}'] = CountMinSketch()
                self._cache['sketches'][f'hll_{col}'] = HyperLogLog()
                self._cache['samples'][col] = ReservoirSampler(1000)

                # Populate sketches and samples
                for value in self.df[col]:
                    self._cache['sketches'][f'count_min_{col}'].add(value)
                    self._cache['sketches'][f'hll_{col}'].add(value)
                    self._cache['samples'][col].add(value)

    def add_streaming_data(self, new_data: pd.DataFrame) -> None:
        """Add new data in streaming mode."""
        if not self.streaming_mode:
            raise ValueError("Engine not in streaming mode")

        # Update main DataFrame
        self.df = pd.concat([self.df, new_data], ignore_index=True)
        self._cache['total_rows'] += len(new_data)

        # Update sketches and samples
        for col in new_data.select_dtypes(include=[np.number]).columns:
            if col not in self._cache['sketches']:
                self._cache['sketches'][f'count_min_{col}'] = CountMinSketch()
                self._cache['sketches'][f'hll_{col}'] = HyperLogLog()
                self._cache['samples'][col] = ReservoirSampler(1000)

            for value in new_data[col]:
                self._cache['sketches'][f'count_min_{col}'].add(value)
                self._cache['sketches'][f'hll_{col}'].add(value)
                self._cache['samples'][col].add(value)

    def run_query(self, query: str) -> Dict[str, Any]:
        """Run a query with support for exact and approximate results."""
        try:
            q = self.parse_query(query)
            start = time.perf_counter()
            
            if q["is_approx"]:
                result = self._run_approximate(q)
            else:
                result = self._run_exact(q)
                
            elapsed = time.perf_counter() - start
            return {**result, "time": elapsed}
        except Exception as e:
            return {"error": str(e), "time": 0.001}

    def parse_query(self, query: str) -> Dict[str, Any]:
        """Parse SQL-like query with support for approximation and accuracy settings."""
        pattern = r"(APPROXIMATE\s+)?SELECT\s+(.*?)\s+FROM\s+(\w+)(?:\s+GROUP BY\s+(.*?))?(?:\s+WITH ACCURACY\s+(\d+)%?)?(?:\s+WITHIN\s+(\d+)%\s+CONFIDENCE)?;"
        m = re.match(pattern, query.strip(), re.IGNORECASE)
        if not m:
            raise ValueError("Invalid query format")
        return {
            "is_approx": bool(m.group(1)),
            "select_expr": m.group(2),
            "table": m.group(3),
            "group_by": m.group(4),
            "accuracy": int(m.group(5)) if m.group(5) else 95,
            "confidence": int(m.group(6)) if m.group(6) else 95
        }

    def _run_exact(self, q: Dict[str, Any]) -> Dict[str, Any]:
        """Run exact query computation."""
        if "COUNT(*)" in q["select_expr"] and not q["group_by"]:
            return {
                "result": pd.DataFrame([{"count": self._cache['total_rows']}]),
                "approx": False
            }

        group_cols = [c.strip() for c in q["group_by"].split(",")] if q["group_by"] else None
        agg_dict = self._parse_select_expr(q["select_expr"])
        
        if group_cols:
            result = self.df.groupby(group_cols, as_index=False).agg(agg_dict)
        else:
            result = pd.DataFrame([self.df.agg(agg_dict)])
            
        return {"result": result, "approx": False}

    def _run_approximate(self, q: Dict[str, Any]) -> Dict[str, Any]:
        """Run approximate query using sketches and sampling."""
        if "COUNT(*)" in q["select_expr"] and not q["group_by"]:
            # Use HyperLogLog for distinct counts
            if "DISTINCT" in q["select_expr"].upper():
                hll = self._cache['sketches'].get('hll_id', HyperLogLog())
                count = hll.count()
            else:
                count = self._cache['total_rows']
            
            return {
                "result": pd.DataFrame([{"count": count}]),
                "approx": True,
                "ci": self._calculate_confidence_interval(q["confidence"], count)
            }

        sample_size = self._get_sample_size(q["accuracy"])
        sample = self._get_stratified_sample(sample_size, q) if q["group_by"] else self.df.sample(n=sample_size)
        
        group_cols = [c.strip() for c in q["group_by"].split(",")] if q["group_by"] else None
        agg_dict = self._parse_select_expr(q["select_expr"])
        
        if group_cols:
            result = sample.groupby(group_cols, as_index=False).agg(agg_dict)
            scale = len(self.df) / len(sample)
            for col in result.columns:
                if col not in group_cols and ('count' in col.lower() or 'sum' in col.lower()):
                    result[col] *= scale
        else:
            result = pd.DataFrame([sample.agg(agg_dict)])
            
        ci = self._calculate_confidence_interval(q["confidence"], sample_size)
        return {
            "result": result, 
            "approx": True, 
            "ci": ci,
            "sample_size": sample_size,
            "total_size": len(self.df)
        }

    def _get_stratified_sample(self, sample_size: int, query: Dict[str, Any]) -> pd.DataFrame:
        """Get a stratified sample based on GROUP BY columns."""
        group_cols = [c.strip() for c in query["group_by"].split(",")]
        groups = self.df.groupby(group_cols)
        
        # Calculate per-group sample sizes proportionally
        total_rows = len(self.df)
        samples = []
        
        for name, group in groups:
            group_size = len(group)
            group_sample_size = max(1, int(sample_size * (group_size / total_rows)))
            samples.append(group.sample(n=min(group_sample_size, group_size)))
            
        return pd.concat(samples)

    def _calculate_confidence_interval(self, confidence: int, sample_size: int) -> float:
        """Calculate confidence interval based on sample size and desired confidence level."""
        z_score = {
            90: 1.645,
            95: 1.96,
            99: 2.576
        }.get(confidence, 1.96)
        
        return z_score * (1.0 / np.sqrt(sample_size))

    def _get_sample_size(self, accuracy: int) -> int:
        """Calculate required sample size based on desired accuracy."""
        base_size = len(self.df)
        min_size = 1000
        max_size = min(100000, base_size)
        
        # Using statistical formula for sample size
        z_score = 1.96  # 95% confidence level
        margin_of_error = (100 - accuracy) / 100.0
        sample_size = int((z_score**2 * 0.25) / (margin_of_error**2))
        
        return max(min_size, min(max_size, sample_size))

    def _parse_select_expr(self, expr: str) -> Dict[str, List[str]]:
        """Parse SELECT expressions into aggregation dictionary."""
        agg_dict = defaultdict(list)
        
        for item in expr.split(','):
            item = item.strip().upper()
            if 'COUNT(*)' in item:
                agg_dict['*'].append('count')
            elif 'COUNT(DISTINCT' in item:
                m = re.search(r'COUNT\(DISTINCT\s+(\w+)\)', item, re.IGNORECASE)
                if m:
                    agg_dict[m.group(1)].append('nunique')
            elif 'COUNT(' in item:
                m = re.search(r'COUNT\((\w+)\)', item, re.IGNORECASE)
                if m:
                    agg_dict[m.group(1)].append('count')
            elif 'AVG(' in item:
                m = re.search(r'AVG\((\w+)\)', item, re.IGNORECASE)
                if m:
                    agg_dict[m.group(1)].append('mean')
            elif 'SUM(' in item:
                m = re.search(r'SUM\((\w+)\)', item, re.IGNORECASE)
                if m:
                    agg_dict[m.group(1)].append('sum')
            elif item != '*':
                agg_dict[item].append('first')
        
        return dict(agg_dict)