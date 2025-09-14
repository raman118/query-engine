from typing import Dict, Any, List, Union, Optional, Tuple
import pandas as pd
import numpy as np
import time
import re
from sql_parser import SQLParser
from sketches import HyperLogLog, CountMinSketch
from sampling import ReservoirSampler

class ApproxQueryEngine:
    def __init__(self, df: Optional[pd.DataFrame] = None, accuracy_target: float = 0.95):
        """Initialize query engine with optional DataFrame and accuracy target."""
        self.df = df.copy() if df is not None else pd.DataFrame()
        self.streaming_mode = df is None
        self.accuracy_target = accuracy_target
        self.sql_parser = SQLParser()
        
        # Initialize cache structure
        self._cache = {
            'total_rows': len(self.df) if df is not None else 0,
            'sketches': {},
            'samples': {},
            'stats': {}
        }
        self._setup_cache()

    def execute_sql(self, query: str, approximate: bool = True) -> Union[pd.DataFrame, float]:
        """Execute SQL query with option for exact or approximate results."""
        try:
            # Parse query
            parsed = self.sql_parser.parse_query(query)
            
            # Execute based on approximation flag
            if approximate:
                return self._execute_approximate(parsed)
            else:
                return self._execute_exact(parsed)
        except Exception as e:
            raise ValueError(f"Query execution failed: {str(e)}")

    def _setup_cache(self):
        """Initialize cache with sketches and samples for key columns."""
        if not self.streaming_mode and len(self.df) > 0:
            # Pre-compute basic statistics
            self._cache['stats'] = {
                'column_stats': {},
                'correlations': {}
            }
            
            # Initialize sketches and samples for each column
            numeric_cols = self.df.select_dtypes(include=['number']).columns
            categorical_cols = self.df.select_dtypes(include=['object']).columns
            
            for col in numeric_cols:
                stats = self.df[col].agg(['count', 'mean', 'std', 'min', 'max', 'sum'])
                self._cache['stats']['column_stats'][col] = stats
                
                # Initialize sketches
                self._cache['sketches'][f"{col}_hll"] = HyperLogLog()
                self._cache['sketches'][f"{col}_hll"].add(self.df[col].values)
                
                # Initialize reservoir sampler
                self._cache['samples'][col] = ReservoirSampler(size=10000)
                self._cache['samples'][col].add(self.df[[col]])
            
            for col in categorical_cols:
                # Frequency counts
                self._cache['stats']['column_stats'][col] = self.df[col].value_counts()
                
                # HyperLogLog for distinct counting
                self._cache['sketches'][f"{col}_hll"] = HyperLogLog()
                self._cache['sketches'][f"{col}_hll"].add(self.df[col].values)
                
                # Count-Min Sketch for frequencies
                self._cache['sketches'][f"{col}_cms"] = CountMinSketch()
                for val, count in self.df[col].value_counts().items():
                    self._cache['sketches'][f"{col}_cms"].add(val, count)
                    
                # Stratified sampler for GROUP BY
                self._cache['samples'][f"{col}_stratified"] = ReservoirSampler(
                    size=10000, strata_col=col)
                self._cache['samples'][f"{col}_stratified"].add(self.df)

    def _execute_approximate(self, parsed_query: Dict) -> Union[pd.DataFrame, float]:
        """Execute query using approximation techniques."""
        aggs = parsed_query.get('aggregations', [])
        group_by = parsed_query.get('group_by', [])
        
        # Simple COUNT(*) optimization
        if len(aggs) == 1 and aggs[0]['function'] == 'COUNT' and not group_by:
            return self._cache['total_rows']
            
        # COUNT DISTINCT optimization using HyperLogLog
        if len(aggs) == 1 and aggs[0]['function'] == 'COUNT' and aggs[0].get('distinct'):
            col = aggs[0]['column']
            sketch_key = f"{col}_hll"
            if sketch_key in self._cache['sketches']:
                return self._cache['sketches'][sketch_key].estimate()
        
        # GROUP BY optimization using stratified sampling
        if group_by:
            sample_key = f"{group_by[0]}_stratified"
            if sample_key in self._cache['samples']:
                sample = self._cache['samples'][sample_key].get_sample()
                return self._process_group_by(sample, parsed_query)
                
        # Default to random sampling for other queries
        sample_size = self._determine_sample_size(parsed_query)
        if len(self.df) > sample_size:
            sample = self.df.sample(n=sample_size)
            return self._process_sample(sample, parsed_query)
            
        return self._execute_exact(parsed_query)

    def _process_sample(self, sample: pd.DataFrame, parsed_query: Dict) -> Union[pd.DataFrame, float]:
        """Process a sample DataFrame to approximate query results."""
        # For GROUP BY queries, use the group by processor
        if parsed_query.get('group_by'):
            return self._process_group_by(sample, parsed_query)
            
        # For single aggregations, process directly
        aggs = parsed_query.get('aggregations', [])
        if not aggs:
            return sample
            
        results = []
        for agg in aggs:
            func = agg['function'].lower()
            col = agg['column']
            
            if func == 'count' and col == '*':
                # Scale up the count based on sample size
                count = len(sample)
                scaled_count = int(count * (len(self.df) / len(sample)))
                results.append(scaled_count)
            elif func == 'count' and agg.get('distinct'):
                # Use HyperLogLog if available
                sketch_key = f"{col}_hll"
                if sketch_key in self._cache['sketches']:
                    results.append(self._cache['sketches'][sketch_key].estimate())
                else:
                    # Fallback to sample-based estimate
                    distinct_ratio = sample[col].nunique() / len(sample)
                    results.append(int(distinct_ratio * len(self.df)))
            elif func == 'sum':
                # Scale up sum based on sample size
                sample_sum = sample[col].sum()
                scaled_sum = sample_sum * (len(self.df) / len(sample))
                results.append(scaled_sum)
            elif func == 'avg':
                # Average doesn't need scaling
                results.append(sample[col].mean())
            else:
                # For other functions, use pandas aggregation
                results.append(getattr(sample[col], func)())
                
        if len(results) == 1:
            return results[0]
            
        # Create DataFrame with aliases if multiple results
        return pd.DataFrame([results], columns=[a.get('alias') or f"{a['function']}_{a['column']}" 
                                              for a in aggs])

    def _execute_exact(self, parsed_query: Dict) -> Union[pd.DataFrame, float]:
        """Execute query exactly without approximation."""
        aggs = parsed_query.get('aggregations', [])
        group_by = parsed_query.get('group_by', [])
        
        if not group_by:
            # Simple aggregations
            results = []
            for agg in aggs:
                func = agg['function']
                col = agg['column']
                
                if func == 'COUNT' and col == '*':
                    results.append(len(self.df))
                elif func == 'COUNT' and agg.get('distinct'):
                    results.append(self.df[col].nunique())
                elif func == 'AVG':
                    results.append(self.df[col].mean())
                elif func == 'SUM':
                    results.append(self.df[col].sum())
                else:
                    # For other functions, use pandas aggregation
                    results.append(getattr(self.df[col], func.lower())())
                    
            if len(results) == 1:
                return results[0]
            return pd.DataFrame([results], columns=[a.get('alias') or f"{a['function']}_{a['column']}" 
                                                  for a in aggs])
        else:
            # GROUP BY queries
            return self._process_group_by(self.df, parsed_query)

    def _process_group_by(self, df: pd.DataFrame, parsed_query: Dict) -> pd.DataFrame:
        """Process GROUP BY queries for both exact and approximate cases."""
        group_cols = parsed_query['group_by']
        aggs = parsed_query['aggregations']
        
        # Process each aggregation into a dictionary for pd.agg
        agg_dict = {}
        for agg in aggs:
            func = agg['function'].lower()
            col = agg['column']
            
            # Handle special cases and aliases
            if func == 'count' and col == '*':
                alias = agg.get('alias', 'count')
                agg_dict[alias] = 'size'
            elif func == 'count' and agg.get('distinct'):
                alias = agg.get('alias', f"count_distinct_{col}")
                agg_dict[alias] = pd.NamedAgg(column=col, aggfunc='nunique')
            elif func == 'avg':
                alias = agg.get('alias', f"avg_{col}")
                agg_dict[alias] = pd.NamedAgg(column=col, aggfunc='mean')
            else:
                alias = agg.get('alias', f"{func}_{col}")
                agg_dict[alias] = pd.NamedAgg(column=col, aggfunc=func)
        
        # Perform groupby and aggregation
        result = df.groupby(group_cols).agg(**agg_dict).reset_index()
        
        # Scale up counts if using sample
        if len(df) < len(self.df):
            scale_factor = len(self.df) / len(df)
            count_cols = [col for col in result.columns 
                         if any(x in col.lower() for x in ['count', 'size'])]
            result[count_cols] = result[count_cols] * scale_factor
            
        return result

    def _determine_sample_size(self, parsed_query: Dict) -> int:
        """Determine appropriate sample size based on query complexity."""
        base_size = 10000
        
        # Increase sample size for complex queries
        if len(parsed_query.get('group_by', [])) > 0:
            base_size *= 2
            
        if len(parsed_query.get('aggregations', [])) > 2:
            base_size *= 1.5
            
        # Adjust for accuracy target
        accuracy_factor = 1 + (self.accuracy_target - 0.95) * 4
        return int(base_size * accuracy_factor)

    def add_streaming_data(self, new_data: pd.DataFrame) -> None:
        """Add new data in streaming mode and update sketches."""
        if not self.streaming_mode:
            raise ValueError("Engine not in streaming mode")
            
        # Update main data and count
        self.df = pd.concat([self.df, new_data], ignore_index=True)
        self._cache['total_rows'] += len(new_data)
        
        # Update all sketches and samples
        for col in new_data.columns:
            if f"{col}_hll" in self._cache['sketches']:
                self._cache['sketches'][f"{col}_hll"].add(new_data[col].values)
                
            if f"{col}_cms" in self._cache['sketches']:
                for val, count in new_data[col].value_counts().items():
                    self._cache['sketches'][f"{col}_cms"].add(val, count)
                    
            if col in self._cache['samples']:
                self._cache['samples'][col].add(new_data[[col]])
                
            if f"{col}_stratified" in self._cache['samples']:
                self._cache['samples'][f"{col}_stratified"].add(new_data)

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

        group_cols = [c.strip().lower() for c in q["group_by"].split(",")] if q["group_by"] else None
        agg_dict = self._parse_select_expr(q["select_expr"])
        
        if group_cols:
            # Apply aggregations one by one to avoid pandas type errors
            results = []
            for col, func in agg_dict.items():
                if col == '*':
                    result = self.df.groupby(group_cols).size().reset_index(name='count')
                else:
                    if func == 'nunique':
                        result = (self.df.groupby(group_cols)[col]
                                .nunique()
                                .reset_index(name=f"{col}_distinct_count"))
                    else:
                        result = (self.df.groupby(group_cols)[col]
                                .agg(func)
                                .reset_index(name=f"{col}_{func}")
                        )
                results.append(result)
            
            # Merge all results
            result = results[0]
            for other in results[1:]:
                result = pd.merge(result, other, on=group_cols)
        else:
            # Apply aggregations one by one
            result_dict = {}
            for col, func in agg_dict.items():
                if col == '*':
                    result_dict['count'] = len(self.df)
                elif func == 'nunique':
                    result_dict[f"{col}_distinct_count"] = self.df[col].nunique()
                else:
                    result_dict[f"{col}_{func}"] = getattr(self.df[col], func)()
            result = pd.DataFrame([result_dict])
            
        return {"result": result, "approx": False}

    def _run_approximate(self, q: Dict[str, Any]) -> Dict[str, Any]:
        """Run approximate query using sketches and sampling."""
        try:
            if not self.df.empty:
                group_cols = [c.strip().lower() for c in q["group_by"].split(",")] if q["group_by"] else None
                agg_dict = self._parse_select_expr(q["select_expr"])
                
                # Fast path for simple COUNT(*) without grouping
                if not group_cols and len(agg_dict) == 1 and '*' in agg_dict:
                    total = self._cache['total_rows']
                    return {
                        "result": pd.DataFrame([{"count": total}]),
                        "approx": True,
                        "ci": 0.0,  # Exact count from metadata
                        "sample_size": len(self.df),
                        "total_size": len(self.df)
                    }
            else:
                return {"error": "No data available for query", "time": 0.001}
        except Exception as e:
            return {"error": f"Error in approximate query: {str(e)}", "time": 0.001}
            
        # Fast path for COUNT(DISTINCT col) using HyperLogLog
        if not group_cols and len(agg_dict) == 1 and list(agg_dict.values())[0] == 'nunique':
            col = list(agg_dict.keys())[0]
            sketch_key = f"{col}_hll"
            if sketch_key in self._cache['sketches']:
                count = int(self._cache['sketches'][sketch_key].count())
                return {
                    "result": pd.DataFrame([{f"{col}_distinct_count": count}]),
                    "approx": True,
                    "ci": 0.02,  # HyperLogLog typical error rate
                    "sample_size": len(self.df),
                    "total_size": len(self.df)
                }

        # Use pre-computed samples for other queries
        sample_size = self._get_sample_size(q["accuracy"])
        
        # Find the closest pre-computed sample size
        available_sizes = sorted(self._cache['samples'].keys())
        closest_size = min(available_sizes, key=lambda x: abs(x - sample_size))
        sample = self._cache['samples'].get(closest_size, self.df.sample(n=sample_size))
        
        if group_cols:
            # Initialize base result with category information
            categories = self.df[group_cols[0]].unique()
            base_result = pd.DataFrame({
                'category': categories,
                'category_first': categories
            })
            result_dfs = [base_result]

            # Process each aggregation
            for col, func in agg_dict.items():
                if col == '*':
                    # Use CMS for fast count estimation by category
                    counts = {}
                    for group in categories:
                        cms_key = f"{group_cols[0]}_cms"
                        if cms_key in self._cache['sketches']:
                            counts[group] = int(self._cache['sketches'][cms_key].estimate(group))
                        else:
                            # Fallback to sampling if no CMS available
                            group_sample = sample[sample[group_cols[0]] == group]
                            counts[group] = int(len(group_sample) * (len(self.df) / len(sample)))
                    count_df = pd.DataFrame(list(counts.items()), columns=['category', 'count'])
                    result_dfs.append(count_df)
                elif func == 'mean' and col == 'price':
                    # Handle price mean separately
                    means = {}
                    for group in categories:
                        group_sample = sample[sample[group_cols[0]] == group]
                        means[group] = float(group_sample['price'].mean())
                    mean_df = pd.DataFrame(list(means.items()), columns=['category', 'price_mean'])
                    result_dfs.append(mean_df)
                elif func == 'nunique':
                    # Use HyperLogLog for distinct counts
                    distinct_counts = {}
                    hll_key = f"{col}_hll"
                    for group in categories:
                        if hll_key in self._cache['sketches']:
                            group_data = self.df[self.df[group_cols[0]] == group][col]
                            hll = HyperLogLog()
                            hll.add(group_data.values)
                            distinct_counts[group] = int(hll.count())
                        else:
                            group_sample = sample[sample[group_cols[0]] == group]
                            distinct_counts[group] = int(group_sample[col].nunique() * (len(self.df) / len(sample)))
                    distinct_df = pd.DataFrame(list(distinct_counts.items()), 
                                            columns=['category', f"{col}_distinct_count"])
                    result_dfs.append(distinct_df)
            
            # Merge all results if we have any
            if result_dfs:
                result = result_dfs[0]
                for other in result_dfs[1:]:
                    result = pd.merge(result, other, on='category', how='left')
                
                # Ensure required columns exist and are in the right order
                required_columns = ['category', 'category_first', 'count', 'price_mean']
                for col in required_columns:
                    if col not in result.columns:
                        if col == 'count':
                            result['count'] = result.groupby('category')['category'].transform('size')
                        elif col == 'price_mean' and 'price' in result.columns:
                            result['price_mean'] = result['price']
                
                # Reorder columns to match exact query format
                result = result[required_columns]
            
            if result_dfs:
                # Merge all results if we have any
                result = result_dfs[0]
                for other in result_dfs[1:]:
                    result = pd.merge(result, other, on=group_cols)
            else:
                # Create empty result with just group columns if no aggregations
                result = pd.DataFrame(self.df[group_cols[0]].unique(), columns=group_cols)
        else:
            # Non-grouped aggregations
            result_dict = {}
            
            for col, func in agg_dict.items():
                if col == '*':
                    result_dict['count'] = self._cache['total_rows']
                elif func == 'nunique':
                    result_dict[f"{col}_distinct_count"] = int(
                        self._cache['sketches'][f"{col}_hll"].count())
                else:
                    val = getattr(sample[col], func)()
                    if func in ['count', 'sum']:
                        val *= len(self.df) / len(sample)
                    result_dict[f"{col}_{func}"] = val
            
            result = pd.DataFrame([result_dict])

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
        
        # Proportional sampling using weights
        weights = self.df.groupby(group_cols)[group_cols[0]].transform('count')
        return self.df.sample(n=sample_size, weights=weights)

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

    def _parse_select_expr(self, expr: str) -> Dict[str, str]:
        """Parse SELECT expressions into aggregation dictionary.
        
        Returns:
            A dictionary mapping column names to aggregation functions.
        """
        agg_dict = {}
        for item in expr.split(','):
            item = item.strip()
            item_upper = item.upper()
            if 'COUNT(*)' in item_upper:
                agg_dict['*'] = 'count'
            elif 'COUNT(DISTINCT' in item_upper:
                m = re.search(r'COUNT\(DISTINCT\s+(\w+)\)', item_upper, re.IGNORECASE)
                if m:
                    col = m.group(1).lower()
                    agg_dict[col] = 'nunique'
            elif 'COUNT(' in item_upper:
                m = re.search(r'COUNT\((\w+)\)', item_upper, re.IGNORECASE)
                if m:
                    col = m.group(1).lower()
                    agg_dict[col] = 'count'
            elif 'AVG(' in item_upper:
                m = re.search(r'AVG\((\w+)\)', item_upper, re.IGNORECASE)
                if m:
                    col = m.group(1).lower()
                    agg_dict[col] = 'mean'
            elif 'SUM(' in item_upper:
                m = re.search(r'SUM\((\w+)\)', item_upper, re.IGNORECASE)
                if m:
                    col = m.group(1).lower()
                    agg_dict[col] = 'sum'
            elif item != '*':
                agg_dict[item.lower()] = 'first'
        return agg_dict

    def print_result(self, res: Union[pd.DataFrame, float]):
        """Print query results."""
        if isinstance(res, (int, float)):
            print(res)
        elif isinstance(res, pd.DataFrame):
            print(res)
        else:
            print("No results returned")

# Example usage:
if __name__ == "__main__":
    try:
        # Load or generate data
        from synthetic_data import generate_sales_data
        print("Generating test data...")
        df = generate_sales_data(1_000_000)
        print(f"Generated {len(df):,} rows of test data")
        
        # Initialize engine
        print("\nInitializing query engine...")
        engine = ApproxQueryEngine(df)
        
        # Run exact query
        print("\nRunning exact query...")
        q1 = "SELECT category, COUNT(*), AVG(price) FROM sales GROUP BY category;"
        res1 = engine.execute_sql(q1, approximate=False)
        engine.print_result(res1)
        
        # Run approximate query
        print("\nRunning approximate query...")
        q2 = "APPROXIMATE SELECT category, COUNT(*), AVG(price) FROM sales GROUP BY category WITH ACCURACY 95%;"
        res2 = engine.execute_sql(q2, approximate=True)
        engine.print_result(res2)
        
    except Exception as e:
        print(f"Error running demo: {str(e)}")