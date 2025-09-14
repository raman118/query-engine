from typing import List, Optional, Union, Any
import numpy as np
import pandas as pd
from collections import defaultdict

class ReservoirSampler:
    """Reservoir sampling with support for stratified sampling."""
    
    def __init__(self, size: int = 1000, strata_col: Optional[str] = None):
        self.size = size
        self.strata_col = strata_col
        self.sample: Union[pd.DataFrame, defaultdict] = pd.DataFrame()
        self.count = 0
        
        if strata_col:
            self.sample = defaultdict(lambda: pd.DataFrame())
            self.strata_counts = defaultdict(int)
            
    def add(self, data: pd.DataFrame) -> None:
        """Add new data to the reservoir sample."""
        if self.strata_col:
            self._add_stratified(data)
        else:
            self._add_simple(data)
            
    def _add_simple(self, data: pd.DataFrame) -> None:
        """Add data using simple reservoir sampling."""
        for _, row in data.iterrows():
            self.count += 1
            if len(self.sample) < self.size:
                self.sample = pd.concat([self.sample, pd.DataFrame([row])], ignore_index=True)
            else:
                j = np.random.randint(0, self.count)
                if j < self.size:
                    self.sample.iloc[j] = row
                    
    def _add_stratified(self, data: pd.DataFrame) -> None:
        """Add data using stratified reservoir sampling."""
        for stratum, group in data.groupby(self.strata_col):
            stratum_size = max(1, self.size // len(data[self.strata_col].unique()))
            
            for _, row in group.iterrows():
                self.strata_counts[stratum] += 1
                
                if len(self.sample[stratum]) < stratum_size:
                    self.sample[stratum] = pd.concat(
                        [self.sample[stratum], pd.DataFrame([row])],
                        ignore_index=True
                    )
                else:
                    j = np.random.randint(0, self.strata_counts[stratum])
                    if j < stratum_size:
                        self.sample[stratum].iloc[j] = row
                        
    def get_sample(self) -> pd.DataFrame:
        """Get the current sample."""
        if self.strata_col:
            samples = []
            for stratum_sample in self.sample.values():
                samples.append(stratum_sample)
            return pd.concat(samples, ignore_index=True)
        return self.sample