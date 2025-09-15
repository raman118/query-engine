"""Sampling strategies for approximate query processing with error bounds.

This module provides a comprehensive set of sampling techniques for data analysis
and approximate query processing. Each sampling method comes with statistical
error bounds and confidence intervals, supporting both parametric and non-parametric
approaches.
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from typing import List, Optional, Union, Any, Dict, Tuple, Callable, Protocol, runtime_checkable, Literal, TypeVar, overload, cast, Final
from numpy.typing import NDArray, ArrayLike
from dataclasses import dataclass, field
from collections import defaultdict
from abc import ABC, abstractmethod
from numpy.random import RandomState, Generator
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from enum import Enum, auto

# Type aliases
Number = Union[int, float]
SampleData = Union[DataFrame, Series]
EstimatorFunction = Callable[[NDArray[Any]], float]

# Constants
DEFAULT_CONFIDENCE: Final[float] = 0.95
DEFAULT_BOOTSTRAP_ITERATIONS: Final[int] = 1000
DEFAULT_BATCH_SIZE: Final[int] = 100_000

__all__ = [
    "ErrorBounds",
    "SamplingConfig",
    "DistributionType",
    "AbstractSampler",
    "SimpleRandomSampler",
    "StratifiedSampler",
    "ReservoirSampler",
    "DEFAULT_CONFIDENCE",
    "DEFAULT_BOOTSTRAP_ITERATIONS",
    "DEFAULT_BATCH_SIZE"
]

class DistributionType(Enum):
    """Supported distribution types for error bounds."""
    NORMAL = auto()
    STUDENT_T = auto()
    BOOTSTRAP = auto()
    NON_PARAMETRIC = auto()
    
    @classmethod
    def from_str(cls, value: str) -> "DistributionType":
        """Convert string to DistributionType."""
        try:
            return {
                "normal": cls.NORMAL,
                "student_t": cls.STUDENT_T,
                "bootstrap": cls.BOOTSTRAP,
                "non_parametric": cls.NON_PARAMETRIC
            }[value.lower()]
        except KeyError:
            raise ValueError(f"Unknown distribution type: {value}")

@dataclass(frozen=True)
class ErrorBounds:
    """Statistical bounds for sample-based estimates."""
    estimate: float
    lower_bound: float
    upper_bound: float
    confidence_level: float
    sample_size: int
    population_size: Optional[int] = None
    method: str = field(default="normal")
    distribution_type: Optional[DistributionType] = None
    n_iterations: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate the error bounds parameters."""
        if not 0 < self.confidence_level < 1:
            raise ValueError("Confidence level must be between 0 and 1")
        if self.sample_size < 0:
            raise ValueError("Sample size cannot be negative")
        if self.population_size is not None and self.population_size < self.sample_size:
            raise ValueError("Population size must be >= sample size")
        if not self.lower_bound <= self.estimate <= self.upper_bound:
            raise ValueError("Estimate must be within bounds")

    @property
    def margin_of_error(self) -> float:
        """Calculate the margin of error."""
        return (self.upper_bound - self.lower_bound) / 2

    @property
    def relative_error(self) -> float:
        """Calculate the relative error bound."""
        return self.margin_of_error / abs(self.estimate) if self.estimate != 0 else float("inf")

@dataclass(frozen=True)
class SamplingConfig:
    """Configuration parameters for sampling operations."""
    confidence_level: float = DEFAULT_CONFIDENCE
    error_method: str = "normal"
    n_bootstrap: int = DEFAULT_BOOTSTRAP_ITERATIONS
    batch_size: int = DEFAULT_BATCH_SIZE
    random_state: Optional[Union[int, RandomState, Generator]] = None

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if not 0 < self.confidence_level < 1:
            raise ValueError("Confidence level must be between 0 and 1")
        if self.error_method not in {"normal", "student_t", "bootstrap", "non_parametric"}:
            raise ValueError(f"Unknown error method: {self.error_method}")
        if self.n_bootstrap < 1:
            raise ValueError("Number of bootstrap iterations must be positive")
        if self.batch_size < 1:
            raise ValueError("Batch size must be positive")

class AbstractSampler(ABC):
    """Base class for sampling implementations."""
    
    def __init__(self, config: Optional[SamplingConfig] = None):
        """Initialize sampler with configuration parameters."""
        if config is None:
            config = SamplingConfig()
        self.config = config
        self.confidence_level = config.confidence_level
        self.error_method = config.error_method
        self.n_bootstrap = config.n_bootstrap
        self._rng = self._initialize_rng(config.random_state)
        
    def _initialize_rng(self,
                     random_state: Optional[Union[int, RandomState, Generator]] = None
                     ) -> Generator:
        """Initialize the random number generator."""
        if isinstance(random_state, Generator):
            return random_state
        elif isinstance(random_state, RandomState):
            seed = random_state.randint(0, 2**32 - 1)
            return np.random.default_rng(seed)
        return np.random.default_rng(random_state)
        
    def _bootstrap_estimate(self, values: np.ndarray,
                          estimator: Callable[[np.ndarray], float] = np.mean
                          ) -> Tuple[float, float]:
        """Calculate confidence intervals using bootstrapping."""
        if len(values) == 0:
            return 0.0, 0.0
        
        bootstrap_estimates = []
        for _ in range(self.n_bootstrap):
            resampled = self._rng.choice(values, size=len(values), replace=True)
            bootstrap_estimates.append(estimator(resampled))
            
        alpha = 1 - self.confidence_level
        return np.percentile(bootstrap_estimates, [alpha/2 * 100, (1-alpha/2) * 100])

    @abstractmethod
    def sample(self, data: DataFrame, size: int, **kwargs) -> DataFrame:
        """Take a sample from the data."""
        pass
        
    @abstractmethod
    def estimate_error_bounds(self, sample: DataFrame, column: str,
                          population_size: Optional[int] = None) -> ErrorBounds:
        """Calculate error bounds for the sample."""
        pass

class SimpleRandomSampler(AbstractSampler):
    """Simple random sampling with vectorized operations."""
    
    def sample(self, data: DataFrame, size: int) -> DataFrame:
        """Take a simple random sample from the data."""
        indices = self._rng.choice(len(data), size=size, replace=False)
        return data.iloc[indices].copy()
    
    def estimate_error_bounds(self, sample: DataFrame, column: str,
                          population_size: Optional[int] = None) -> ErrorBounds:
        """Calculate error bounds for the sample mean."""
        values = sample[column].to_numpy()
        estimate = np.mean(values)
        
        if self.error_method == "bootstrap":
            lower, upper = self._bootstrap_estimate(values)
        else:
            std_error = np.std(values, ddof=1) / np.sqrt(len(values))
            if population_size is not None:
                # Apply finite population correction
                fpc = np.sqrt((population_size - len(values)) / (population_size - 1))
                std_error *= fpc
                
            if self.error_method == "student_t":
                t_value = stats.t.ppf((1 + self.confidence_level) / 2, df=len(values)-1)
                margin = t_value * std_error
            else:  # normal distribution
                z_value = stats.norm.ppf((1 + self.confidence_level) / 2)
                margin = z_value * std_error
                
            lower = estimate - margin
            upper = estimate + margin
            
        return ErrorBounds(
            estimate=estimate,
            lower_bound=lower,
            upper_bound=upper,
            confidence_level=self.confidence_level,
            sample_size=len(values),
            population_size=population_size,
            method=self.error_method,
            distribution_type=DistributionType.from_str(self.error_method)
        )

class StratifiedSampler(AbstractSampler):
    """Stratified sampling with optimal allocation."""
    
    def sample(self, data: DataFrame, size: int, strata_column: str,
             allocation: str = "proportional") -> DataFrame:
        """Take a stratified sample from the data."""
        strata = data[strata_column].unique()
        stratum_data = {s: data[data[strata_column] == s] for s in strata}
        
        if allocation == "optimal":
            # Neyman allocation using stratum standard deviations
            stratum_std = {s: df['value'].std() for s, df in stratum_data.items()}
            total_std = sum(std * len(df) for std, (_, df) 
                          in zip(stratum_std.values(), stratum_data.items()))
            stratum_sizes = {
                s: int(size * (std * len(df)) / total_std)
                for s, std, (_, df) in zip(strata, stratum_std.values(), stratum_data.items())
            }
        else:  # proportional allocation
            stratum_sizes = {
                s: int(size * len(df) / len(data))
                for s, df in stratum_data.items()
            }
            
        # Adjust for rounding errors
        total = sum(stratum_sizes.values())
        if total < size:
            # Add remaining samples to largest stratum
            largest = max(stratum_data.items(), key=lambda x: len(x[1]))[0]
            stratum_sizes[largest] += size - total
            
        samples = []
        for stratum, stratum_size in stratum_sizes.items():
            if stratum_size > 0:
                stratum_sample = SimpleRandomSampler(self.config).sample(
                    stratum_data[stratum], stratum_size
                )
                samples.append(stratum_sample)
                
        return pd.concat(samples, ignore_index=True)
    
    def estimate_error_bounds(self, sample: DataFrame, column: str,
                          population_size: Optional[int] = None,
                          strata_column: Optional[str] = None) -> ErrorBounds:
        """Calculate error bounds for stratified sample mean."""
        if strata_column is None:
            # If no strata column is specified, fall back to simple random sampling bounds
            return super().estimate_error_bounds(sample, column, population_size)
            
        if self.error_method == "bootstrap":
            # For bootstrap, we can use the simple method
            values = sample[column].to_numpy()
            estimate = np.mean(values)
            lower, upper = self._bootstrap_estimate(values)
        else:
            # For parametric methods, use stratified variance formula
            strata = sample.groupby(strata_column)
            stratum_means = strata[column].mean()
            stratum_vars = strata[column].var(ddof=1)
            stratum_sizes = strata.size()
            
            # Calculate stratified mean and variance
            if population_size is not None:
                stratum_weights = stratum_sizes / population_size
            else:
                stratum_weights = stratum_sizes / len(sample)
                
            estimate = np.sum(stratum_means * stratum_weights)
            var_estimate = np.sum(stratum_vars * (stratum_weights ** 2) / stratum_sizes)
            
            if population_size is not None:
                # Apply finite population correction
                var_estimate *= (population_size - len(sample)) / (population_size - 1)
                
            std_error = np.sqrt(var_estimate)
            
            if self.error_method == "student_t":
                # Use conservative degrees of freedom (smallest stratum - 1)
                df = min(sizes - 1 for sizes in stratum_sizes)
                t_value = stats.t.ppf((1 + self.confidence_level) / 2, df=df)
                margin = t_value * std_error
            else:  # normal distribution
                z_value = stats.norm.ppf((1 + self.confidence_level) / 2)
                margin = z_value * std_error
                
            lower = estimate - margin
            upper = estimate + margin
            
        return ErrorBounds(
            estimate=estimate,
            lower_bound=lower,
            upper_bound=upper,
            confidence_level=self.confidence_level,
            sample_size=len(sample),
            population_size=population_size,
            method=self.error_method,
            distribution_type=DistributionType.from_str(self.error_method)
        )
