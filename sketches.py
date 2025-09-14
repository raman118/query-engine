from typing import Any, List
import mmh3
import numpy as np
import pandas as pd

class HyperLogLog:
    def __init__(self, p: int = 14):
        """Initialize HyperLogLog with precision parameter p (4-16)."""
        self.p = p
        self.m = 1 << p
        self.alpha = self._get_alpha(p)
        self.registers = np.zeros(self.m, dtype=np.uint8)

    def _get_alpha(self, p: int) -> float:
        if p <= 4:
            return 0.673
        elif p <= 6:
            return 0.697
        elif p <= 16:
            return 0.709
        return 0.7213 / (1.0 + 1.079 / self.m)

    def add(self, values: Any) -> None:
        """Add a single value or a collection of values to the counter."""
        if isinstance(values, (np.ndarray, pd.Series)):
            # Optimized vectorized implementation for numpy/pandas
            if isinstance(values, pd.Series):
                values = values.values
            # Ensure values are strings before encoding
            encoded_values = np.char.encode(values.astype(str))
            hashes = np.array([mmh3.hash(v, signed=False) for v in encoded_values], dtype=np.uint32)
            
            j = (hashes >> (32 - self.p)).astype(np.int32)
            w = hashes << self.p
            
            # Vectorized rho (leading zero count)
            rho = np.zeros_like(w, dtype=np.uint8)
            rho[w == 0] = 32
            non_zero_w = w[w != 0]
            rho[w != 0] = np.minimum(32, np.floor(np.log2(non_zero_w & -non_zero_w)).astype(np.int8) + 1)

            # Update registers using a group-by maximum operation
            updates = pd.DataFrame({'j': j, 'rho': rho}).groupby('j').max()['rho']
            self.registers[updates.index] = np.maximum(self.registers[updates.index], updates.values)
        else:
            # Original implementation for single values
            x = mmh3.hash(str(values).encode(), signed=False)
            j = x >> (32 - self.p)
            w = x << self.p
            self.registers[j] = max(self.registers[j], self._rho(w))

    def _rho(self, w: int) -> int:
        """Count leading zeros + 1."""
        if w == 0:
            return 32
        return min(32, len(bin(w)) - 2)

    def count(self) -> float:
        """Estimate cardinality."""
        E = self.alpha * self.m * self.m / np.sum(2.0 ** -self.registers)
        if E <= 2.5 * self.m:  # Small range correction
            V = np.sum(self.registers == 0)
            if V > 0:
                return float(self.m * np.log(self.m / V))
        return float(E)

class CountMinSketch:
    def __init__(self, width: int = 2000, depth: int = 5):
        """Initialize Count-Min Sketch with given width and depth."""
        self.width = width
        self.depth = depth
        self.table = np.zeros((depth, width), dtype=np.int32)

    def add(self, key: Any, count: int = 1) -> None:
        """Add a key with given count to the sketch."""
        key_bytes = str(key).encode()
        for i in range(self.depth):
            h = mmh3.hash(key_bytes, seed=i)
            w = abs(h) % self.width
            self.table[i, w] += count

    def estimate(self, key: Any) -> int:
        """Estimate the count of a key."""
        key_bytes = str(key).encode()
        min_count = float('inf')
        for i in range(self.depth):
            h = mmh3.hash(key_bytes, seed=i)
            w = abs(h) % self.width
            min_count = min(min_count, self.table[i, w])
        return int(min_count)

class ReservoirSampler:
    def __init__(self, size: int):
        """Initialize reservoir sampler with given sample size."""
        self.size = size
        self.reservoir = []
        self.count = 0

    def add(self, item: Any) -> None:
        """Add an item to the reservoir sample."""
        self.count += 1
        if len(self.reservoir) < self.size:
            self.reservoir.append(item)
        else:
            j = np.random.randint(0, self.count)
            if j < self.size:
                self.reservoir[j] = item

    def get_sample(self) -> List:
        """Get the current reservoir sample."""
        return self.reservoir.copy()

    def clear(self) -> None:
        """Clear the reservoir."""
        self.reservoir = []
        self.count = 0
