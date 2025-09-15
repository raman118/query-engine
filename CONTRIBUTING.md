# Contributing to Approximate Query Engine

Thank you for your interest in contributing! This document provides guidelines and information for contributors.

## 🎯 How to Contribute

### 🐛 Reporting Bugs
- **Search existing issues** before creating new ones
- **Use the bug report template** with detailed information
- **Include reproduction steps** and system information
- **Provide error messages** and stack traces when applicable

### ✨ Suggesting Features
- **Check the roadmap** to see if it's already planned
- **Use the feature request template** with clear use cases
- **Explain the problem** your feature would solve
- **Consider backward compatibility** implications

### 🔧 Development Workflow

#### 1. Setup Development Environment
```bash
# Fork the repository on GitHub
git clone https://github.com/YOUR_USERNAME/approx-query-engine.git
cd approx-query-engine

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install in development mode
pip install -e .
pip install -r requirements.txt

# Install development dependencies
pip install pytest pytest-cov black isort mypy flake8
```

#### 2. Create Feature Branch
```bash
git checkout -b feature/your-feature-name
# or
git checkout -b bugfix/issue-number
```

#### 3. Make Changes
- **Write tests first** (TDD approach recommended)
- **Follow code style guidelines** (see below)
- **Add documentation** for new features
- **Update type hints** for all public APIs

#### 4. Run Tests
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# Check type hints
mypy .

# Format code
black .
isort .

# Check style
flake8 .
```

#### 5. Submit Pull Request
- **Write descriptive commit messages**
- **Reference related issues** (#123)
- **Include tests** for new functionality
- **Update documentation** as needed

## 📝 Code Style Guidelines

### Python Style
- **Follow PEP 8** with 88-character line limit
- **Use Black** for code formatting
- **Use isort** for import organization
- **Add type hints** for all functions and classes
- **Write docstrings** in Google style

### Example Code Style
```python
from typing import List, Optional, Union, Tuple
import pandas as pd


def sample_data(
    data: pd.DataFrame,
    size: int,
    method: str = "random",
    confidence_level: float = 0.95
) -> Tuple[pd.DataFrame, "ErrorBounds"]:
    """Sample data using specified method with error bounds.
    
    Args:
        data: Input DataFrame to sample from
        size: Number of samples to draw
        method: Sampling method ('random', 'stratified', 'reservoir')
        confidence_level: Confidence level for error bounds
        
    Returns:
        Tuple of sampled data and error bounds
        
    Raises:
        ValueError: If size is larger than data length
        
    Example:
        >>> df = pd.DataFrame({'x': range(1000)})
        >>> sample, bounds = sample_data(df, 100)
        >>> print(f"Sample size: {len(sample)}")
    """
    if size > len(data):
        raise ValueError("Sample size cannot exceed data length")
    
    # Implementation here...
    return sample, bounds
```

## 🧪 Testing Guidelines

### Test Structure
```
tests/
├── conftest.py              # Shared fixtures
├── test_sampling.py         # Sampling algorithm tests
├── test_error_bounds.py     # Statistical validation tests
├── test_performance.py      # Performance benchmarks
├── test_integration.py      # End-to-end tests
└── test_edge_cases.py       # Boundary condition tests
```

### Writing Tests
- **Use pytest fixtures** for common setup
- **Test edge cases** and error conditions
- **Include performance tests** for critical paths
- **Mock external dependencies** appropriately
- **Add statistical tests** for sampling algorithms

## 💬 Community Guidelines

### Communication
- **Be respectful** and inclusive in all interactions
- **Ask questions** if anything is unclear
- **Share knowledge** and help other contributors
- **Follow the code of conduct**

### Getting Help
- **GitHub Discussions** for general questions
- **Issue tracker** for bugs and feature requests
- **Pull request reviews** for code feedback

---

Thank you for contributing to the Approximate Query Engine project! Your contributions help advance the state of approximate query processing and benefit the entire data science community.