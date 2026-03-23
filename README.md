# Nonparametric Statistics in Python

A Python study repo for nonparametric statistics, combining theory, implementation, simulation, and visualization.

This project is designed to deepen understanding of nonparametric statistical methods by building them from scratch where reasonable, validating them against standard libraries, and examining their behavior through examples and simulation.

## Project Goals

- Implement core nonparametric methods in Python
- Validate results against SciPy, statsmodels, and scikit-learn where appropriate
- Compare parametric and nonparametric procedures
- Build intuition through plots, simulations, and worked notebooks
- Create a reusable reference repo for nonparametric statistics

## Current Scope

This repo will cover:

1. Empirical CDF and order statistics
2. Sign and rank tests
3. Permutation tests
4. Bootstrap methods
5. Density estimation and KDE
6. Nonparametric regression
7. Power and robustness studies
8. Real-data case studies

## Current Progress

Completed modules:
- Module 1: ECDF and rank-based foundations
- Module 2: Permutation tests
- Module 3: Bootstrap methods
- Module 4: Density estimation and KDE

Next up:
- Module 5: Nonparametric regression

## Repository Structure

```text
nonparametric-statistics-python/
│
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   └── 01_ecdf_and_rank_tests.ipynb
│
├── src/
│   └── nonparametric_stats/
│       ├── __init__.py
│       ├── ecdf.py
│       ├── rank_tests.py
│       ├── permutation.py
│       ├── bootstrap.py
│       ├── kde.py
│       ├── regression.py
│       └── utils.py
│
├── tests/
│   ├── test_ecdf.py
│   └── test_rank_tests.py
│
└── reports/
    └── figures/
```

## Module 1: Foundations

The first module focuses on three foundational nonparametric tools:

- **ECDF** as the canonical nonparametric estimator of a distribution function
- **Sign test** as an exact procedure based only on directional information
- **Mann-Whitney framework** interpreted through pairwise ordering probabilities

## Development Approach

Each module in the repo follows the same pattern:

1. implement the core method in `src/`
2. validate behavior with tests
3. demonstrate usage in a notebook
4. interpret results in plain language
5. compare with parametric counterparts when useful

## Why This Repo Exists

Nonparametric methods are often presented as a grab-bag of “alternatives” to classical parametric tests. This repo treats them instead as a coherent part of statistical reasoning: methods for estimation and inference under weaker structural assumptions.

The aim is not only to run the methods, but to understand what they estimate, what assumptions they avoid, and what tradeoffs they introduce.

## Status

Module 1 is in progress.
Next up: permutation tests.

