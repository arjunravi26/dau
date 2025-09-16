# DAU: Density-Aware Undersampling

[![PyPI version](https://img.shields.io/pypi/v/dau-undersampling.svg)](https://pypi.org/project/dau-undersampling/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

`dau-undersampling` is a Python package that implements **Density-Aware Undersampling (DAU)**, a novel undersampling technique for handling imbalanced datasets.
It intelligently reduces the majority class by:
- Keeping sparse (boundary / hard-to-learn) points.
- Clustering dense regions using **DBSCAN** and selecting one representative per cluster.
- Preserving noise and outliers.

This ensures that your classifier learns from diverse and informative samples without being overwhelmed by redundant majority-class points.

---

## 🚀 Installation

```bash
pip install dau-undersampling
