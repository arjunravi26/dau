
# DAU Undersampling (Density-Aware Undersampling)

**DAU (Density-Aware Undersampling)** is a Python package to handle imbalanced datasets by reducing the majority class *without losing important information*.

Instead of random undersampling, DAU keeps:

* **Sparse points (outliers / rare cases)** → retained fully
* **Dense clusters** → represented by a few points (using DBSCAN)
* **Noise points** → kept separately

This leads to **smarter undersampling** and better ML performance compared to random undersampling.


---

## Installation

From [PyPI](https://pypi.org/project/dau-undersampler/):

```bash
pip install dau-undersampling
```

(Optionally, for testing on [TestPyPI](https://test.pypi.org/)):

```bash
pip install -i https://test.pypi.org/simple/ dau-undersampling
```

---

## ⚡ Quickstart

```python
import pandas as pd
from sklearn.datasets import make_classification
from dau_undersampling import DAU

# 1. Create an imbalanced dataset
X, y = make_classification(
    n_samples=1000, n_features=10,
    n_classes=2, weights=[0.9, 0.1],
    random_state=42
)

X = pd.DataFrame(X)
y = pd.Series(y)

# 2. Apply DAU undersampling
dau = DAU(n_neighbors=5, min_samples=3, eps=0.5, percentile=25)
X_resampled, y_resampled = dau.fit_transform(X, y)

print("Original dataset shape:", y.value_counts().to_dict())
print("Resampled dataset shape:", y_resampled.value_counts().to_dict())
```

---

## 🛠 Usage & Parameters

### Class: `DAU`

```python
DAU(n_neighbors=3, min_samples=5, eps=0.05, percentile=25)
```

### Parameters:

* **`n_neighbors`** *(int, default=3)*
  Number of neighbors for KNN distance calculation.
* **`min_samples`** *(int, default=5)*
  Minimum samples per cluster (DBSCAN).
* **`eps`** *(float, default=0.05)*
  Maximum neighborhood radius (DBSCAN).
* **`percentile`** *(int, default=25)*
  Threshold to split sparse vs dense points.

---

###  Method: `fit_transform(X, y)`

Performs density-aware undersampling.

**Arguments:**

* `X`: `pd.DataFrame` → features of majority class or dataset.
* `y`: `pd.Series` → labels (binary classification).

**Returns:**

* `X_resampled`: Reduced features after undersampling.
* `y_resampled`: Reduced labels aligned with features.

---

## Example in Pipeline

You can also integrate DAU into an ML pipeline (with `imblearn`):

```python
from imblearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ('undersample', DAU(n_neighbors=7, min_samples=5, eps=0.4, percentile=30)),
    ('clf', LogisticRegression())
])

pipeline.fit(X, y)
```

---

## Why DAU vs Other Methods?

| Method                 | Behavior                                                                  |
| ---------------------- | ------------------------------------------------------------------------- |
| Random undersampling   | Drops samples randomly (risk of losing rare but important cases).         |
| NearMiss / Tomek Links | Works with distances but may remove outliers or boundary points.          |
| **DAU (this package)** | Preserves outliers + keeps 1 representative per dense cluster (balanced). |

---

## Contributing

1. Fork this repo
2. Create a new branch (`git checkout -b feature-xyz`)
3. Commit changes (`git commit -m "Added xyz"`)
4. Push (`git push origin feature-xyz`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

