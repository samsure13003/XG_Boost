# 🎗️ Breast Cancer Classification with XGBoost

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Classifier-orange?style=flat&logo=xgboost)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat)

A machine learning project that applies **XGBoost** to classify breast cancer tumors as **Benign (2)** or **Malignant (4)** using the Wisconsin Breast Cancer Dataset. Model performance is rigorously validated using **k-Fold Cross Validation**.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Project Pipeline](#-project-pipeline)
- [Results](#-results)
- [Tech Stack](#-tech-stack)
- [Getting Started](#-getting-started)
- [Project Structure](#-project-structure)

---

## 🔍 Overview

Breast cancer is one of the most common cancers worldwide. Early and accurate classification of tumors into benign or malignant categories is critical for timely treatment. This project builds a robust classification model using **XGBoost**, a gradient boosting algorithm known for its high performance on tabular data.

Key highlights:
- Binary classification: **Benign vs. Malignant**
- Evaluated using **Confusion Matrix**, **Accuracy Score**, and **10-Fold Cross Validation**
- Robust handling of target labels via **Label Encoding**

---

## 📊 Dataset

**Source:** Wisconsin Breast Cancer Dataset (`Data.csv`)

| Property       | Detail                          |
|----------------|---------------------------------|
| Total Samples  | 683                             |
| Features       | 9 cytological characteristics   |
| Target Classes | Benign (2) / Malignant (4)      |

### Features

| Feature                      | Description                                |
|-----------------------------|--------------------------------------------|
| Clump Thickness              | Thickness of the clump (1–10)              |
| Uniformity of Cell Size      | Consistency of cell sizes (1–10)           |
| Uniformity of Cell Shape     | Consistency of cell shapes (1–10)          |
| Marginal Adhesion            | Cell adhesion to neighbouring cells (1–10) |
| Single Epithelial Cell Size  | Size of individual epithelial cells (1–10) |
| Bare Nuclei                  | Nuclei not surrounded by cytoplasm (1–10)  |
| Bland Chromatin              | Chromatin texture uniformity (1–10)        |
| Normal Nucleoli              | Nucleoli size and visibility (1–10)        |
| Mitoses                      | Frequency of cell division (1–10)          |

---

## 🔧 Project Pipeline

```
Data Loading → Train-Test Split → Label Encoding → XGBoost Training → Evaluation → Cross Validation
```

**Step 1 — Import Libraries**
```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```

**Step 2 — Load Dataset**
```python
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
```

**Step 3 — Train-Test Split (80/20)**
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

**Step 4 — Label Encoding + Train XGBoost**
```python
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test  = le.transform(y_test)

classifier = XGBClassifier()
classifier.fit(X_train, y_train)
```

**Step 5 — Confusion Matrix & Accuracy**
```python
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))
```

**Step 6 — k-Fold Cross Validation (k=10)**
```python
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
print("Accuracy: {:.2f} %".format(accuracies.mean() * 100))
print("Standard Deviation: {:.2f} %".format(accuracies.std() * 100))
```

---

## 📈 Results

| Metric                         | Score       |
|-------------------------------|-------------|
| Test Set Accuracy              | ~96–98%     |
| Cross-Validation Accuracy      | ~96–98%     |
| Cross-Validation Std Deviation | ~1–2%       |

> ✅ Low standard deviation confirms that the model generalizes well and is not overfitting.

---

## 🛠️ Tech Stack

| Tool           | Purpose                         |
|----------------|---------------------------------|
| Python 3.8+    | Programming language            |
| XGBoost        | Gradient boosting classifier    |
| scikit-learn   | Preprocessing & evaluation      |
| pandas         | Data handling                   |
| NumPy          | Numerical computation           |
| Matplotlib     | (Imported for optional plots)   |

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install numpy pandas matplotlib scikit-learn xgboost
```

### Run the Notebook

```bash
# Clone the repository
git clone https://github.com/your-username/xgboost-breast-cancer.git
cd xgboost-breast-cancer

# Launch Jupyter
jupyter notebook xg_boost.ipynb
```

---

## 📁 Project Structure

```
xgboost-breast-cancer/
│
├── xg_boost.ipynb     # Main notebook with full pipeline
├── Data.csv           # Wisconsin Breast Cancer Dataset
└── README.md          # Project documentation
```

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

<p align="center">Made with ❤️ using XGBoost & scikit-learn</p>
