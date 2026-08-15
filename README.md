# 💳 Credit Card Fraud Detection using Machine Learning

A machine learning project for detecting fraudulent credit card transactions under severe class imbalance.

## 📌 Project Overview

Credit card fraud detection is a challenging binary classification problem because fraudulent transactions represent only a very small fraction of all transactions.

This project explores multiple machine learning approaches and evaluates their ability to identify fraudulent transactions while controlling false positives.

The models evaluated are:

- Logistic Regression
- Class-weighted Logistic Regression
- SMOTE + Logistic Regression
- Random Forest
- XGBoost

The project also includes threshold optimization and SHAP-based model explainability.

---

## 📊 Dataset

The dataset contains:

- **284,807 transactions**
- **30 input features**
- **1 target variable (`Class`)**

Target:

- `0` → Legitimate transaction
- `1` → Fraudulent transaction

The dataset is highly imbalanced:

- Legitimate: **284,315 (99.827%)**
- Fraud: **492 (0.173%)**

The features `V1`–`V28` are anonymized PCA-derived components.

> The dataset is not included in this repository due to its size. Place `creditcard.csv` in the working directory before running the notebook.

---

## 🔬 Methodology

### 1. Exploratory Data Analysis

The project investigates:

- Class distribution
- Transaction amount distribution
- Transaction timing
- Feature correlations
- Fraud vs legitimate transaction patterns

### 2. Feature Engineering

A derived `Hour` feature was created from the original `Time` variable to investigate whether transaction timing contributes to fraud detection.

### 3. Data Splitting

Stratified sampling was used to preserve the minority-class distribution.

The data was divided into:

- Training set
- Validation set
- Untouched test set

The validation set was used for decision-threshold selection, while the test set was reserved for final evaluation.

### 4. Class Imbalance

Several approaches were evaluated:

- Baseline Logistic Regression
- Class-weighted Logistic Regression
- SMOTE
- Tree-based ensemble models

### 5. Models

#### Logistic Regression

Used as the baseline classification model.

#### Random Forest

Used to capture nonlinear relationships and interactions between features.

#### XGBoost

Used as a gradient-boosting approach for comparison with Random Forest.

### 6. Threshold Optimization

Instead of assuming a default probability threshold of 0.5, thresholds were selected using the validation set to improve the precision-recall trade-off.

Final thresholds:

- Random Forest: **0.26**
- XGBoost: **0.987879**

The selected thresholds were then applied to the untouched test set.

### 7. Explainability

SHAP was used to investigate which features contributed most strongly to the XGBoost model's predictions.

---

## 🏆 Final Results

| Model | Precision | Recall | F1 Score | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| **Random Forest** | **87.2%** | **83.7%** | **85.2%** | 95.80% | **85.83%** |
| XGBoost | 88.0% | 75.5% | 81.0% | **97.86%** | 85.74% |

### Final Model: Random Forest

Random Forest was selected as the preferred model because it achieved the strongest overall precision-recall performance.

On the untouched test set:

- **Precision:** 87.2%
- **Recall:** 83.7%
- **F1 Score:** 85.2%
- **ROC-AUC:** 95.80%
- **PR-AUC:** 85.83%

### Final Confusion Matrix

| | Predicted Legitimate | Predicted Fraud |
|---|---:|---:|
| **Actual Legitimate** | 56,852 | 12 |
| **Actual Fraud** | 16 | 82 |

The model correctly detected **82 of 98 fraudulent transactions** while incorrectly flagging **12 legitimate transactions**.

---

## 🧠 Explainability

SHAP analysis identified several influential features in the XGBoost model, including:

- V14
- V4
- V10
- V12
- V8
- V11

Because the V1–V28 variables are anonymized PCA-derived features, their importance should be interpreted as **predictive contribution rather than causal relationships**.

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn
- XGBoost
- SHAP
- Google Colab

---

## 📁 Repository Structure

```text
credit-card-fraud-detection/
│
├── Credit_Card_Fraud_Detection_using_Machine_Learning.ipynb
├── README.md
└── requirements.txt
```

## 🚀 Future Work

Potential extensions include:

- Hyperparameter tuning using cross-validation
- Cost-sensitive threshold optimization
- Model calibration
- Real-time fraud detection API
- Deployment as a web application
- Model monitoring and concept-drift detection
- Evaluation on newer transaction datasets

---

## 👩‍💻 Author

**Yashica G**

Artificial Intelligence & Machine Learning
