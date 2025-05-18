# Mobile Addiction Behavioral Analysis

This repository presents a comprehensive machine learning pipeline designed to analyze patterns of smartphone and social media overuse. The objective is to build predictive models that quantify and classify addiction severity based on behavioral and demographic indicators.

---

## Project Scope

The project leverages a publicly available dataset from Kaggle to investigate:

* Behavioral signals associated with excessive smartphone usage.
* Predictive modeling of self-reported addiction scores.
* Classification of users into addicted vs. non-addicted categories.
* Feature importance interpretation and exploration of underlying usage patterns.

---

## Dataset Overview

* **Source**: Kaggle - Mobile Addiction Dataset
* **Features**: Includes screen time, app usage, notifications, stress levels, demographics, and more.
* **Quality**: Clean, no missing values; may include synthetically generated data (R² = 1 observed in some models).

---

## Methodology

### Data Processing

* **Cleaning**: Removal of irrelevant and non-numeric columns.
* **Splitting**: 80/20 train-test division using `train_test_split`.
* **Scaling**: Standardization via `StandardScaler` to ensure model robustness.

### Modeling Approaches

#### Regression Models

* Linear Regression
* Polynomial Regression (built-in & manual)
* Ridge Regression (built-in & manual)
* Decision Tree Regressor
* Locally Weighted Regression

#### Classification Models

* K-Nearest Neighbors (manual)
* Logistic Regression
* Multi-layer Perceptron (MLP)
* XGBoost
* Quantum Support Vector Machine (QSVM)

---

## Key Results

### Regression

| Model                   | R² Score | MSE    |
| ----------------------- | -------- | ------ |
| Linear Regression       | 1.0000   | 0.0000 |
| Ridge Regression        | 1.0000   | 0.0000 |
| Polynomial Regression   | 0.9999   | 0.2331 |
| Decision Tree Regressor | 0.8245   | 52.77  |
| Locally Weighted        | 0.9882   | 3.5343 |

### Classification

| Model               | Accuracy | Precision | Recall | F1 Score |
| ------------------- | -------- | --------- | ------ | -------- |
| KNN                 | 97%      | 98%       | 96%    | 97%      |
| Logistic Regression | 97%      | 98%       | 96%    | 97%      |
| MLP                 | 98%      | 97%       | 97%    | 97%      |
| XGBoost             | 98%      | 98%       | 97%    | 98%      |
| QSVM                | 69%      | —         | —      | —        |

Note: Gradient boosting exhibited the strongest performance overall with the highest ROC AUC.

---

## Insights

* Nighttime usage was the most influential predictor across all models.
* Younger individuals were more prone to addiction patterns.
* The models exhibited high consistency in identifying digital engagement features as key predictors.

---

## Usage Instructions

### Environment Setup

Install required Python packages:

```bash
pip install -r requirements.txt
```

### Launch Streamlit App

```bash
streamlit run Classification_App.py
```

### Run Models in Jupyter

Use the provided notebooks to explore or reproduce results:

* `regression_model.ipynb`
* `Classification Model on mobile addiction data.ipynb`

---

## Contributors

* Salah El-Din Sayed – Team Lead, Classification Models, App Development
* Omar El-Nabarawy – Data Acquisition and Preparation
* Abdulrahman Abougendia – Data Cleaning, Regression Models
* Soliman El-Hasanen – Manual Regression, Decision Tree, Clustering
* Mohamed Farouk – Documentation, Data Organization
* Fouad Hashesh – QSVM Implementation, Classification Assistance

---

## Acknowledgements

This project was developed as part of the CSE 271: Data Science course at the Egypt University of Informatics under the mentorship of Prof. Fatma, whose guidance was invaluable throughout the semester.

---

## References

* Keles, McCrae, & Grealish (2020) — Psychological effects of social media.
* Twenge et al. (2018) — Sleep disruption and academic effects.
* Yildirim & Correia (2015) — Nomophobia study.

---

## Future Work

* Integration of clustering results into interactive applications.
* Real-time data integration from mobile APIs.
* Further exploration of neural network architectures for temporal behavioral data.
