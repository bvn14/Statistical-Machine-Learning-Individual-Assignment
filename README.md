# Customer Value Prediction for a Telecom Company (Regression)

**DA2111 – Statistical & Machine Learning | Individual Assignment**
---

## Project Overview

This project predicts **Customer Value** (a continuous number) for telecom customers. It uses customer details, service usage, and interaction data from the **Iranian Churn Dataset** (UCI Machine Learning Repository).

**Research question:** How can we predict Customer Value based on customer demographics, service usage patterns, and interaction with the service?

**Task type:** Regression

## Dataset

- Source: [Iranian Churn Dataset – UCI](https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset)
- Size: 3,150 customers, 14 columns
- Missing values: none

| Feature | Description |
|---|---|
| Call Failure | Number of failed calls |
| Complains | Whether the customer complained (0/1) |
| Subscription Length | Months of subscription |
| Charge Amount | Charge level (0–9) |
| Seconds of Use | Total seconds of calls |
| Frequency of use | Total number of calls |
| Frequency of SMS | Total number of text messages |
| Distinct Called Numbers | Number of unique numbers called |
| Age Group | Age group (categorical) |
| Tariff Plan | Type of tariff plan |
| Status | Active / not active |
| Age | Age of the customer |
| **Customer Value** | **Target variable** |
| Churn | Whether the customer left (0/1) |

## Project Workflow

1. **Problem definition** – define the target variable, features, and research question.
2. **Data exploration and preprocessing**
   - Checked for missing values (none found).
   - Handled outliers by capping at the 5th and 95th percentiles.
   - Used one-hot encoding for categorical features (`drop_first=True`).
   - Scaled numerical features with `StandardScaler`.
3. **Exploratory Data Analysis (EDA)** – summary statistics, histograms, box plots, and a correlation heatmap.
4. **Model selection** – four regression models, starting with a simple baseline.
5. **Model training and evaluation** – 80/20 train-test split (`random_state=42`). Metrics: MSE, RMSE, MAE, R². Training and test results were compared to check for overfitting.
6. **Feature importance** – for the tree-based models.
7. **Hyperparameter tuning** – `GridSearchCV` with 5-fold cross-validation.
8. **Comparison and conclusion.**

## Models Used

| Model | Why it was chosen |
|---|---|
| Linear Regression | Simple, easy to understand; used as the baseline |
| Decision Tree Regressor | Captures non-linear patterns; handles outliers well |
| Random Forest Regressor | Reduces overfitting by averaging many trees |
| Gradient Boosting Regressor | Learns step by step from errors; strong for complex patterns |

## Results (Test Set)

| Model | MSE (before tuning) | R² (before tuning) | MSE (after tuning) | R² (after tuning) |
|---|---|---|---|---|
| Linear Regression | 6146.69 | 0.97 | – | – |
| Decision Tree | 3838.39 | 0.98 | 4619.93 | 0.98 |
| Random Forest | 2258.73 | 0.99 | 2204.26 | 0.99 |
| **Gradient Boosting** | 1267.96 | 0.99 | **855.39** | **0.9965** |

**Best model:** Gradient Boosting (`learning_rate=0.1`, `max_depth=5`, `min_samples_split=2`, `n_estimators=300`).

## Key Findings

- **Frequency of SMS** is the most important feature (about 86% importance in all three tree models).
- **Frequency of use**, **Seconds of Use**, and **Age** also help, but much less.
- Most other features have very small importance.
- Ensemble models (Random Forest and Gradient Boosting) work better than Linear Regression and a single Decision Tree.

## Technologies

Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Jupyter Notebook

## License

This project was made for academic purposes.
