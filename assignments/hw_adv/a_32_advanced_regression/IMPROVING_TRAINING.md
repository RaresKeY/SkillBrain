# 5 Major Strategies to Improve the Real Estate Price Prediction Pipeline

This document outlines the five most impactful changes that can be implemented to significantly boost the performance, robustness, and accuracy of the current machine learning pipeline.

## 1. Target Variable Transformation (Log-Scaling)

**Why:** Real estate prices typically follow a **right-skewed distribution** (long tail of expensive properties). Linear models (and even tree-based models to some extent) struggle with skewed targets because they assume errors are normally distributed.

**How:**
- Apply a **Log Transformation** (`np.log1p`) to the target variable (`pret`) before training.
- Train the models on the log-transformed prices.
- Apply the inverse transformation (`np.expm1`) to the predictions to get the actual price values back.
- **Expected Impact:** Reduced impact of expensive outliers and better linearity, leading to lower RMSE.

## 2. Advanced Feature Engineering & Selection

**Why:** The current model relies mostly on raw features. Models (especially Linear Regression) cannot easily capture complex, non-linear relationships without explicit help.

**How:**
- **Interaction Terms:** Create features like `price_per_sqm_by_zone` or `suprafata * numar_camere`.
- **Polynomial Features:** Add squared terms for continuous variables like `suprafata` to capture non-linear growth.
- **Binning:** Group `an_constructie` into eras (e.g., "Pre-1977", "1977-1990", "New Development").
- **Feature Selection:** Use Recursive Feature Elimination (RFE) or SelectFromModel to remove noisy or irrelevant features that confuse the model.

## 3. Gradient Boosting Models (XGBoost / LightGBM / CatBoost)

**Why:** While Random Forest is a strong baseline, **Gradient Boosting Machines (GBMs)** are the state-of-the-art for tabular data. They build trees sequentially, correcting the errors of previous trees, which often yields higher accuracy than independent random trees.

**How:**
- Replace or augment Random Forest with **XGBoost**, **LightGBM**, or **CatBoost**.
- These libraries handle missing values natively and often require less preprocessing (CatBoost handles categorical features automatically).
- **Expected Impact:** Significant drop in bias and improved generalization on the test set.

## 4. Bayesian Hyperparameter Optimization (Optuna)

**Why:** The current pipeline uses basic `GridSearchCV` (for Ridge) or manual parameters (for Random Forest). This is inefficient and likely misses the optimal configuration.

**How:**
- Implement **Optuna** or `RandomizedSearchCV`.
- Search a wider space for critical parameters:
    - **Random Forest:** `n_estimators`, `max_depth`, `min_samples_split`, `max_features`.
    - **Ridge/Lasso:** `alpha` (regularization strength).
- **Expected Impact:** Finding the "sweet spot" for regularization that minimizes overfitting while maximizing predictive power.

## 5. Robust Evaluation with K-Fold Cross-Validation

**Why:** A single 80/20 train-test split is sensitive to randomness. The current test score might be lucky (or unlucky) depending on which specific houses ended up in the test set.

**How:**
- Switch to **K-Fold Cross-Validation** (e.g., 5 or 10 folds).
- Train the model 5 times on different subsets of data and average the scores.
- **Expected Impact:** A much more reliable estimate of the model's true performance on unseen data, ensuring that improvements are real and not just random noise.
