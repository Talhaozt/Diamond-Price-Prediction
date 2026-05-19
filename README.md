#  Diamond Price Prediction with Support Vector Regressor (SVR)

This repository contains an end-to-end Machine Learning project focused on predicting diamond prices based on their structural, physical, and quality attributes using the famous **Diamonds Dataset**. The project involves comprehensive exploratory data analysis, robust preprocessing, and hyperparameter tuning using Support Vector Regression (SVR).

##  Project Overview
Predicting the market value of a diamond is a complex task because it relies heavily on non-linear combinations of features like the **4Cs** (Carat, Cut, Color, Clarity) alongside precise physical dimensions. In this project, we built a highly optimized regression model capable of capturing these non-linear relationships to output accurate price estimations.

---

##  Data Pipeline & Architecture

### 1. Exploratory Data Analysis (EDA)
- Analyzed the distribution of the target variable (`price`) and independent features.
- Visualized correlations using Seaborn heatmaps to detect multi-collinearity among physical dimensions (`x`, `y`, `z`).

### 2. Outlier Detection & Removal
- Investigated structural anomalies and measurement errors within the dataset (e.g., non-zero values for dimensions and extreme outliers in spatial variables).
- Cleaned the dataset using specific boolean masking boundaries to ensure the mathematical stability of the SVR algorithm.

### 3. Feature Engineering & Encoding
- **Label / Ordinal Encoding:** Applied to categorical features with intrinsic hierarchy (`cut`, `color`, `clarity`) to preserve their quality rankings.
- **Feature Scaling:** Since SVR maximizes margins using distance-based metrics, all numerical characteristics were standardized to prevent feature dominance.

### 4. Model Training & Hyperparameter Tuning
- Implemented **Support Vector Regression (SVR)** to model complex non-linear spatial boundaries.
- Conducted exhaustive hyperparameter optimization using **GridSearchCV** with **5-Fold Cross-Validation** (`n_jobs=-1` for parallel processing) across the following parameters:
  - `kernel`: Tested both `linear` and non-linear `rbf` (Radial Basis Function) kernels.
  - `C`: Tuned the regularization boundary parameter.
  - `gamma`: Optimized the kernel coefficient for the RBF spatial mapping.

---

