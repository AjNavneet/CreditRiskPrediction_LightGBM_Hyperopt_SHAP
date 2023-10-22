# Credit Risk Prediction with LightGBM, Hyperopt and SHAP

## Project Overview

### Business Context
Credit risk refers to the potential loss a lender may face when a borrower fails to repay a loan or meet contractual obligations. The goal of a credit risk assessment is to determine if potential borrowers are creditworthy, have the means to repay their debts, and to minimize credit risk. This project involves building a classification model for default prediction using LightGBM, optimizing hyperparameters with Hyperopt, and using SHAP for model explainability.

---

### Aim
The aim of this project is to predict loan defaulters and reduce the risk of financial loss by analyzing credit history, employment, and demographic data.

---

### Data Description
The dataset includes information on 143,727 borrowers, including attributes such as employment type, work experience, income, dependents, total loans, and total payments.

---

### Tech Stack
- Language: `Python`
- Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `lightgbm`, `hyperopt`, `shap`

---

## Approach

1. **Data Reading**
2. **Data Processing**
    - Drop Columns
    - Split Data
    - Define Label
    - Roll Rate Analysis
    - Window Roll Analysis
3. **Feature Engineering**
    - Label
    - % Amount Paid as interest in past Loan Repayment
    - % of Loans defaulted in the last 2 years
4. **Exploratory Data Analysis (EDA)**
    - Univariate Analysis
        - Numerical Summary: Min, Max, Mean, Median, etc.
        - Categorical Summary: Top, Unique, Count, etc.
    - Bivariate Analysis
        - Correlation Plot
        - Box Plots
5. **Target Encoding**
6. **Feature Selection**
    - Random Forest
    - Decision Tree
7. **ML Model Development**
    - LightGBM
    - Hyperparameter Tuning using Hyperopt
8. **Model Evaluation**
    - ROC AUC
    - PRAUC
    - Score Distribution
9. **Feature Importance**
    - Split and Gain
    - SHAP
10. **Class Rate Curve and Right Threshold**

---

## Modular Code Overview

1. `input`: Contains the raw data for analysis, in this case, `credit_risk_data.csv`.
2. `documents`: Contains supporting learning material.
3. `lib`: A reference folder containing the original iPython notebook used in lectures.
4. `ml_pipeline`: Contains functions in various Python files for processing the data and training the model.
5. `output`: The folder where the trained model is saved.
6. `engine.py`: A script that calls the functions in the `ml_pipeline` to run the entire process and save the model.
7. `requirements.txt`: Lists required libraries and versions.
8. `readme.md`: Instructions for running the code.

---

## Concepts Explored

1. Credit risk assessment
2. Understand “default” using Roll Rate Analysis
3. Importance of “days past due” or “dpd” in credit risk
4. Exploratory data analysis for categorical target variable
5. Target Encoding and why is it preference over other encoding techniques
6. Feature Selection with Random Forest and Deep Decision Tree
7. LightGBM and optimize its parameters
8. Hyperopt
9. Model selection on the basis of metrics such as ROC AUC, and PR AUC.
10. Split and Gain its use for Feature Importance
11. SHAP
12. Visualize Feature Importance with SHAP
13. Class Rate Curve
14. Choosing the right threshold for minimum credit loss.

---

