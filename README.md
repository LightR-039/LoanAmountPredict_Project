# Machine Learning Project
## Loan Amount Prediction

### Project Overview
This project implements two regression models—Random Forest and Linear Regression—to predict the loan amount using the Loan Approval Classification Dataset from Kaggle. While the dataset is designed for classifying loan approval status, this project takes a novel approach by repurposing it to predict the continuous loan amount based on features related to credit profiles, demographic profiles, and loan-related data. The models are developed in Python using scikit-learn within Jupyter notebooks. The Random Forest model achieves an R² score of 0.77 and an RMSE of 2755.16, while the Linear Regression model achieves an R² score of 0.62 and an RMSE of 3535.29.

### Dataset
The dataset used in this project is the Loan Approval Classification Dataset from Kaggle. It contains 45,000 records and 14 variables, including features related to credit profiles (e.g., credit score, payment history), demographic profiles (e.g., age, income), and loan-related data (e.g., loan term, interest rate). Originally intended for classifying loan approval status, this project uses the dataset for regression to predict the loan amount, a continuous target variable. The dataset is provided in CSV format and should be placed in the data/ directory (not tracked in Git). Download the dataset from the Kaggle link and place it in the data/ folder before running the notebooks. Refer to the Kaggle dataset page for licensing and usage terms.

[Data Source](https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data)

### Features
- Preprocessing:
  - Checks for missing values and removes or imputes them as needed.
  - Identifies and eliminates duplicate records to ensure data quality.
  - Imputes loan_amnt for non-approved loans (loan_status == 0) using KNeighborsRegressor trained on approved loans (loan_status == 1) with features: person_income, credit_score, loan_int_rate, person_education, person_home_ownership_RENT, loan_intent_VENTURE, mitigating noise in the target variable.
- Encodes categorical variables:
  - Binary encoding for binary features like gender and loan default history.
  - Ordinal encoding for education level to preserve order.
  - One-hot encoding for multi-category features like home ownership and loan intention.
  - Handles outliers in numerical features (e.g., income, age, loan interest rate, loan_percent_income) using robust statistical methods.
- Exploratory Data Analysis (EDA):
  - Inspects distributions of all features to understand data characteristics.
  - Creates catplots and subplots using seaborn and matplotlib to visualize relationships between features and both loan status and loan amount.
- Model Training:
  - Applies RobustScaler to numerical features to handle outliers and ensure consistent scaling.
  - Selects relevant features and tunes hyperparameters to prevent overfitting in Random Forest and Linear Regression models.
- Visualization:
  - Generates visualizations of model performance, including R² scores, RMSE, residual plots, and other metrics to evaluate and compare model effectiveness.
 
### Installation

#### Prerequisites

- **Python**: Version 3.11.12
- **pip**: For package management
- **Jupyter Notebook**: For running the analysis
- **Key Packages**:

| Package           | Version |
|-------------------|---------|
| pandas            | 2.2.2   |
| numpy             | 2.0.2   |
| matplotlib        | 3.10.0  |
| seaborn           | 0.13.2  |
| scikit-learn      | 1.6.1   |
| category_encoders | 2.8.1   |
| joblib            | 1.4.2   |

### Results

The performance of the Random Forest and Linear Regression models is evaluated using multiple metrics on the test set, with cross-validation R² scores for robustness. A baseline model (mean prediction) is included for comparison.

| Model             | Train R² | Test R² | MSE         | RMSE    | MAE      | Median AE | R² Score | CV R²  |
|-------------------|----------|---------|-------------|---------|----------|-----------|----------|--------|
| Linear Regression | 0.6239   | 0.6229  | 12,498,270  | 3535.29 | 2708.38  | 2075.58   | 0.6229   | 0.6227 |
| Random Forest     | 0.7990   | 0.7710  | 7,590,916   | 2755.16 | 1997.61  | 1395.97   | 0.7710   | 0.7658 |

**Baseline (Mean Prediction)**:
- Mean Squared Error (MSE): 33,147,846.34
- Mean Absolute Error (MAE): 4,677.43
- R² Score: -0.00

![Model Result](https://github.com/user-attachments/assets/300dd231-fc43-4fd7-a2da-8f628303350b)

#### Learning Curve
The following learning curve illustrates how the R² score varies with training set size for the Random Forest and Linear Regression models:

![Learning Curve](https://github.com/user-attachments/assets/3abab688-10f6-4265-9816-af5057484992)


Additional visualizations are available in `notebooks/Loan_Amount_prediction.ipynb`.

### Acknowledgments
- Thanks to scikit-learn for the machine learning framework.
- Dataset provided by Kaggle: Loan Approval Classification Dataset.
- Visualization tools powered by seaborn and matplotlib.
- Additional tools: category_encoders for encoding, joblib for model persistence, and comprehensive dependencies managed via requirements.txt.
