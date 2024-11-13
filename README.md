# House Price Prediction

## Project Overview

This project aims to predict **house sale prices** using the **Ames Housing Dataset**. Multiple machine learning models, including **Random Forest Regressor** and **Linear Regression**, are utilized to predict the **SalePrice** based on various features related to the properties.

## Key Features

- **Data Cleaning & Preprocessing**: 
  - Handled missing values, removed low-variance features, and imputed data using mean/mode.
  - Label encoding was applied to categorical variables.
- **Feature Engineering**: 
  - Created new features such as `TotalSF` (total square footage) and `Age`.
  - Log-transformed `SalePrice` for normality.
- **Exploratory Data Analysis (EDA)**: 
  - Analyzed distribution of features, missing data, and correlation with `SalePrice`.
  - Visualizations included histograms, scatter plots, and heatmaps for understanding key relationships.
- **Model Training**: 
  - Implemented **Random Forest Regressor** and **Linear Regression** to predict `SalePrice`.
  - Evaluated models using **Root Mean Squared Error (RMSE)** and **R-squared** scores.
- **Model Evaluation & Submission**: 
  - The trained Random Forest model was used to make predictions on the test set and a submission file was created.

## Dependencies

- **Programming Language**: Python 3.x
- **Libraries**:
  - Data Manipulation: `pandas`, `numpy`
  - Visualization: `seaborn`, `matplotlib`
  - Machine Learning: `scikit-learn`
