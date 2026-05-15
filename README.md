# India House Price Prediction Using Machine Learning

A Machine Learning project that predicts house prices in India using real-world style housing data and Linear Regression.

This project focuses on preprocessing messy datasets, feature engineering, and training regression models to estimate property prices based on multiple house features.

---

# Project Overview

House prices depend on many factors such as:

- Location
- Number of bedrooms
- Bathrooms
- Area size
- Parking availability
- Property type
- Furnishing status
- Building age

This project explores how Machine Learning can analyze housing data and predict house prices using regression algorithms.

The dataset intentionally contains messy and inconsistent data to simulate real-world datasets used in practical ML projects.

---

# Features

- Real-world style messy dataset
- Data preprocessing
- Missing value handling
- Feature scaling using StandardScaler
- One-hot encoding
- Linear Regression model
- MAE, MSE, and RMSE evaluation
- Beginner-friendly ML workflow

---

# Dataset Features

| Feature | Description |
|---|---|
| city | City name |
| locality | Area/locality |
| property_type | Type of property |
| bedrooms | Number of bedrooms |
| bathrooms | Number of bathrooms |
| area_sqft | Property size in square feet |
| age_years | Property age |
| parking_spaces | Parking spaces |
| floor | Floor number |
| furnishing_status | Furnishing condition |
| price_inr | House price in INR |

---

# Technologies Used

- Python
- pandas
- NumPy
- scikit-learn
- Matplotlib

---

# Machine Learning Algorithm

## Linear Regression

The project uses Linear Regression to predict house prices based on multiple housing features.

---

# Data Preprocessing

The project includes:

- Removing missing values
- Handling messy categorical data
- Feature scaling
- One-hot encoding
- Converting text labels into machine-readable format

---

# Model Evaluation

The model is evaluated using:

- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)

---

# Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_LINK
cd india-house-price-prediction
```

Install dependencies:

```bash
pip install pandas numpy matplotlib scikit-learn
```

---

# Run The Project

```bash
python house_price.py
```

---

# Future Improvements

- Add larger real-world datasets
- Improve data cleaning
- Add visualization dashboards
- Train advanced ML models
- Add Deep Learning support
- Build a web application

---

# Author

Dev / Samrat  
Beginner Machine Learning Developer

---

# License

This project is open-source and available for educational purposes.
