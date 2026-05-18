# Task 3: Car Price Prediction with Machine Learning

## Project Overview

This project is completed as part of the Oasis Infobyte Data Science Internship.

The main objective of this project is to predict the selling price of cars using machine learning. The model uses car-related features such as year, present price, kilometers driven, fuel type, selling type, transmission, and owner details to estimate the selling price.

## Dataset

The dataset contains information about used cars, including:

- Car_Name
- Year
- Selling_Price
- Present_Price
- Driven_kms
- Fuel_Type
- Selling_type
- Transmission
- Owner

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook
- VS Code

## Project Workflow

1. Imported required Python libraries
2. Loaded the car dataset
3. Checked dataset information
4. Checked missing values
5. Performed exploratory data analysis
6. Created a new feature called Car_Age
7. Removed unnecessary columns
8. Encoded categorical columns
9. Split data into training and testing sets
10. Trained Linear Regression model
11. Trained Random Forest Regressor model
12. Evaluated models using regression metrics
13. Compared actual and predicted car prices
14. Visualized feature importance

## Models Used

- Linear Regression
- Random Forest Regressor

## Best Model Performance

The Random Forest Regressor gave the best performance.

### Random Forest Regressor Results

- Mean Absolute Error: 0.6368
- Mean Squared Error: 0.9339
- Root Mean Squared Error: 0.9664
- R2 Score: 0.9594

## Key Insights

- Present price is an important factor in predicting selling price.
- Car age affects the selling price of a car.
- Fuel type, transmission, and selling type also influence car price.
- Random Forest Regressor performed better than Linear Regression.
- The predicted prices were close to the actual selling prices.

## Conclusion

In this project, a car price prediction model was built using machine learning. The dataset was cleaned and prepared for model training. Categorical features were converted into numerical values, and a new feature called Car_Age was created.

The Random Forest Regressor model achieved a high R2 Score of 0.9594, showing strong prediction performance. This project improved understanding of regression, feature engineering, model evaluation, and machine learning workflow.

## Author

Nikhil