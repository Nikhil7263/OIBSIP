# Task 4: Email Spam Detection with Machine Learning

## Project Overview

This project is completed as part of the Oasis Infobyte Data Science Internship.

The main objective of this project is to build a machine learning model that can classify messages as spam or ham. The project uses text preprocessing, TF-IDF vectorization, and a classification algorithm to detect spam messages.

## Dataset

The dataset contains SMS/email messages labeled as:

- Ham: Normal message
- Spam: Unwanted or promotional message

The important columns used in this project are:

- v1: Message label
- v2: Message text

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
2. Loaded the spam detection dataset
3. Removed unwanted columns
4. Renamed columns for better understanding
5. Checked missing values
6. Visualized spam and ham message count
7. Converted labels into numerical values
8. Split data into training and testing sets
9. Converted text messages into numerical features using TF-IDF Vectorizer
10. Trained a Multinomial Naive Bayes model
11. Predicted results on test data
12. Evaluated model performance using accuracy score, confusion matrix, and classification report
13. Tested the model with custom messages

## Model Used

- Multinomial Naive Bayes

## Model Accuracy

The model achieved an accuracy of approximately:

```text
96.68%

## Author

Nikhil