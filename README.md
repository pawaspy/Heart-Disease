# Predicting Heart Disease Using Machine Learning

This project aims to predict the presence of heart disease in individuals based on their medical attributes. We explore various machine learning models and techniques to achieve this goal. 

## Technology Stack

The project utilizes a range of technologies and libraries:

- **Python:** The primary programming language for data analysis and model development.
- **pandas:** Used for data manipulation and analysis.
- **numpy:** Provides support for numerical operations on data.
- **matplotlib and seaborn:** Used for data visualization.
- **scikit-learn:** A popular machine learning library for building and evaluating models.
- **Jupyter Notebook:** Used for interactive development and documentation.

## Data

We have utilized the "heart-disease.csv" dataset, which contains various medical attributes such as age, sex, cholesterol levels, and more. These attributes are used to predict the presence or absence of heart disease (target variable).

## Data Exploration

We began by exploring the dataset to gain insights into the data:

- Checked the distribution of target classes (presence/absence of heart disease).
- Examined basic statistics, such as mean, standard deviation, minimum, and maximum values for each attribute.
- Visualized the relationship between certain attributes (e.g., age vs. max heart rate) to understand potential correlations.
- Calculated the correlation matrix to identify relationships between attributes.

## Models Implemented

We implemented several machine learning models to predict heart disease:

1. **Logistic Regression**
    - Logistic regression is a linear model used for binary classification.
    - Achieved an accuracy of approximately **88.52%**.
2. **K-Nearest Neighbors (KNN)**
    - KNN is a non-linear model that classifies data points based on the majority class of their k-nearest neighbors.
    - Achieved an accuracy of approximately 68.85% with default hyperparameters.
    - Further hyperparameter tuning improved the accuracy to approximately **75.41%**.
3. **Random Forest Classifier**
    - Random forests are an ensemble learning method that combines multiple decision trees.
    - Achieved an accuracy of approximately 83.61% with default hyperparameters.
    - After hyperparameter tuning, the accuracy reached approximately **86.89%**.

## Best-Performing Model - Logistic Regression

The logistic regression model achieved the highest accuracy among the models we tested, with an accuracy of approximately 88.52%. Here are some reasons why we selected this model as the best-performing one for this task:

- **Interpretability:** Logistic regression is a straightforward model that provides interpretable results. It allows us to understand the impact of each feature on the prediction.
- **Efficiency:** Logistic regression is computationally efficient and performs well even with smaller datasets.
- **Good baseline:** Logistic regression provides a solid baseline for binary classification problems, and its performance can be further improved with feature engineering and hyperparameter tuning.

## Hyperparameter Tuning

We performed hyperparameter tuning to optimize the models. RandomizedSearchCV and GridSearchCV were used to find the best hyperparameters for the Logistic Regression and Random Forest models, respectively.

## Model Evaluation

We evaluated the best-performing model (Logistic Regression) using various metrics, including precision, recall, F1-score, and the confusion matrix. The model showed promising results, with an accuracy of approximately 88.52% and good precision and recall values.

## Cross-Validation

Cross-validation was performed to ensure the model's robustness and reliability. The Logistic Regression model exhibited consistent performance across different cross-validation folds.

## Feature Importance

Feature importance was examined to determine which attributes had the most significant impact on the model's predictions. This information can be valuable for understanding the factors contributing to heart disease.

In summary, this project demonstrates the process of building a machine learning model to predict heart disease using Python and scikit-learn. The best-performing model, Logistic Regression, achieved an accuracy of approximately 88.52%, making it a reliable tool for heart disease prediction. Further improvements and refinements can be made by exploring additional features and advanced techniques.
