# new-car-evaluation

## Objective

Develop a machine learning model to accurately predict the suitability of a car based on its various attributes. This typically involves classifying cars into different categories such as "acceptable," "good," "very good," or "unacceptable" based on criteria like price, maintenance cost, number of doors, capacity, safety features, etc. This model aims to assist users in making informed decisions when evaluating different cars for potential purchase.

## Summary

The model performs well in predicting car evaluations based on the provided dataset, showing a good balance between precision and recall. It has high accuracy, precision, and ROC AUC scores, indicating its effectiveness in classifying instances correctly and distinguishing between different classes. It is useful for identifying cars that may be considered "acceptable" or "unacceptable" based on their attributes.

## Project Workflow 

* Data Preprocessing = Handled missing and null values ,Removed duplicate entries, Renamed columns for clarity, Converted columns to appropriate data types, Added new columns if necessary, Checked unique values.
* Data Wrangling = Converted categorical columns to numerical columns, Performed Exploratory Data Analysis (EDA) to examine relationships between variables and identify trends.
* Hypothesis Testing = Conducted hypothesis testing using the Shapiro-Wilk test to obtain p-values.
* Handling Imbalanced Data = Used Synthetic Minority Over-sampling Technique (SMOTE) to balance the dataset. SMOTE generates synthetic samples for the minority class by interpolating between existing minority class instances. This helps in balancing the class distribution by increasing the number of minority class samples.
* Model Building = Built a Logistic Regression model, Built a Decision Tree Classifier model.
* Hyperparameter Tuning = Performed hyperparameter optimization using Grid Search Cross-Validation (GridSearchCV) to find the best combination of hyperparameters from a predefined grid of values.
* Model Evaluation = Evaluated models based on accuracy, precision, recall, F1 score, and ROC AUC scores.

## Web Application Development

* Model Deployment = Created a pickle file of the best-performing model , Deployed the model using Flask.
* Application Structure = Developed the web application using PyCharm, Created necessary files: model.py, index.html, and app.py.
* User Interface = Designed a user-friendly interface with index.html to interact with the model.

## Conclusion

The decision tree classifier model demonstrates outstanding performance with perfect scores across all metrics, suggesting optimal and error-free predictions for the given binary classification task. The model is highly effective for evaluating car suitability based on the attributes provided. The developed web application facilitates easy interaction with the model, making it accessible for users to evaluate car suitability seamlessly.

## Web Development Image

![image](https://github.com/user-attachments/assets/b0ab49d8-3a47-4806-b224-201bbafa0c5f)

