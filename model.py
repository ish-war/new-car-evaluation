import numpy as np
import pandas as pd

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('car_eval.csv')

df.drop('Unnamed: 0', axis= 1, inplace= True)
print(df.head())

x= df.drop('target',axis=1)
y= df['target']

smote = SMOTE(random_state = 42)
x_resampled, y_resampled = smote.fit_resample(x, y)

from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.3, random_state= 42)
param_grid = {
    'max_depth': [3, 5, 20],
    'min_samples_split': [2, 5, 20],
    'min_samples_leaf': [1, 3, 5]
}

dt_classifier = DecisionTreeClassifier()

grid_search = GridSearchCV(dt_classifier, param_grid, cv=5, scoring='accuracy')
grid_search.fit(x_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_

# Train a new model with the best parameters
best_dt_classifier = DecisionTreeClassifier(**best_params)
best_dt_classifier.fit(x_resampled, y_resampled)

y_pred_train_best = best_dt_classifier.predict(x_train)
y_pred_test_best = best_dt_classifier.predict(x_test)

train_accuracy = accuracy_score(y_train, y_pred_train_best)
test_accuracy = accuracy_score(y_test, y_pred_test_best)

print("train accuracy :", train_accuracy)
print("test accuracy :", test_accuracy)

import pickle

pickle.dump(best_dt_classifier, open('model.pkl', 'wb'))      # wb - written binary
model = pickle.load(open('model.pkl', 'rb'))


