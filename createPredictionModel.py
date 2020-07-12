#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 07:43:46 2019

@author: jaisi8631
"""

# -------------------------
# IMPORT NECESSARY MODULES
# -------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn import decomposition
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier


# -------------------------
# DATA PREPROCESSING
# -------------------------
# load required files and allocate data 
data = pd.read_csv('data/all_dataset.csv', header = None)
X = data.iloc[:, 4:-1].values
y = data.iloc[:, -1].values

# apply Principal Component Analysis
pca = decomposition.PCA(n_components = 10)
pca.fit(X)
X = pca.transform(X)

# create a train-test split with ratio 15:85
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15)

# scale the training and testing data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# save the scaler for future use
scalerfile = 'scaler.save'
pickle.dump(sc, open(scalerfile, 'wb'))


# -------------------------
# BASELINE MODELS CREATION
# -------------------------
from sklearn.metrics import accuracy_score

# 1 - logistic regression
print("Training Logistic Regression model...")
from sklearn.linear_model import LogisticRegression
classifier_lr = LogisticRegression(solver = 'lbfgs')
classifier_lr.fit(X_train, y_train)

y_pred = classifier_lr.predict(X_test)
score_lr = accuracy_score(y_test, y_pred)

# 2 - random forest
print("Training Random Forest model...")
from sklearn.ensemble import RandomForestClassifier
classifier_rf = RandomForestClassifier(n_estimators = 75, criterion = 'entropy')
classifier_rf.fit(X_train, y_train)

y_pred = classifier_rf.predict(X_test)
score_rf = accuracy_score(y_test, y_pred)

# 3 - naive bayes
print("Training Naive Bayes model...")
from sklearn.naive_bayes import GaussianNB
classifier_nb = GaussianNB()
classifier_nb.fit(X_train, y_train)

y_pred = classifier_nb.predict(X_test)
score_nb = accuracy_score(y_test, y_pred)


# -------------------------
# NEURAL NETWORK CREATION
# -------------------------
print("Training Neural Network model...")
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras import optimizers

# create function to build classifier
def build_classifier(optimizer = 'adam'):
    classifier_nn = Sequential()
    classifier_nn.add(Dense(50, input_dim = X_train.shape[1], 
                    kernel_initializer = 'random_uniform', 
                    activation = 'sigmoid'))
    classifier_nn.add(Dropout(0.2))
    classifier_nn.add(Dense(100, activation = 'relu'))
    classifier_nn.add(Dropout(0.5))
    classifier_nn.add(Dense(100, activation = 'relu'))
    classifier_nn.add(Dropout(0.5))
    classifier_nn.add(Dense(25, activation = 'relu'))
    classifier_nn.add(Dropout(0.2))
    classifier_nn.add(Dense(1, kernel_initializer = 'normal', activation = 'sigmoid'))
    classifier_nn.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
    return classifier_nn

# convert to sklearn-readable model
classifier_nn = KerasClassifier(build_fn = build_classifier)

# declare search values
batch_size = [5, 20, 50]
epochs = [3, 5, 7]
optimizers = ['adam', 'rmsprop']

# create grid for gridsearch and perform gridsearch
grid = {'epochs': epochs,
        'batch_size': batch_size,
        'optimizer': optimizers}
validator = GridSearchCV(classifier_nn,
                         param_grid = grid,
                         scoring = 'accuracy',
                         n_jobs = 1)
validator.fit(X_train, y_train)

# get the best model from the gridsearch
print('The parameters of the best Neural Network model are: ')
print(validator.best_params_)
classifier_nn = validator.best_estimator_.model

# make predictions using the model
y_pred = classifier_nn.predict(X_test)
y_pred = np.where(y_pred > 0.5, 1, 0)
score_nn = accuracy_score(y_test, y_pred)


# -------------------------
# MODEL EVALUATION
# -------------------------
# print the accuracy score of each  network
print()
print()
print("Logisitc Regression Accuracy: " + str(round(score_lr, 3)))
print("Random Forest Accuracy: " + str(round(score_rf, 3)))
print("Naive Bayes Accuracy: " + str(round(score_nb, 3)))
print("Neural Network Accuracy: " + str(round(score_nn, 3)))


# -------------------------
# MODEL STORAGE
# -------------------------
# store the neural network model with the highest accuracy score
with open('predictor.pkl', 'wb') as fid:
    pickle.dump(classifier_nn, fid)

# output feedback
print()
print()
print("Predictive Models creation complete.")
print()
print()