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


# -------------------------
# DATA PREPROCESSING
# -------------------------
# load required files and allocate data 
data = pd.read_csv('data/all_dataset.csv', header = None)
X = data.iloc[:, 4:-1].values
y = data.iloc[:, -1].values

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
# MODEL CREATION
# -------------------------
# 1 - logistic regression
print("Training Logistic Regression model...")

from sklearn.linear_model import LogisticRegression
classifier_lr = LogisticRegression(solver = 'lbfgs')
classifier_lr.fit(X_train, y_train)

y_pred = classifier_lr.predict(X_test)
from sklearn.metrics import accuracy_score
score_lr = accuracy_score(y_test, y_pred)

# 2 - random forest
print("Training Random Forest model...")
from sklearn.ensemble import RandomForestClassifier
classifier_rf = RandomForestClassifier(n_estimators = 75, criterion = 'entropy')
classifier_rf.fit(X_train, y_train)

y_pred = classifier_rf.predict(X_test)
from sklearn.metrics import accuracy_score
score_rf = accuracy_score(y_test, y_pred)

# 3 - naive bayes
print("Training Naive Bayes model...")
from sklearn.naive_bayes import GaussianNB
classifier_nb = GaussianNB()
classifier_nb.fit(X_train, y_train)

y_pred = classifier_nb.predict(X_test)
from sklearn.metrics import accuracy_score
score_nb = accuracy_score(y_test, y_pred)

# 4 - neural network
print("Training Neural Network model...")
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras import optimizers

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
classifier_nn.add(Dense(1, kernel_initializer='normal', activation = 'sigmoid'))

adam = optimizers.Adam(lr = 0.005)
classifier_nn.compile(loss = 'binary_crossentropy', optimizer = adam)
classifier_nn.fit(X_train, y_train, batch_size = 20, epochs = 5)

y_pred = classifier_nn.predict(X_test)
y_pred = np.where(y_pred > 0.5, 1, 0)
from sklearn.metrics import accuracy_score
score_nn = accuracy_score(y_test, y_pred)


# -------------------------
# MODEL EVALUATION
# -------------------------
# print the accuracy score of each 
print()
print()
print("Logisitc Regression Accuracy: " + str(round(score_lr, 3)))
print("Random Forest Accuracy: " + str(round(score_rf, 3)))
print("Naive Bayes Accuracy: " + str(round(score_nb, 3)))
print("Neural Network Accuracy: " + str(round(score_nn, 3)))


# -------------------------
# MODEL STORAGE
# -------------------------
# identify the model with the highest accuracy score
classifiers = [classifier_lr, classifier_rf, classifier_nb, classifier_nn]
scores = [score_lr, score_rf, score_nb, score_nn]
x = scores.index(max(scores))

# store the model with the highest accuracy score to disk
with open('predictor.pkl', 'wb') as fid:
    pickle.dump(classifiers[x], fid)

# output feedback
print()
print()
print("Predictive Models creation complete.")
print()
print()