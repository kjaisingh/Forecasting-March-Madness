#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 07:43:46 2019

@author: jaisi8631
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data = pd.read_csv('dataset.csv', header=None)
X = data.iloc[:, 4:-1].values
y = data.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# 1 - logistic regression
print("Training Logistic Regression model...")

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
from sklearn.metrics import accuracy_score
score_lr = accuracy_score(y_test, y_pred)


# 2 - random forest
print("Training Random Forest model...")
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 50, criterion = 'entropy')
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
from sklearn.metrics import accuracy_score
score_rf = accuracy_score(y_test, y_pred)


# 3 - naive bayes
print("Training Naive Bayes model...")
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
from sklearn.metrics import accuracy_score
score_nb = accuracy_score(y_test, y_pred)


# 4 - neural network
print("Training Neural Network model...")
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras import optimizers

classifier = Sequential()
classifier.add(Dense(50, input_dim = X_train.shape[1], 
                kernel_initializer = 'random_uniform', 
                activation = 'sigmoid'))
classifier.add(Dropout(0.2))
classifier.add(Dense(50, activation = 'relu'))
classifier.add(Dropout(0.4))
classifier.add(Dense(30, activation = 'relu'))
classifier.add(Dropout(0.4))
classifier.add(Dense(20, activation = 'relu'))
classifier.add(Dropout(0.4))
classifier.add(Dense(1, kernel_initializer='normal'))

adam = optimizers.Adam(lr = 0.01)
classifier.compile(loss = 'mean_squared_error', optimizer = adam)
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 10)

y_pred = classifier.predict(X_test)
y_pred = np.where(y_pred > 0.5, 1, 0)
from sklearn.metrics import accuracy_score
score_nn = accuracy_score(y_test, y_pred)


print("Logisitc Regression Accuracy: " + str(round(score_lr, 3)))
print("Random Forest Accuracy: " + str(round(score_rf, 3)))
print("Naive Bayes Accuracy: " + str(round(score_nb, 3)))
print("Neural Network Accuracy: " + str(round(score_nn, 3)))