#!/usr/bin/env python
# coding: utf-8

#importing neccesary libraries 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

url = ("C:/Supervised Learning/Datasets/iris.data")

# Assigning column names to the dataset
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

# Reading dataset to pandas dataframe
dataset = pd.read_csv(url, names=names)

#Displaying original data
show_data=dataset.head(n=50)
show_data

#Plotting scatter_matrix of dataframe
pd.plotting.scatter_matrix(dataset)

#Assigning X and y
X = dataset.iloc[:, :-1].values #X-Attributes
y = dataset.iloc[:, 4].values   #y-Labels

#Train-Test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20) #80%-TrainData, 20%-Testdata

#Scaling the features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


#Training & Prediction using KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5)
classifier = knn
classifier.fit(X_train, y_train)

#Estimating accuracy of classifier
knn.score(X_test, y_test)

y_pred = classifier.predict(X_test) #predicting
print (y_pred)

#Evaluation
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#Comparing Error Rate with the K Value
error = []

# Calculating error for K values between 1 and 40
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))
plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')    







