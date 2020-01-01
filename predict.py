#using input_data.py, with existing data sets and training to predict the 47th and the 55th data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (15.0, 12.0)

# Preprocessing Input data
data = pd.read_csv('D://data.csv')
X = data.iloc[:, 0]
Y = data.iloc[:, 1]
plt.scatter(X, Y)
plt.show()

# Building the model
m = 0
c = 0

L = 0.0001  # The learning Rate
epochs = 1000  # The number of iterations to perform gradient descent

n = float(len(X))  # Number of elements in X

# Performing Gradient Descent
for i in range(epochs):
    Y_pred = m * X + c  # The current predicted value of Y
    D_m = (-2 / n) * sum(X * (Y - Y_pred))  # Derivative wrt m
    D_c = (-2 / n) * sum(Y - Y_pred)  # Derivative wrt c
    m = m - L * D_m  # Update m
    c = c - L * D_c  # Update c

print(m, c)
# Making predictions
Y_pred = m*X + c

plt.scatter(X, Y)
plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color='red')  # regression line
plt.show()

print('--------------')

print('the 47th training:%s'%(47*m+c))

print('-------------')

print('the 55th training:%s'%(55*m+c))




