# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 16:25:37 2021

@author: aicha
"""

import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
boston_dataset = load_boston()
import pandas as pd
import numpy as np
boston = pd.DataFrame(boston_dataset.data, 
columns=boston_dataset.feature_names)
boston['MEDV'] = boston_dataset.target
#information
boston.shape
boston.columns
boston[['CHAS','RM','AGE','RAD','MEDV']].head(n=3)
boston.iloc[1]
boston.describe().round(2)
boston['AGE'].describe()
#visualization
boston.hist(column='CHAS')
plt.show()
boston.hist(column='RM',bins=20)
plt.show()
#Correlation
corr_matrix=boston.corr().round(2)
boston.plot(kind='scatter',x='RM',y='MEDV',figsize=(8,6))
boston.plot(kind='scatter',x = 'LSTAT',y = 'MEDV',figsize=(8,6))

X=boston[['RM']]
print(X.shape)
Y=boston['MEDV']
print(Y.shape)
#Instantiating the Model
from sklearn.linear_model import LinearRegression
model=LinearRegression()
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, 
  test_size = 0.3, 
  random_state=1)
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)
model.fit(X_train, Y_train)
#B dans Y=B+mX
model.intercept_.round(2)
#m pente
model.coef_.round(2)
new_RM=np.array([6.5]).reshape(-1,1)
model.predict(new_RM)
y_test_predicted=model.predict(X_test)
y_test_predicted.shape
type(y_test_predicted)
#Residual
plt.scatter(X_test, Y_test,label='testing data');
plt.plot(X_test, y_test_predicted,label='prediction', linewidth=3)
plt.xlabel('RM'); 
plt.ylabel('MEDV')
plt.legend(loc='upper left')
plt.show()

residuals = Y_test-y_test_predicted
# plot the residuals
plt.scatter(X_test, residuals)
# plot a horizontal line at y = 0
plt.hlines(y = 0,xmin = X_test.min(),xmax=X_test.max(),
linestyle='--')
# set xlim
plt.xlim((4, 9))
plt.xlabel('RM'); plt.ylabel('residuals')
plt.show()
residuals[:5]
residuals.mean()
(residuals**2).mean()
from sklearn.metrics import mean_squared_error
mean_squared_error(Y_test,
y_test_predicted)
model.score(X_test,Y_test)
((Y_test-Y_test.mean())**2).sum()
(residuals**2).sum()
 ## data preparation
X2=boston[['RM', 'LSTAT']]
Y = boston['MEDV']
## train test split
## same random state to ensure the same splits
X2_train, X2_test, Y_train, Y_test =train_test_split(X2, Y,
test_size = 0.3,
random_state=1)
model2=LinearRegression()
model2.fit(X2_train, Y_train)

model2.intercept_
model2.coef_
y_test_predicted2=model2.predict(X2_test)
mean_squared_error(Y_test, y_test_predicted).round(2)
mean_squared_error(Y_test, y_test_predicted2).round(2)
