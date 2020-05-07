# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 15:49:11 2018

@author: Administrator
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

df = pd.read_csv('Regression_Trainingset.csv')

X = df.iloc[:, :-1].values
y = df.iloc[:, 1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y , test_size = 1.0/3,random_state = 0)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_test)

#visualisation

plt.title('Exp vs salary')
plt.scatter(X,y,s = 200, c = 'red',marker = '*')
plt.plot(X_train,regressor.predict(X_train))
ax = plt.subplot()
ax.set_xticks(range(0,12))
ax.set_xticklabels(range(0,12))
plt.xlabel('Exp')
plt.ylabel('sal')
plt.show()
