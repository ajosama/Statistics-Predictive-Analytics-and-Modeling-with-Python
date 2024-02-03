# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 21:13:51 2023

@author: AbuOsama
"""

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import dataset
data = pd.read_csv("Salary_data.csv")

# Dependent and Independent variables
x = data.iloc[:,:-1].values
y = data.iloc[:,1].values

# Splitting dataset into training and test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=1/3,random_state=0)

# Fit simple linear regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# To see the coefficient and intercept
regressor.coef_
regressor.intercept_

# Predicting test_set results
pred = regressor.predict(x_test)

# RMSE
from sklearn.metrics import mean_squared_error
from math import sqrt
rmse = sqrt(mean_squared_error(y_test, pred))
rmse

# Plotting training set results
plt.scatter(x_train,y_train,color="blue")
plt.plot(x_train,regressor.predict(x_train),color="green")
plt.title("Training set (sal vs Exp)")
plt.xlabel("Experience in years")
plt.ylabel("Salary")
plt.show()

# Plotting test set results
plt.scatter(x_test,y_test,color="blue")
plt.plot(x_train,regressor.predict(x_train),color="green")
plt.title("Test set (sal vs Exp)")
plt.xlabel("Experience in years")
plt.ylabel("Salary")
plt.show()

# Plotting together
plt.scatter(x_train,y_train,color="blue")
plt.scatter(x_test,y_test,color="red")
plt.plot(x_train,regressor.predict(x_train),color="green")
plt.title("sal vs Exp")
plt.xlabel("Experience in years")
plt.ylabel("Salary")
plt.show()

# Summarized model
import statsmodels.formula.api as sf
lm = sf.ols(formula ="Salary ~ YearsExperience",data=data).fit()
lm.params    # same B0 and B1 as above method

print(lm.summary())