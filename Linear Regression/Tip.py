# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 00:53:56 2023

@author: AbuOsama
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as sf

# import the dataset
data = pd.read_csv("Tip.csv")

# what will be the tip in the future so our dependent variable is tip
x = data.iloc[:,:-1].values
y = data.iloc[:,1].values

# fitting Linear Regression
regressor = LinearRegression()
regressor.fit(x, y)

regressor.coef_     # B1
regressor.intercept_ # B0

# Plot
x_mean = [np.mean(x) for i in x]
y_mean = [np.mean(y) for i in y]

plt.scatter(x,y)
plt.plot(x,regressor.predict(x),color = "red")
plt.plot(x,y_mean,linestyle="--")
plt.plot(x_mean,y,linestyle="--")
plt.title("Tip vs Bill")
plt.xlabel("Bill in $")
plt.ylabel("Tip in $")
plt.show()

# Summarized model
lm = sf.ols(formula ="Tip ~ Bill",data=data).fit()
lm.params    # same B0 and B1 as above method

print(lm.summary())