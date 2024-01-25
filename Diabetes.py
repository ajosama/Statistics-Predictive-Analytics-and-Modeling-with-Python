# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 12:32:23 2024

@author: AbuOsama
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import dataset
data = pd.read_csv("diabetes.csv")

x = data.iloc[:,:-1].values
y = data.iloc[:,8].values

# Encode categorical variables
from sklearn.preprocessing import LabelEncoder
label_encoder_y = LabelEncoder()
y = label_encoder_y.fit_transform(y) 

# Splitting x and y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x, y,test_size=0.25,random_state=0)

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()

x_train = sc_x.fit_transform(x_train)
x_test = sc_x.fit_transform(x_test)

# Fit logistic regression to training set using sklearn
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(x_train,y_train)

# Predictions
pred1 = classifier.predict(x_test)

# confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, pred1)
cm

# TP = 117
# TN = 36
# FP = 26
# FN = 12

# Accuracy = (117+36)/192 = 0.7969 = 79.69%

#############################################################################
# for summary of model statsmodel is prefered
# fitting model with statsmodel package
# split

from sklearn.model_selection import train_test_split
x_train_s,x_test_s,y_train_s,y_test_s = train_test_split(x,y,test_size=0.25,random_state=0)

# feature scaling
from sklearn.preprocessing import StandardScaler
sc_x_s = StandardScaler()
x_train_s = sc_x_s.fit_transform(x_train_s)
x_test_s = sc_x_s.fit_transform(x_test_s)

# Adding one extra column
import statsmodels.api as sm
x_train_s = sm.add_constant(x_train_s)
x_test_s = sm.add_constant(x_test_s)

# logistic regression model
import statsmodels.formula.api as smf
classifier2 = sm.Logit(endog =y_train_s,exog =x_train_s).fit()
classifier2.summary()

# Predictions

pred2 = classifier2.predict(x_test_s)

# pred1 directly gave predictions as 0 or 1. But statsmodel gives probabilities
# Hence setting threshold

pred2 = (pred2 > 0.5).astype(int)

# confusion matrix

from sklearn.metrics import confusion_matrix
cm2 = confusion_matrix(y_test_s,pred2)
cm2

######################################################################
# Backward Elimination for making optimal model

# AIC The Akaike information criterion (AIC) is a mathematical method
# for evaluating how well a model fits the data it was generated from.
# In statistics, AIC is used to compare different possible models and
# determine which one is the best fit for the data.

# formula to calculate AIC score
# AIC = -(2*Log-Likelihood) + (2*Number of variables in the model)
# Log-Likelihood - we get this in summary classifier2 = sm.Logit(endog =y_train_s,exog =x_train_s).fit()
# classifier2.summary()

#aic1 = -(2*(-277.93)) + (2 * 8)
# aic1 = 571.86

####################################################################

x_opt = x_train_s[:,[0,1,2,3,4,5,6,7,8]]

classifier_b = sm.Logit(endog =y_train_s,exog =x_opt).fit()
classifier_b.summary()

#aic1 = -(2*(-277.93)) + (2 * 8)
# aic1 = 571.86

# Misclassifications = Total row - (117+36) = 192-153 = 39
# from summary, x4 i.e SkinThickness variable has highest p value
# Hence, it is not significant and will remove that variable

# Step2

x_opt = x_train_s[:,[0,1,2,3,5,6,7,8]]

classifier_b = sm.Logit(endog =y_train_s,exog =x_opt).fit()
classifier_b.summary()

#aic2 = -(2*(-278.20)) + (2 * 7)
#aic2 = 570.4

pred3 = classifier_b.predict(x_test_s[:,[0,1,2,3,5,6,7,8]])

# confusion matrix and setting threshold
pred3 = (pred3>0.5).astype(int)

from sklearn.metrics import confusion_matrix
cm3 = confusion_matrix(y_test_s,pred3)
cm3

# Misclassifications = Total row - (117+38) = 192-155 = 37
# from summary, x4 i.e Insulin variable has highest p value
# Hence, it is not significant and will remove that variable

# Step 3

x_opt = x_train_s[:,[0,1,2,3,6,7,8]]

classifier_b = sm.Logit(endog =y_train_s,exog =x_opt).fit()
classifier_b.summary()

#aic3 = -(2*(-278.50)) + (2 * 6)
#aic3 = 569.0

pred4 = classifier_b.predict(x_test_s[:,[0,1,2,3,6,7,8]])

# confusion matrix and setting threshold
pred4 = (pred4>0.5).astype(int)

from sklearn.metrics import confusion_matrix
cm4 = confusion_matrix(y_test_s,pred4)
cm4

# Misclassifications = 25 + 13 =38

# Model misclassification is more is step3 as compared to step2
# Hence model in step2 as final model
# Don't remove Insulin

# Final Model

x_opt = x_train_s[:,[0,1,2,3,5,6,7,8]]

classifier_b = sm.Logit(endog =y_train_s,exog =x_opt).fit()
classifier_b.summary()

#aic2 = -(2*(-278.20)) + (2 * 7)
#aic2 = 570.4

pred3 = classifier_b.predict(x_test_s[:,[0,1,2,3,5,6,7,8]])

# confusion matrix and setting threshold
pred3 = (pred3>0.5).astype(int)

from sklearn.metrics import confusion_matrix
cm3 = confusion_matrix(y_test_s,pred3)
cm3

pd.crosstab(y_test_s,pred3,rownames =['True'],colnames =['Predicted'],margins =True)

# [TP  FN
#  FP  TN]

##############################################################################

# FP-FalsePositive is 24 means it is high and it is very dangerous for
# our model that's why we trying to optimize the FP. for this we will
# use ROC curves with the help of ROC curves we can make sure for threshold 
# value pred3 = (pred3>0.5).astype(int) 0.5 is a threshold value
# An ROC curve (receiver operating characteristic curve) is a graph
# showing the performance of a classification model at all classification
# thresholds. This curve plots two parameters: True Positive Rate. False Positive Rate.

# ROC curve to change threshold 

from sklearn.metrics import confusion_matrix,roc_auc_score,roc_curve

pred_r = classifier_b.predict(x_train_s[:,[0,1,2,3,5,6,7,8]])

# Computing true positive rates and false positive rates

fpr,tpr,threshold = roc_curve(y_true=y_train_s,y_score=pred_r,
                              drop_intermediate=False)

# plotting roc curve
import matplotlib.collections
plt.figure()

#Add roc
plt.plot(fpr,tpr,lw=2,color='red')

# Random TPR and FPR Limits
plt.plot([0,1],[0,1],lw=2,color='blue')

# Title and label
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC CURVE')
plt.show()

# Area under the curve
roc_auc_score(y_true=y_train_s,y_score=pred_r)
# 0.83

# changing threshold
# TPR must be high
# FPR must be less

# Reduce threshold

pred5 = classifier_b.predict(x_test_s[:,[0,1,2,3,5,6,7,8]])

# confusion matrix and setting threshold
pred5 = (pred5>0.35).astype(int)

from sklearn.metrics import confusion_matrix
cm5 = confusion_matrix(y_test_s,pred5)
cm5

