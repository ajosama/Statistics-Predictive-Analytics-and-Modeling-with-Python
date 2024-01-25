# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 13:02:58 2024

@author: AbuOsama
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import dataset
datac = pd.read_csv("Credit Risk.csv")

# getting dependent and independent variables
x = datac.iloc[:,:-1]
y = datac.iloc[:,12]

# Encode target variable
from sklearn.preprocessing import LabelEncoder
label_encoder_y = LabelEncoder()
y = label_encoder_y.fit_transform(y)

# pre processing or data cleaning

# check for missing values
# check for outliers

# Calculate total number of missing values
miss = x.apply(lambda x: sum(x.isnull()),axis=0)

# Loan_ID - no preprocessing required

############################################################################

# Gender - total missing value is 13 but we also check it by the column name as
x.Gender.isnull().sum()

# Now check the counts of categories if the variable is categorical
x.Gender.value_counts()

# Now impute missing values with Males or level with highest number of occurance
x['Gender'].fillna('Male',inplace =True)

############################################################################

# Married
x.Married.isnull().sum()

# Now check the counts of categories if the variable is categorical
x.Married.value_counts()

# Now impute missing values with Yes or level with highest number of occurance
x['Married'].fillna('Yes',inplace =True)

############################################################################

# Dependents
x.Dependents.isnull().sum()

# Now check the counts of categories if the variable is categorical
x.Dependents.value_counts()

# Now impute missing values with 0 or level with highest number of occurance
x['Dependents'].fillna('0',inplace =True)

############################################################################

# Education
x.Education.isnull().sum()

# No miising values hence no preprocessing required

############################################################################

# Self Employed
x.Self_Employed.isnull().sum()

# Now check the counts of categories if the variable is categorical
x.Self_Employed.value_counts()

# Now impute missing values with No or level with highest number of occurance
x['Self_Employed'].fillna('No',inplace =True)

####################################################################################

# ApplicantIncome
x.ApplicantIncome.isnull().sum()

# No miising values but it is a numeric data column that's why we will check
# for outliers using boxplot

x.boxplot('ApplicantIncome')

# Now there is an outliers present in the data
# Outliers treatment

q75,q25 = np.percentile(x.ApplicantIncome,[75,25])

# Interquartile Range
iqr = q75 - q25

# Threshold
a = q75 + (1.5 * iqr)

# any value above 'a' will be considered as an outliers
# Now we will replace outliers by mean 
# higher will be the applicant income, higher will be probability of getting loan

x.ApplicantIncome.loc[x['ApplicantIncome'] >= a] = np.mean(x['ApplicantIncome'])
x.boxplot('ApplicantIncome')

#####################################################################################

# CoapplicantIncome
x.CoapplicantIncome.isnull().sum()

# No miising values but it is a numeric data column that's why we will check
# for outliers using boxplot

x.boxplot('CoapplicantIncome')

# Now there is an outliers present in the data
# Outliers treatment

q75,q25 = np.percentile(x.CoapplicantIncome,[75,25])

# Interquartile Range
iqr = q75 - q25

# Threshold
a = q75 + (1.5 * iqr)

# any value above 'a' will be considered as an outliers
# Now we will replace outliers by mean 
# higher will be the applicant income, higher will be probability of getting loan

x.CoapplicantIncome.loc[x['CoapplicantIncome'] >= a] = np.mean(x['CoapplicantIncome'])
x.boxplot('CoapplicantIncome')

######################################################################################

# LoanAmount
x.LoanAmount.isnull().sum()

# Describe

x['LoanAmount'].describe()

# 22 miising values 
# Dealing with missing values 
# Lower the loan amount, higher the probability of getting loan
# Replace missing values with median because median is less than mean in this case 
# and we want lower amount that's why we will use median not mean

x['LoanAmount'].fillna(x['LoanAmount'].median(),inplace = True)

# for outliers using boxplot

x.boxplot('LoanAmount')

# Now there is an outliers present in the data
# Outliers treatment

q75,q25 = np.percentile(x.LoanAmount,[75,25])

# Interquartile Range
iqr = q75 - q25

# Threshold
a = q75 + (1.5 * iqr)

# any value above 'a' will be considered as an outliers
# Now we will replace outliers by median 
# lower will be the LoanAmount, higher will be probability of getting loan

x.LoanAmount.loc[x['LoanAmount'] >= a] = np.median(x['LoanAmount'])
x.boxplot('LoanAmount')

################################################################################

# Loan Amount Term
x.Loan_Amount_Term.isnull().sum()

# Describe

x['Loan_Amount_Term'].describe()

# Here quartiles are same
# now we will check the values means categories
x.Loan_Amount_Term.value_counts()

# 14 miising values 
# impute missing values by 360 beacuse it comes highest number of times

x['Loan_Amount_Term'].fillna(360,inplace = True)

################################################################################

# Credit History
x.Credit_History.isnull().sum()

# now we will check the values means categories
x.Credit_History.value_counts()

# 50 miising values 
# impute missing values by 1 beacuse it comes highest number of times

x['Credit_History'].fillna(1,inplace = True)

###############################################################################

# Property area
x.Property_Area.isnull().sum()

# no missing values

###############################################################################

# Binning of loan amount term
# because loan_amount_term categories are more that's why we will make classes
# using bins

x['Loan_Amount_Term_Bins'] = pd.cut(x['Loan_Amount_Term'],
                                    bins = [0,120,240,360,480],
                                    labels=['0-120','120-240','240-360','360-480'])

x.Loan_Amount_Term_Bins.value_counts()

# Remove irrelevant variables

x1 = x.drop(['Loan_ID','Loan_Amount_Term'],axis = 1)

# Making dummies

x2 = pd.get_dummies(x1,drop_first=True).astype(int)

# Splitting the dataset into training and splitting data
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x2,y,test_size=0.2,random_state =0)

####################################################################################

# Model

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state=0)
classifier.fit(x_train,y_train)

# Lets see how good it predicts training set

y_pred_r = classifier.predict(x_train)

# confusion matrix

from sklearn.metrics import confusion_matrix
cm_tr = confusion_matrix(y_train, y_pred_r)
cm_tr

# (68+327)/491 = 0.8044...means 80% correct predictions

# Area under ROC curve
from sklearn.metrics import roc_auc_score
roc_auc_score(y_true=y_train,y_score=y_pred_r)

# AUC = 0.7063....means 71% area comes under the curve means model is good

###############################################################################

# Prediction in test set

y_pred = classifier.predict(x_test)
cm = confusion_matrix(y_test,y_pred)

# (40+88)/123 = 0.8292 means 83% correct prediction on test set




































