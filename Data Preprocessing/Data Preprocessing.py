# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd
import os
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

os.getcwd()

# import dataset
data1 = pd.read_csv("C:\\Users\\Abu Osama\\Desktop\\Predictive Modeling with Python\\Data Preprocessing\\Data1.csv")

# another way to import dataset
os.chdir("C:\\Users\\Abu Osama\\Desktop\\Predictive Modeling with Python\\Data Preprocessing")

data2 = pd.read_csv("Data1.csv")

# Export dataset

pd.DataFrame.to_csv(data2,"data4.csv")


# Data Preprocessing
data = pd.read_csv("Data1.csv")

# Our main task is to predict wheather the user has purchased the product or not
# so the our dependend variable will be purchased

# df[Rows,Columns]      .values is numpy method
x = data.iloc[:,:-1].values
y = data.iloc[:,3].values

# Handling missing values in data

imput1 = SimpleImputer(missing_values=np.nan, strategy='mean')
imput2 = imput1.fit(x[:,1:3])
x[:,1:3] = imput2.fit_transform(x[:,1:3])

# Encode categorical variables using this we convert yes and no into 0s and 1s and also countries
label_encoder_y = LabelEncoder()
y = label_encoder_y.fit_transform(y)

label_encoder_x = LabelEncoder()
x[:,0] = label_encoder_x.fit_transform(x[:,0])

# Create Dummies
categorical_features = [0]
# Creating a ColumnTransformer with OneHotEncoder
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_features)
    ],
    remainder='passthrough'
)
# Fit and transform the data
x = preprocessor.fit_transform(x)

# Splitting dataset into training set and test set
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

# Feature Scaling


