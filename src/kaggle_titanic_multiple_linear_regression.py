#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 10:21:17 2018

@author: akhil-6580
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler


def process_X(dataframe):
    columns_in_X = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    val_set_x = dataframe[columns_in_X].iloc[:, :].values
    
    # setting ages to missing ages, fares to median
    imputer = Imputer(strategy='median')
    val_set_x[:, 2:3] = imputer.fit_transform(val_set_x[:, 2:3])
    val_set_x[:, 5:6] = imputer.transform(val_set_x[:, 5:6])
    
    # Encoding Embarkment to labels
    label_encoder_sex = LabelEncoder()
    val_set_x[:, 1] = label_encoder_sex.fit_transform(val_set_x[:, 1])
    
    # Embarked: removing na values, replacing ''
    label_encoder_embarked = LabelEncoder()
    for x in range(len(val_set_x[:,6])):
        if val_set_x[:,6][x] is np.nan:
            val_set_x[:,6][x] = ''

    val_set_x[:, 6] = label_encoder_embarked.fit_transform(val_set_x[:, 6])
    

    return val_set_x




train_df = pd.read_csv('titanic_train.csv')
test_df = pd.read_csv('titanic_test.csv')

train_X = process_X(train_df)
test_X = process_X(test_df)


# extract dependent values 
train_y = train_df.iloc[:,1:2].values

# Standardize values
sc_X = StandardScaler()
train_X = sc_X.fit_transform(train_X)
test_X = sc_X.transform(test_X)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(train_X, train_y)

y_pred = regressor.predict(test_X)

for y in range(len(y_pred)):
    if y_pred[y] >= 0.5:
        y_pred[y] = 1
    else:
        y_pred[y] = 0
        
with open('result.csv', 'w+') as f:
    passenger_ids = [x for x in test_df['PassengerId']]
    itr = 0
    op_string = ''
    op_string += 'PassengerId,Survived\n'
    for y in y_pred:
        op_string += str(passenger_ids[itr]) +','+ str(int(y[0])) + '\n'
        itr += 1
    f.write(op_string)
        