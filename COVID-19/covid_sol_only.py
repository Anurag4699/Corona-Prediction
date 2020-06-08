# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 00:17:53 2020

@author: Anurag
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

train = pd.read_excel("Train_dataset.xlsx")
test = pd.read_excel("Test_dataset.xlsx")

train_df = train.drop(columns=['Name', 'Designation', 'people_ID'], axis=1)
test_df = test.drop(columns=['Name', 'Designation', 'people_ID'], axis=1)

test_df= test_df.replace("nan", np.nan)
train_df = train_df.replace("nan", np.nan)
    
num_train = train_df.select_dtypes(include='number');
cat_train = train_df.select_dtypes(exclude='number');
num_test = test_df.select_dtypes(include='number');
cat_test = test_df.select_dtypes(exclude='number');

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

for col in num_train:
    train_df[[col]] = imputer.fit_transform(train_df[[col]]).round()

imputer1 = SimpleImputer(strategy='most_frequent')
train_df["Mode_transport"].mode()
train_df["Mode_transport"].fillna("Public", inplace=True)

train_df["cardiological pressure"].fillna("Normal", inplace=True)
train_df["comorbidity"].fillna("None", inplace=True)
train_df["Occupation"].fillna("Unknown", inplace=True)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
encoder = LabelEncoder()
encoder_2 = OneHotEncoder(categories="auto")

Y = train_df.Infect_Prob
X = train_df.drop(columns=['Infect_Prob'], axis=1)

for col in cat_train:
    X[col]  = encoder.fit_transform(X[col])
    
for col in cat_test:    
    test_df[col] = encoder.fit_transform(test_df[col])
    
X = pd.get_dummies(X, columns=["Occupation","Mode_transport","comorbidity","cardiological pressure","Region"])
test_df = pd.get_dummies(test_df, columns=["Occupation","Mode_transport","comorbidity","cardiological pressure","Region"])

X = X.iloc[:, 1:]
test_df = test_df.iloc[:, 1:]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
test_df = sc.fit_transform(test_df)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
test_df = sc.fit_transform(test_df)

Y=Y.to_frame()
#This is to convert series into dataframe

Y=Y.to_numpy()
#To convert dataframe to array

Y = sc.fit_transform(Y)

X_train = pd.DataFrame(data=X_train)
y_train = pd.DataFrame(data=y_train)

#FEATURE SELECTION
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100)
rfe = RFE(model,30)
fit = rfe.fit(X_train,y_train.values.ravel())

print("Num Features: %d" % fit.n_features_)
print("Selected Features: %s" % fit.support_)
print("Feature Ranking: %s" % fit.ranking_)

cols = [0,18,19,20,21,23,24,25,26,27,36,38,39,42]

#Taking 10 features
#cols = [0,3,5,6,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43]
X_train1 = X_train.drop(X_train.columns[cols], axis=1)
X_test1 = X_test.drop(X_test.columns[cols], axis=1)

#TESTING THE ALGORITHMS

#LASSO
from sklearn.linear_model import Lasso
lasso = Lasso(alpha=0.01,max_iter=10000)
lasso.fit(X_train1,y_train)


train_score = lasso.score(X_train1,y_train)
print("Training Set Score : ",train_score)

test_score = lasso.score(X_test1,y_test)
print("Test Set Score : ",test_score)

print("Number of features used : {}".format(np.sum(lasso.coef_ !=0)))


#RIDGE
from sklearn.linear_model import Ridge
ridge = Ridge()
ridge.fit(X_train1,y_train)

train_score = ridge.score(X_train1,y_train)
print("Training Set Score : ",train_score)

test_score = ridge.score(X_test1,y_test)
print("Test Set Score : ",test_score)


#Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor(n_estimators=100)
forest.fit(X_train, y_train.values.ravel())

print("Accuracy on training set: {:.3f}".format(forest.score(X_train, y_train)))

print("Accuracy on test set: {:.3f}".format(forest.score(X_test, y_test)))