import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

dataset = pd.read_csv(r'D:\data.csv')

X = dataset.iloc[:, :-1].values

y = dataset.iloc[:,3].values

from sklearn.impute import SimpleImputer

imputer = SimpleImputer()
#imputer = SimpleImputer(strategy='median')
#imputer = SimpleImputer(strategy='most_frequent')

imputer = imputer.fit(X[:, 1:3])

X[:, 1:3] = imputer.transform(X[:, 1:3])

from sklearn.preprocessing import LabelEncoder

LabelEncoder_X = LabelEncoder()

LabelEncoder_X.fit_transform(X[:,0])

X[:,0] = LabelEncoder_X.fit_transform(X[:,0])

LabelEncoder_y = LabelEncoder()

y = LabelEncoder_y.fit_transform(y)

#Split the data

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, train_size = 0.7, random_state = 0)

