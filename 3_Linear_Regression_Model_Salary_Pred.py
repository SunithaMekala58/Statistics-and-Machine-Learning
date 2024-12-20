import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

dataset = pd.read_csv(r'D:\SUNITHA\Sample Datasets\Salary_Data.csv')

#yrs of exp is independent variable
X = dataset.iloc[:, :-1]

#Salary is dep
y = dataset.iloc[:, -1]

#Split the data

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20, train_size= 0.80, random_state = 0)

X_train = X_train.values.reshape(-1,1)
X_test = X_test.values.reshape(-1,1)

#Now building simple linear regresssion model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

m_slope = regressor.coef_
print(m_slope)

c_interept = regressor.intercept_
print(c_interept)

y_20 = m_slope * 20 + c_interept
y_20

comparison = pd.DataFrame({'Actual':y_test, 'Predicted': y_pred})
print(comparison)

#Statistics in machine learning
dataset.mean()

dataset['Salary'].mean()

dataset.median()

dataset['Salary'].median()

dataset.mode()

dataset.var()

dataset['Salary'].var()

dataset.std()

dataset['Salary'].std()

#Coefficent of variance(cv)
from scipy.stats import variation

variation(dataset.values)

variation(dataset['Salary'])

#Correlation
dataset.corr()

dataset['Salary'].corr(dataset['YearsExperience'])

#Skewness
dataset.skew()

dataset['Salary'].skew()

#Z-Score - Inferential Statistics
import scipy.stats as stats

dataset.apply(stats.zscore)

stats.zscore(dataset['Salary'])

#SSR (ANNOVA)
y_mean = np.mean(y)
SSR = np.sum((y_pred-y_mean)**2)
print(SSR)

#SSE is actual - predicted
y = y[0:6]
SSE = np.sum((y-y_pred)**2)
print(SSE)

#SST
mean_total = np.mean(dataset.values)
SST = np.sum((dataset.values-mean_total)**2)
print(SST)

#R2 Square

r_square = 1- (SSR/SST)
r_square

print(regressor)
