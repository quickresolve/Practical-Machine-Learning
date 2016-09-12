import pandas as pd
import quandl, math
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression

# Dataframe - get datasets (ticker) from Quandl.com to work with
df = quandl.get('WIKI/GOOGL')

print(df.head())
# allows you to see all features of this dataset in the terminal


df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume',]]
# Single out features you want to show relationships between in this case we are looking at the opening, high, low, and closing stock price each day in this dataset

df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.00
# Calculates the High minus Low % per row

df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.00
#Calculates the percent change of the new - old/old *100


df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]


forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)
# Input outlier for your dataset to improve accuracy and keep from getting rid of data

forecast_out = int(math.ceil(0.01*len(df)))
# length of the dataset you want to forecast out from

df['label'] = df[forecast_col].shift(-forecast_out)
# df.dropna(inplace=True)
# print(df.head())


X = np.array(df.drop(['label'],1))
# features

y = np.array(df['label'])
# labels

X = preprocessing.scale(X)
# scales new values alongside your other values - adds to processing time

y = np.array(df['label'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
# You want to train and test on different datasets

print(accuracy)
