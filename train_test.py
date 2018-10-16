# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 14:08:18 2018

@author: Anand
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv('new_data.csv')

dates = list(pd.Series(data['Date']))
times = list(pd.Series(data['Time']))
items = list(pd.Series(data['Items']))
transactions = pd.Series(data['Transactions'])
transactions.columns = ['Transactions']


x = pd.DataFrame([dates, times, items])
x = x.transpose()
x.columns = ['Date', 'Time', 'Items']
y = transactions


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)


linear_regression = LinearRegression()


classifier = linear_regression.fit(x_train, y_train)

print(classifier.score(x_test, y_test))

p = [[20170304, 135647, 35]]

prediction = linear_regression.predict(p)

print(prediction)