# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 07:58:53 2021

@author: thejutn
"""

import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv('advertising.csv')
data.head()

fig,axis=plt.subplots(1,3,sharey=True)
data.plot(kind='scatter',x='TV',y='Sales',ax=axis[0],figsize=(14,7))
data.plot(kind='scatter',x='Radio',y='Sales',ax=axis[1])
data.plot(kind='scatter',x='Newspaper',y='Sales',ax=axis[2])


feature_cols = ['TV']
X = data[feature_cols]
y = data.Sales

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X, y)
print(lr.intercept_)
print(lr.coef_)


result=6.974+0.0554*50
print(result)

X_new=pd.DataFrame({'TV':[data.TV.min(),data.TV.max()]})
X_new.head()

preds = lr.predict(X_new)
preds


data.plot(kind='scatter', x='TV', y='Sales')
plt.plot(X_new, preds, c='red', linewidth=2)


import statsmodels.formula.api as smf
lr = smf.ols(formula='Sales ~ TV', data=data).fit()
lr.conf_int()

lr.pvalues

lr.rsquared


feature_cols = ['TV', 'Radio', 'Newspaper']
X = data[feature_cols]
y = data.Sales

lm = LinearRegression()
lm.fit(X, y)

# print intercept and coefficients
print(lm.intercept_)
print(lm.coef_)

lr = smf.ols(formula='Sales ~ TV + Radio + Newspaper', data=data).fit()
lr.conf_int()
lr.summary()

lr= smf.ols(formula='Sales ~ TV + Radio', data=data).fit()
lr.rsquared

lr = smf.ols(formula='Sales ~ TV + Radio + Newspaper', data=data).fit()
lr.rsquared