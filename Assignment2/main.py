import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('train.csv')

# making x the [x2, x3, x4] discussed in readme
X = data[['X2','X3','X4']]

# Y is the output (house price per unit)
y = data['Y']


print("===========")
print("Ordinary least squares")
print("===========")

# Ordinary least squares
reg = LinearRegression().fit(X, y)

# test fitness on test data
print("test score:")
test = pd.read_csv('test.csv')
print( reg.score(test[['X2','X3','X4']], test['Y']) )
print("train score:")
print( reg.score(X, y) )

# coef result
print("coefs:")
print( reg.coef_ )
# intercept result
print("intercept:")
print( reg.intercept_ )

print("===========")
print("Linear regression with gradient descent")
print("===========")

# Ordinary least squares
sgd = SGDRegressor(max_iter=1000000, tol=1e-3, alpha=.0001)
reg = make_pipeline(StandardScaler(), sgd)
fitted = reg.fit(X,y)

print("iterations:")
print( sgd.n_iter_)
print("test score:")
test = pd.read_csv('test.csv')
print( fitted.score( test[['X2','X3','X4']], test['Y']) )
print("train score:")
print( fitted.score(X, y) )

# coef result
print("coefs:")
print(sgd.coef_)
# intercept result
print("intercept:")
print(sgd.intercept_)