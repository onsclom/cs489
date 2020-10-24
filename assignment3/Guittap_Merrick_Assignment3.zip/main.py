import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

data = pd.read_csv('train.csv')

# making x the as discussed in readme
X = data[['X0','X1','X2','X3']]

# Y wether or not genuine
Y = data['Y']

print(X)
print(Y)

classifier = SGDClassifier(max_iter=1000, tol=1e-3, )
clf = make_pipeline(StandardScaler(), classifier)
fitted = clf.fit(X, Y)

print("coef:")
print(classifier.coef_)

print("intercept:")
print(classifier.intercept_)

print("score on training:")
print( fitted.score(X, Y) )

print("score on test:")
data = pd.read_csv('test.csv')
# making x the as discussed in readme
X = data[['X0','X1','X2','X3']]
# Y wether or not genuine
Y = data['Y']
print( fitted.score(X, Y) )

print("iterations:")
print( classifier.n_iter_)







# print("===========")
# print("Ordinary least squares")
# print("===========")

# # Ordinary least squares
# reg = LinearRegression().fit(X, y)

# # test fitness on test data
# print("test score:")
# test = pd.read_csv('test.csv')
# print( reg.score(test[['X2','X3','X4']], test['Y']) )
# print("train score:")
# print( reg.score(X, y) )

# # coef result
# print("coefs:")
# print( reg.coef_ )
# # intercept result
# print("intercept:")
# print( reg.intercept_ )

# print("===========")
# print("Linear regression with gradient descent")
# print("===========")

# # Ordinary least squares
# sgd = SGDRegressor(max_iter=1000000, tol=1e-3, alpha=.0001)
# reg = make_pipeline(StandardScaler(), sgd)
# fitted = reg.fit(X,y)

# print("iterations:")
# print( sgd.n_iter_)
# print("test score:")
# test = pd.read_csv('test.csv')
# print( fitted.score( test[['X2','X3','X4']], test['Y']) )
# print("train score:")
# print( fitted.score(X, y) )

# # coef result
# print("coefs:")
# print(sgd.coef_)
# # intercept result
# print("intercept:")
# print(sgd.intercept_)