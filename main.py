import numpy as np
from sklearn.linear_model import LinearRegression

# multiple linear regression example from 
# https://heartbeat.fritz.ai/implementing-multiple-linear-regression-using-sklearn-43b3d3f2fe8b

X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
# y = 1 * x_0 + 2 * x_1 + 3
y = np.dot(X, np.array([1, 2])) + 3
reg = LinearRegression().fit(X, y)

print( reg.score(X, y) )

print( reg.coef_ )

print( reg.intercept_ )

print( reg.predict(np.array([[3, 5]])) )