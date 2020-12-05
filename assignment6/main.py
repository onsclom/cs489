import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('train.csv')

# making x the [x2, x3, x4] discussed in readme
X = data[['X2','X3','X4']]

# Y is the output (house price per unit)
y = data['Y']

# Ordinary least squares
sgd = MLPRegressor(max_iter=10000, hidden_layer_sizes=(00,))
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
print(sgd.n_layers_)
#print(sgd.coef_)
# intercept result
print("intercept:")
#print(sgd.intercept_)
