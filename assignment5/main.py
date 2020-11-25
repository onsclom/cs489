import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

data = pd.read_csv('train.csv')

# making x the as discussed in readme
X = data[['X0','X1','X2','X3']]

# Y wether or not genuine
Y = data['Y']

print(X)
print(Y)

classifier = MLPClassifier(
        hidden_layer_sizes=500,
        activation='relu',
        learning_rate='adaptive')
clf = make_pipeline(StandardScaler(), classifier)
fitted = clf.fit(X, Y)

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
