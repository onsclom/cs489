import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

data = pd.read_csv('train.csv')

# making x the as discussed in readme
X = data[['X0','X1','X2','X3']]

# Y wether or not genuine
Y = data['Y']

print(X)
print(Y)

classifier = KNeighborsClassifier( 3 )
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