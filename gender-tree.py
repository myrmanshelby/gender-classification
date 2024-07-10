from sklearn import tree
from sklearn import neighbors
from sklearn import gaussian_process


# [height in cm, weight in kgs, shoe size]

X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37],
     [166, 65, 40], [190, 90, 47], [175, 64, 39], [177, 70, 40], [159, 55, 37],
     [171, 75, 42], [181, 85, 43]]

Y = ['male', 'female', 'female', 'female', 'male', 'male', 'male',
     'female', 'male', 'female', 'male']

clfTree = tree.DecisionTreeClassifier()
clfTree = clfTree.fit(X,Y)
predictionTree = clfTree.predict([[175, 68, 41]])
print(predictionTree)

clfNeighbors = neighbors.KNeighborsClassifier(n_neighbors=3)
clfNeighbors = clfNeighbors.fit(X, Y)
predictionKNN = clfNeighbors.predict([[175, 68, 41]])
print(predictionKNN)

clfGpc = gaussian_process.GaussianProcessClassifier(kernel=1.0*gaussian_process.kernels.RBF(1.0),random_state=0).fit(X,Y)
predictionGpc = clfGpc.predict([[175, 68, 41]])
print(predictionGpc)