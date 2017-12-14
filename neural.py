import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.neural_network import MLPClassifier
import sklearn
from sklearn.externals import joblib


def kFold(X, y):
    kf = KFold(n_splits=5, shuffle=True)
    neurons = range(100,500, 10)
    HL = range(1,10)
    accuracyMatrix = np.zeros((len(neurons), len(HL)))
    for i, numNeurons in enumerate(neurons):
        for j, numLayers in enumerate(HL):
            avgAccuracy = 0
            counts = 0
            for train_index, dev_index in kf.split(X):
                counts += 1
                X_train, X_dev = X[train_index], X[dev_index]
                y_train, y_dev = y[train_index], y[dev_index]
                clf = MLPClassifier(hidden_layer_sizes = (numNeurons,numLayers), random_state=1, verbose = 0)
                clf.fit(X_train,y_train)
                y_pred = clf.predict(X_dev)
                accuracy = float(sum(1 for x,y in zip(y_pred,y_dev) if x == y)) / len(y_pred)
                avgAccuracy += accuracy
            avgAccuracy /= float(counts)
            accuracyMatrix[i, j] = avgAccuracy
            print("Accuracy:", avgAccuracy, " Number of Neurons:", numNeurons, "Number of HL:", numLayers)
    np.save("accuracyMatrix.npy", avgAccuracy)
    
def main():
    X = []
    with open("X.txt") as file:
        for line in file:
            X.append([float(x) for x in line.strip().split(" ")])
    y = []
    with open("y.txt") as file:
        for line in file:
            y.append([float(x) for x in line.strip().split(" ")])
    X = np.array(X)
    y = np.array(y)
    print X.shape, y.shape
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, shuffle=True)
    #kFold(X_train, y_train)
    clf = MLPClassifier(hidden_layer_sizes = (500, 8), random_state=1, verbose = 1)
    clf.fit(X_train, y_train)
    joblib.dump(clf, "net.sav")
    y_pred = clf.predict(X_test)
    accuracy = float(sum(1 for x,y in zip(y_pred,y_test) if x == y)) / len(y_pred)
    print accuracy
    
        

if __name__=="__main__":
    main()
