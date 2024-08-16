import numpy as np

def load_reshape(path):
    X, Y = np.load(path).values()
    X = X.reshape(X.shape[0], -1)
    return X, Y

def mindist_train(X, Y):
    k = Y.max() + 1
    n = X.shape[1]
    means = np.empty((k, n))
    for c in range(k):
        means[c, :] = X[Y == c, :].mean(0)
    return means

def mindist_inference(X, means):
    k = means.shape[0]
    m = X.shape[0]
    squared_dists = np.empty((m, k))
    for c in range(k):
        squared_dists[:, c] = ((X - means[c, :]) ** 2).sum(1)
    labels = np.argmin(squared_dists, 1)
    return labels

Xtrain, Ytrain = load_reshape("Data/train.npz")
print("Training set after reshape: ", Xtrain.shape, Ytrain.shape)
Xtest, Ytest = load_reshape("Data/test.npz")
print("Test set after reshape: ", Xtest.shape, Ytest.shape)

m = mindist_train(Xtrain, Ytrain)

train_labels = mindist_inference(Xtrain, m)
train_acc = 100 * (train_labels == Ytrain).mean()
print('train accuracy: ', train_acc)

test_labels = mindist_inference(Xtest, m)
test_acc = 100 * (test_labels == Ytest).mean()
print('test accuracy: ', test_acc)
