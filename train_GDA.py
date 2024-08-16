import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, precision_recall_curve

def load_reshape(path):
    X, Y = np.load(path).values()
    X = X.reshape(X.shape[0], -1)
    return X, Y

Xtrain, Ytrain = load_reshape("Data/train.npz")
print("Training set after reshape: ", Xtrain.shape, Ytrain.shape)
Xtest, Ytest = load_reshape("Data/test.npz")
print("Test set after reshape: ", Xtest.shape, Ytest.shape)

def ogda_train(X, Y):
    k = Y.max() + 1
    m, n = X.shape
    means = np.empty((k, n))
    cov = np.zeros((n, n))
    priors = np.bincount(Y) / m
    for c in range(k):
        means[c, :] = X[Y == c, :].mean(0)
        cov += priors[c] * np.cov(X[Y == c, :].T)
    icov = np.linalg.inv(cov)
    W = -(icov @ means.T)
    q = 0.5 * ((means @ icov) * means).sum(1)
    b = q - np.log(priors)
    return W, b

def ogda_inference(X, W, b):
    scores = X @ W + b.T
    labels = np.argmin(scores, 1)
    return labels

classes = open('classes.txt').read().split()

W, b = ogda_train(Xtrain, Ytrain)
print(W.shape, b.shape)
np.savez_compressed('Data/ogda.npz', W, b)

predictions_train = ogda_inference(Xtrain, W, b)
test_labels = ogda_inference(Xtest, W, b)
#print(predictions)

train_acc = 100 * (Ytrain == predictions_train).mean()
print('training accuracy: ', train_acc)
test_acc = 100 * (Ytest == test_labels).mean()
print('test accuracy: ', test_acc)

cm = np.zeros((10, 10))
for i in range(Xtest.shape[0]):
    cm[Ytest[i], test_labels[i]] += 1
   
total = cm.sum(1, keepdims = True)
cm /= total
print(cm)

##precision = precision_score(Ytest, test_labels, average = 'macro')
##print('precision: ', precision)
##recall = recall_score(Ytest, test_labels, average = 'macro')
##print('recall: ', recall)


plt.figure(figsize = [25,25])
# try to do a different
plt.imshow(cm)
for i in range(10):
    for j in range(10):
        plt.text(j, i, int(100 * cm[i, j]), color = 'pink')
plt.xticks(range(10), classes[:10], rotation='vertical')
plt.yticks(range(10), classes[:10])
plt.savefig('figures\cm_GDA.png')
plt.show()



