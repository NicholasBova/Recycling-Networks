import numpy as np
import matplotlib.pyplot as plt
#import os # we need to scan directory
import pvml

#classes = open('classes.txt').read().split()


##cnn = pvml.CNN.load("pvmlnet.npz")
##
def accuracy(net, X, Y):
    ''' Compute the accuracy.

    : param net: MLP neural network.
    : param X: array like.
    : param Y: array like.
    : return acc * 100: number.
    
    '''
    labels, probs = net.inference(X)
    acc = (labels == Y).mean()
    return acc * 100

Xtrain, Ytrain = np.load("Data/train.npz").values()
Xtrain = Xtrain.reshape(60000, 28, 28, 1) - 0.5
print("Training set after reshape: ", Xtrain.shape, Ytrain.shape)
Xtest, Ytest = np.load("Data/test.npz").values()
Xtest = Xtest.reshape(10000, 28, 28, 1) -0.5
print("Test set after reshape: ", Xtest.shape, Ytest.shape)

### create a CNN and train it




network = pvml.CNN([1, 12, 32, 48, 10], [7, 3, 3, 3], [2, 2, 1, 1], [0, 0, 0, 0])

epochs = []
batch_size = 100
lr = 0.0001

train_accs = []
test_accs = []

print('STARTING TRAINING')
plt.ion()
for epoch in range(201):
    steps = Xtrain.shape[0] // batch_size
    network.train(Xtrain, Ytrain, lr=lr, steps=steps, batch=batch_size)
    if epoch % 10 == 0:
       train_acc = accuracy(network, Xtrain, Ytrain)
       test_acc = accuracy(network, Xtest, Ytest)
       print(epoch, train_acc, test_acc)

       train_accs.append(train_acc)
       test_accs.append(test_acc)
       epochs.append(epoch)

       plt.clf()
       plt.plot(epochs, train_accs, color = 'r')
       plt.plot(epochs, test_accs, color ='blue')
       plt.title('Training and test accuracies with CNN')
       plt.xlabel("Epoch")
       plt.ylabel("Accuracy [%]")
       plt.legend(["train", "test"])
       plt.pause(0.01)

plt.savefig('figures/cnn_accuracy.png')
network.save('MLP/dress_network_cnn.npz')
plt.ioff()
print('TRAINING FINISHED')
plt.show()

### evaluation
train_predictions, train_probs = network.inference(Xtrain)
train_acc = 100 * (train_predictions == Ytrain).mean()
print('training accuracy: ', train_acc)

test_predictions, test_probs = network.inference(Xtest)
test_acc = 100 * (test_predictions == Ytest).mean()
print('test accuracy: ', test_acc)
  
