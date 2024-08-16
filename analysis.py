import numpy as np
import matplotlib.pyplot as plt
import pvml
#import inquirer
#import itertools
import os # we need to scan directory
import image_features


def accuracy(net, X, Y):
    ''' Compute the accuracy.

    : param net: MLP neural network.
    : param X: array like.
    : param Y: array like.
    : return acc * 100: number.
    
    '''
    labels, probs = net.inference(X)
    acc = (labels == Y).mean()
    return acc * 100, labels, probs

classes = open('classes.txt').read().split()

Xtrain, Ytrain = np.load("Data/train.npz").values()
Xtrain = Xtrain.reshape(60000, 28, 28, 1) - 0.5
print("Training set after reshape: ", Xtrain.shape, Ytrain.shape)
Xtest, Ytest = np.load("Data/test.npz").values()
Xtest = Xtest.reshape(10000, 28, 28, 1) -0.5
print("Test set after reshape: ", Xtest.shape, Ytest.shape)



#net = pvml.MLP.load("MLP/cakes-mlp_2_" + answers["low-level_features"] + ".npz")
net = pvml.CNN.load("MLP/dress_network_cnn.npz")

### evaluate the model
train_acc, train_labels, train_probs = accuracy(net, Xtrain, Ytrain)
print("Training accuracy: ", train_acc)


test_acc, test_labels, test_probs = accuracy(net, Xtest, Ytest)
print("Test accuracy: ", test_acc)

### CONFUSION MATRIX
##cm = np.zeros((10, 10))
##for i in range(Xtest.shape[0]):
##    cm[Ytest[i], test_labels[i]] += 1
##   
##total = cm.sum(1, keepdims = True)
##cm /= total
##print(cm)
##
##plt.figure(figsize = [25,25])
### try to do a different
##plt.imshow(cm)
##for i in range(10):
##    for j in range(10):
##        plt.text(j, i, int(100 * cm[i, j]), color = 'pink')
##plt.xticks(range(10), classes[:10], rotation='vertical')
##plt.yticks(range(10), classes[:10])
##plt.savefig('figures\cm_cnn')
##plt.show()

maxs = np.array([[0,0,0],[0,0,0],[0,0,0]]).astype(float)
List = []
Predictions = []
for j in range(test_labels.size):
    if test_labels[j] != Ytest[j]:
            if all(x < test_probs[j,test_labels[j]] for x in maxs[:,0]):
                print(test_probs[j, test_labels[j]])
                maxs[2,0] = test_probs[j,test_labels[j]]
                maxs[2,1] = j
                maxs[2,2] = test_labels[j]
                maxs = maxs[np.argsort(maxs[:,0]),:][::-1]
                List.append(j)
                Predictions.append(test_labels[j])

##print(maxs)
##print(len(List))
##print(len(Predictions))

Xtest , Ytest = np.load("Data/test.npz").values()

w = 16
h = 16
fig = plt.figure(figsize=(8, 8))
columns = 3
rows = 2

t = 0
for a in range(len(List)):
    j = List[a]
    h = Predictions[a]
    image = Xtest[j, :]
    plt.figure(figsize = (8,8))
    plt.imshow(image, cmap = 'Greys')
    #plt.colorbar()
    plt.title(classes[Ytest[j]] + ' wrongly predicted as ' + classes[h])
    plt.savefig('figures/wrong_' + str(t) + '.png')
    plt.show()
    t += 1
##for i in range(1,7):
##    j = List[i-1]
##    h = Predictions[i-1]
##    image = Xtest[j, :]
##    fig.add_subplot(rows, columns, i)
##    plt.tick_params(left=False,
##                bottom=False,
##                labelleft=False,
##                labelbottom=False)
##    plt.title(classes[Ytest[j]] + " predicted as " + classes[h] )
##    plt.imshow(image,cmap='gray')
##plt.show()

##BestValues = [17,23,787, 382,1300,2022]
##Bestpredictions = [2,5,3,0,5,9]
##
##w = 16
##h = 16
##fig = plt.figure(figsize=(8, 8))
##columns = 3
##rows = 2
##
##for i in range(1,7):
##    j = BestValues[i-1]
##    h = Bestpredictions[i-1]
##    image = Xtest[j, :]
##    fig.add_subplot(rows, columns, i)
##    plt.tick_params(left=False,
##                bottom=False,
##                labelleft=False,
##                labelbottom=False)
##    plt.title(words[Ytest[j]] + " predicted as " + words[h] )
##    plt.imshow(image,cmap='gray')
##plt.show()            


##### individuo errori
##max = np.array([[0,0,0],[0,0,0],[0,0,0]]).astype(float)
##for j in range(test_labels.size):
##    if test_labels[j] != Ytest[j]:
##            if all(x < test_probs[j,test_labels[j]] for x in max[:,0]):
##                print(test_probs[j, test_labels[j]])
##                max[2,0] = test_probs[j,test_labels[j]]
##                max[2,1] = j
##                max[2,2] = test_labels[j]
##                max = max[np.argsort(max[:,0]),:][::-1]
##
##print(max)
##            
##    # specifichiamo le predizioni sbagliate
##    
##for m in max:
##     files = os.listdir("images/test/" +classes[int(m[1]//20)] )
##     print("images/test/" + classes[int(m[1]//20)]+"/"+files[int(m[1]%20)])
##     print("Ind folder: ", classes[int(m[1]//20)], " Ind file: ", files[int(m[1]%20)], " Wrongly predicted class: ", classes[int(m[2])])
##    
##

