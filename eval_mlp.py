import numpy as np
import matplotlib.pyplot as plt
import pvml
import inquirer

questions = [inquirer.List('normalization', message = 'What normalization technique do you want to use?',
                           choices = ['none', 'mean_variance', 'whitening',
                                      'L1', 'L2'],),]

answers = inquirer.prompt(questions)
print('You chose ' + answers["normalization"] + ' normalization')

questions1 = [inquirer.List('layers', message = 'How many hidden layers do you want to use?',
                           choices = ['none', '1', '2'],),]
answers1 = inquirer.prompt(questions1)
print('You chose ' + answers1["layers"] + ' hidden layers')

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

def load_reshape(path):
    X, Y = np.load(path).values()
    X = X.reshape(X.shape[0], -1)
    return X, Y

def l2_normalization(X):
    q = np.sqrt((X ** 2).sum(1, keepdims = True))
    q = np.maximum(q, 1e-15)
    X = X / q
    return X

def l1_normalization(X):
    q = np.abs(X).sum(1, keepdims = True)
    q = np.maximum(q, 1e-15)
    X = X / q
    return X

def meanvar_normalization(Xtrain, Xval):
    mu = Xtrain.mean(0, keepdims = True) # (1600,)
    std = Xtrain.std(0, keepdims = True)
    Xtrain = (Xtrain - mu) / std
    Xval = (Xval - mu) / std
    return Xtrain, Xval

def whitening(Xtrain , Xval):
    mu = Xtrain.mean(0, keepdims = True)
    sigma = np.cov(Xtrain.T)
    evals , evecs = np.linalg.eigh(sigma)
    w = evecs/np.sqrt(evals)
    Xtrain = (Xtrain - mu) @ w
    Xval = (Xval - mu) @ w
    return Xtrain , Xval

classes = open('classes.txt').read().split()

### download the data
Xtrain, Ytrain = load_reshape("Data/train.npz")
#print("Training set after reshape: ", Xtrain.shape, Ytrain.shape)
Xtest, Ytest = load_reshape("Data/test.npz")
#print("Test set after reshape: ", Xtest.shape, Ytest.shape)

if answers["normalization"] == 'mean_variance':
    Xtrain, Xtest = meanvar_normalization(Xtrain, Xtest)
elif answers["normalization"] == 'whitening':
    Xtrain, Xtest = whitening(Xtrain, Xtest)
elif answers["normalization"] == 'L1':
    Xtrain = l1_normalization(Xtrain)
    Xtest = l1_normalization(Xtest)
elif answers["normalization"] == 'L2':
    Xtrain = l2_normalization(Xtrain)
    Xtest = l2_normalization(Xtest)

### upload the MLP
#net = pvml.MLP.load("MLP\mlp_" + answers["normalization"] + '_' + answers1["layers"] + "hidden.npz")
net = pvml.MLP.load("MLP\mlp_2hidden.npz")

###evaluate the model
train_acc, train_labels, train_probs = accuracy(net, Xtrain, Ytrain)
print("Training accuracy: ", train_acc)

test_acc, test_labels, test_probs = accuracy(net, Xtest, Ytest)
print("Test accuracy: ", test_acc)

### CONFUSION MATRIX
cm = np.zeros((10, 10))
for i in range(Xtest.shape[0]):
    cm[Ytest[i], test_labels[i]] += 1
   
total = cm.sum(1, keepdims = True)
cm /= total
print(cm)

plt.figure(figsize = [25,25])
# try to do a different
plt.imshow(cm)
for i in range(10):
    for j in range(10):
        plt.text(j, i, int(100 * cm[i, j]), color = 'pink')
plt.xticks(range(10), classes[:10], rotation='vertical')
plt.yticks(range(10), classes[:10])
#plt.savefig('figures\cm_' + answers["normalization"] + '_' + answers1["layers"] + 'hidden.png')
plt.show()

### WEIGHTS
if answers1["layers"] == 'none':
   weight = net.weights[0]
   fig = plt.figure(figsize = [14, 7])
   columns = 5
   rows = 2
   for i in range(1,11):
       fig.add_subplot(rows, columns, i)
       plt.imshow(weight[:,i-1].reshape(28,28), cmap = 'bwr', vmin = -0.6,
               vmax = 0.6, aspect = 'auto')
       plt.axis('off')
   #plt.savefig('figures\weights_' + answers["normalization"] + '_' + answers1["layers"] + 'hidden.png')
   plt.show()

max = np.array([[0,0,0],[0,0,0],[0,0,0]]).astype(float)
for j in range(test_labels.size):
    if test_labels[j] != Ytest[j]:
            if all(x < test_probs[j,test_labels[j]] for x in max[:,0]):
                print(test_probs[j, test_labels[j]])
                max[2,0] = test_probs[j,test_labels[j]]
                max[2,1] = j
                max[2,2] = test_labels[j]
                max = max[np.argsort(max[:,0]),:][::-1]

print(max)
            
    # specifichiamo le predizioni sbagliate
    
for m in max:
     files = Xtest #os.listdir("images/test/" +classes[int(m[1]//20)] )
     print("test",  classes[int(m[1]//20)], int(m[1]%20))
     print("Ind folder: ", classes[int(m[1]//20)], " Ind file: ", int(m[1]%20), " Wrongly predicted class: ", classes[int(m[2])])
    


