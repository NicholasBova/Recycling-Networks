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
### compute accuracy
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

def load_reshape(path):
    X, Y = np.load(path).values()
    X = X.reshape(X.shape[0], -1)
    return X, Y

def l2_normalization(X):
    q = np.sqrt((X ** 2).sum(1, keepdims = True))
    q = np.maximum(q, 1e-15)
    X = X / q
    return X

###L_1 NORMALIZATION
def l1_normalization(X):
    q = np.abs(X).sum(1, keepdims = True)
    q = np.maximum(q, 1e-15)
    X = X / q
    return X

def whitening(Xtrain , Xval):
    mu = Xtrain.mean(0, keepdims = True)
    sigma = np.cov(Xtrain.T)
    evals , evecs = np.linalg.eigh(sigma)
    w = evecs/np.sqrt(evals)
    Xtrain = (Xtrain - mu) @ w
    Xval = (Xval - mu) @ w
    return Xtrain , Xval

def meanvar_normalization(Xtrain, Xval):
    mu = Xtrain.mean(0, keepdims = True) # (1600,)
    std = Xtrain.std(0, keepdims = True)
    #save values of mu and std
    # np.savez('mean_var.npz', mu, std)
    # normalize
    Xtrain = (Xtrain - mu) / std
    Xval = (Xval - mu) / std
    return Xtrain, Xval
   
    
   

Xtrain, Ytrain = load_reshape("Data/train.npz")
print("Training set after reshape: ", Xtrain.shape, Ytrain.shape)
Xtest, Ytest = load_reshape("Data/test.npz")
print("Test set after reshape: ", Xtest.shape, Ytest.shape)

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
    


### 1 hidden: 89
### 2 hidden: 183, 43
if answers1["layers"] == 'none':
    net = pvml.MLP([784, 10])
elif answers1["layers"] == '1':
    net = pvml.MLP([784, 89, 10])
elif answers1["layers"] == '2':
    net = pvml.MLP([784, 183, 43, 10])


### TRAINING
m = Ytrain.size

### activate interacting mode with ion()
plt.ion()
train_accs = []
test_accs = []
epochs = []
batch_size = 10

print('STARTING TRAINING')
### what if the number of epochs changes
for epoch in range(201):
    # parameters: training data and learning rate
    # using SGD 
    net.train(Xtrain, Ytrain, 1e-4, batch = batch_size, steps = m // batch_size)
    if epoch % 5 == 0:
      # return predictions and probability
      train_acc = accuracy(net, Xtrain, Ytrain)
      test_acc = accuracy(net, Xtest, Ytest)
      print(epoch, train_acc, test_acc)
    
      train_accs.append(train_acc)
      test_accs.append(test_acc)
      epochs.append(epoch)
    
      plt.clf() # clear the plots
      plt.plot(epochs, train_accs)
      plt.plot(epochs, test_accs)
      if answers1["layers"] == 'none':
          plt.title('Network structure: (784, 10)')
      elif answers1["layers"] == '1':
          plt.title('Network structure: (784, 89, 10)')
      elif answers1["layers"] == '2':
          plt.title('Network structure: (784, 183, 43, 10)')
          
      plt.xlabel("Epoch")
      plt.ylabel("Accuracy [%]")
      plt.legend(['train', 'test'])
      plt.pause(0.01) # stops for a given amount of time (even smaller)
      # common to save after each epoch
      plt.savefig('figures\mlp_' + answers["normalization"] + '_' + answers1["layers"] + 'hidden.png')
      net.save("MLP\mlp_" + answers["normalization"] + '_' + answers1["layers"] + "hidden.npz")

print("TRAINING FINISHED")
plt.ioff() # interactive off
plt.show() # keep the window open
