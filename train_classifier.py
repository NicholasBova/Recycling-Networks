import numpy as np
import pvml
import inquirer
import matplotlib.pyplot as plt

questions = [inquirer.List('low-level_features', message = 'What features do you want to use?',
                           choices = ['color_histogram', 'edge_direction', 'gray_level_cooccurrence_matrix',
                                      'rgb_cooccurrence_matrix'],),]

answers = inquirer.prompt(questions)
print('You chose ' + answers["low-level_features"])

questions1 = [inquirer.List('normalization', message = 'Do you want to use L_2 normalization?',
                           choices = ['yes', 'no'],),]
answers1 = inquirer.prompt(questions1)
print('Use normalization? ' + answers1["normalization"])

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

def load_file(filename):
    ''' Load the data saved in a file.

    : param filename: string.
    : return X, Y: array like.
    
    '''
    data = np.load(filename)
    ### for X we take all the columns but the last one
    X = data["arr_0"]
    ### for Y we take only the last column
    Y = data["arr_1"]
    return X, Y

def l2_normalization(X):
    q = np.sqrt((X ** 2).sum(1, keepdims = True))
    q = np.maximum(q, 1e-15)
    X = X / q
    return X

### Low-level features
Xtrain, Ytrain = load_file("Data/train_" + answers["low-level_features"] + ".npz")
print(Xtrain.shape, Ytrain.shape)
Xtest, Ytest = load_file("Data/test_" + answers["low-level_features"] + ".npz")
print(Xtest.shape, Ytest.shape)

### normalization
if answers1["normalization"] == 'yes':
   Xtrain = l2_normalization(Xtrain)
   Xtest = l2_normalization(Xtest)

nclasses = Ytrain.max() + 1

mlp = pvml.MLP([Xtrain.shape[1], nclasses])
epochs = []
batch_size = 50
lr = 0.0001

train_accs = []
test_accs = []

print('STARTING TRAINING')
plt.ion()
for epoch in range(1000):
    steps = Xtrain.shape[0] // batch_size
    mlp.train(Xtrain, Ytrain, lr=lr, batch=batch_size, steps=steps)
    if epoch % 10 == 0:
       # predictions, probs = mlp.inference(X)
        train_acc = accuracy(mlp, Xtrain, Ytrain)
        test_acc = accuracy(mlp, Xtest, Ytest)
        print(epoch, train_acc, test_acc)
        
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        epochs.append(epoch)
        
        plt.clf()
        plt.plot(epochs, train_accs, color = 'r')
        plt.plot(epochs, test_accs, color ='blue')
        if answers1["normalization"] == 'yes':
            plt.title(answers["low-level_features"] + ' with L_2 normalization')
        else:
            plt.title(answers["low-level_features"])
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy [%]")
        plt.legend(["train", "test"])
        plt.pause(0.01)

if answers1["normalization"] == 'yes':
    plt.savefig("figures/mlp_" + answers["low-level_features"] + "_L2.png")
    mlp.save("MLP/mlp_" + answers["low-level_features"] + "_L2.npz")
else:
    plt.savefig("figures/mlp_" + answers["low-level_features"] + ".png")
    mlp.save("MLP/mlp_" + answers["low-level_features"] + ".npz")
    
plt.ioff()
print('TRAINING FINISHED')
plt.show()

