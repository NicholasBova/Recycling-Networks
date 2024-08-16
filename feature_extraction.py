### it is good to have separate features extraction and classification
import numpy as np
import matplotlib.pyplot as plt
import inquirer
import os # we need to scan directory
import image_features

##classes = open('classes.txt').read().split()
###print(classes)

questions = [inquirer.List('low-level_features', message = 'What features do you want to use?',
                           choices = ['color_histogram', 'edge_direction', 'gray_level_cooccurrence_matrix',
                                      'rgb_cooccurrence_matrix'],),]

answers = inquirer.prompt(questions)
print('You chose ' + answers["low-level_features"])


##def load(path):
##    X, Y = np.load(path).values()
##    return X, Y

Xtrain, Ytrain = np.load("Data/train.npz").values()
print("Training: ", Xtrain.shape, Ytrain.shape)
Xtest, Ytest = np.load("Data/test.npz").values()
print("Test: ", Xtest.shape, Ytest.shape)


def process_lowlevel(X, Y):
    all_features = []
    all_labels = []
    for data in range(X.shape[0]):
        image = X[data, :]
        if answers["low-level_features"] == 'color_histogram':
               features = image_features.color_histogram(image)
        elif answers["low-level_features"] == 'edge_direction':
               features = image_features.edge_direction_histogram(image)
        elif answers["low-level_features"] == 'gray_level_cooccurrence_matrix':
               features = image_features.cooccurrence_matrix(image)
        elif answers["low-level_features"] == 'rgb_cooccurrence_matrix':
               features = image_features.rgb_cooccurrence_matrix(image)
        else:
               print('error')
        features = features.reshape(-1)
        all_features.append(features)
        all_labels.append(Y[data])
    X = np.stack(all_features, 0)
    Y = np.array(all_labels)
    return X, Y

X, Y = process_lowlevel(Xtrain, Ytrain)
print('train', X.shape, Y.shape)
np.savez_compressed("Data/train_" + answers["low-level_features"] + ".npz", X, Y)

X, Y = process_lowlevel(Xtest, Ytest)
print('test', X.shape, Y.shape)
np.savez_compressed("Data/test_" + answers["low-level_features"] + ".npz", X, Y)

