#!/usr/bin/python3

"""assignment1bonus.py: Implementation of a single layer neural network,
with cross-entropy or multi-class SVM loss, applied to the CIFAR10 dataset.

For the DD2424 Deep Learning in Data Science course at KTH Royal Institute of
Technology"""

__author__ = "Bas Straathof"

import pickle
import matplotlib.pyplot as plt
import numpy as np
import unittest
import statistics
import re


def load_batch(filename):
    """Loads a data batch

    Args:
        filename (str): filename of the data batch to be loaded

    Returns:
        X (np.ndarray): data matrix (D, N)
        Y (np.ndarray): one hot encoding of the labels (N,)
        y (np.ndarray): vector containing the labels (C, N)
    """
    with open(filename, 'rb') as f:
        dataDict = pickle.load(f, encoding='bytes')

        X = (dataDict[b"data"] / 255).T
        y = dataDict[b"labels"]
        Y = (np.eye(10)[y]).T

    return X, Y, y


def unpickle(filename):
    """Unpickle a file

    Args:
        filename (str): filename of the data batch to be loaded

    Returns:
        file_dict (dict): contents of the file
    """
    with open(filename, 'rb') as f:
        file_dict = pickle.load(f, encoding='bytes')

    return file_dict



class Classifier():
    """The mini-batch gradient descent classifier"""
    def __init__(self, data, labels, mcsvm=False, xavier=False, lr_decay=False,
            W=None, b=None):
        """Class constructor

        Args:
            data (dict):
                - training, validation and testing:
                    - examples matrix
                    - one-hot-encoded labels matrix
                    - labels vector
            labels   (list): label names (strings)
            mcsvm    (bool): decides whether the Multi-Class SVM loss function
                             should be applied
            xavier   (bool): decides whether Xavier initialization should be
                             applied
            lr_decay (bool): decides whether learning rate decay should be
                             applied
            W  (np.ndarray): weight matrix
            b  (np.ndarray): bias matrix
        """
        for k, v in data.items():
            setattr(self, k, v)

        self.labels = labels

        self.mcsvm = mcsvm

        self.lr_decay = lr_decay

        if xavier:
            self.W = W if W != None else np.random.normal(
                    0,
                    1/np.sqrt(len(self.X_train)),
                    (len(self.labels), len(self.X_train)))
        else:
            self.W = W if W != None else np.random.normal(
                    0, 0.01, (len(self.labels), self.X_train.shape[0]))

        self.b = b if b != None else np.random.normal(
                0, 0.01, (len(self.labels), 1))


    def evaluate_classifier(self, X):
        """ Compute the softmax

        Args:
            X (np.ndarray): data matrix (D, N)

        Returns a stable softmax matrix
        """
        s = self.W@X + self.b
        p = np.exp(s - np.max(s, axis=0)) / \
                np.exp(s - np.max(s, axis=0)).sum(axis=0)

        return p


    def ComputeCost(self, X, Y, labda):
        """ Computes the cost using the cross-entropy or multi-class SVM loss

        Args:
            X (np.ndarray): data matrix (D, N)
            Y (np.ndarray): one-hot encoding labels matrix (C, N)
            labda  (float): regularization term

        Returns:
            cost (float): the cross-entropy loss
        """
        N = X.shape[1]

        if self.mcsvm:
            s = self.evaluate_classifier(X)
            sc = s.T[np.arange(s.shape[1]), np.argmax(Y, axis=0)].T

            margins = np.maximum(0, s - np.asarray(sc) + 1)
            margins.T[np.arange(N), np.argmax(Y, axis=0)] = 0

            mcsvm_loss = Y.shape[0] * np.mean(np.sum(margins, axis=1))

            cost = 1/N * mcsvm_loss + 0.5 * labda * np.sum(self.W**2)

            return cost, margins

        else:
            P = self.evaluate_classifier(X)
            cost = 1/N * - np.sum(Y*np.log(P)) + labda * np.sum(self.W**2)

            return cost, None


    def compute_accuracy(self, X, y):
        """ Computes the accuracy of the classifier

        Args:
            X (np.ndarray): data matrix (D, N)
            y (np.ndarray): labels vector (N)

        Returns:
            acc (float): the accuracy of the classifier
        """
        argMaxP = np.argmax(self.evaluate_classifier(X), axis=0)
        count = argMaxP.T[argMaxP == np.asarray(y)].shape[0]

        acc = count / X.shape[1]


    def compute_gradients(self, X_batch, Y_batch, labda):
        """ Analytically omputes the gradients of the weight and bias parameters

        Args:
            X_batch (np.ndarray): data batch matrix (D, N)
            Y_batch (np.ndarray): one-hot-encoding labels batch vector (C, N)
            labda        (float): regularization term

        Returns:
            grad_W (np.ndarray): the gradient of the weight parameter
            grad_b (np.ndarray): the gradient of the bias parameter
        """
        N = X_batch.shape[1]

        if self.mcsvm:
            _, margins = self.ComputeCost(X_batch, Y_batch, labda)

            binaries = margins
            binaries[margins > 0] = 1
            binaries_row_sum = np.sum(binaries, axis=0)

            binaries.T[np.arange(N), np.argmax(Y_batch, axis=0)] = \
                    -binaries_row_sum.T

            grad_W = np.dot(binaries, X_batch.T) / N + labda * self.W

            grad_b = np.reshape(np.sum(binaries, axis=1) /
                    binaries.shape[1], self.b.shape)

        else:
            P = self.evaluate_classifier(X_batch)
            G = - (Y_batch - P)

            grad_W = 1 / N * G@X_batch.T + 2 * labda * self.W
            grad_b = np.reshape(1 / N * G@np.ones(N), (Y_batch.shape[0], 1))

        return grad_W, grad_b


    def compute_gradients_num(self, X_batch, Y_batch, labda=0, h=1e-6):
        """Numerically computes the gradients of the weight and bias parameters

        Args:
            X_batch (np.ndarray): data batch matrix (D, N)
            Y_batch (np.ndarray): one-hot-encoding labels batch vector (C, N)
            labda        (float): regularization term
            h            (float): marginal offset

        Returns:
            grad_W (np.ndarray): the gradient of the weight parameter
            grad_b (np.ndarray): the gradient of the bias parameter
        """
        grad_W = np.zeros(self.W.shape)
        grad_b = np.zeros(self.b.shape)

        b_try = np.copy(self.b)
        for i in range(len(self.b)):
            self.b = b_try
            self.b[i] = self.b[i] + h
            c2, _ = self.compute_cost(X_batch, Y_batch, labda)
            self.b[i] = self.b[i] - 2*h
            c3, _ = self.compute_cost(X_batch, Y_batch, labda)
            grad_b[i] = (c2-c3) / (2*h)

        W_try = np.copy(self.W)
        for i in np.ndindex(self.W.shape):
            self.W = W_try
            self.W[i] = self.W[i] + h
            c2, _ = self.compute_cost(X_batch, Y_batch, labda)
            self.W[i] = self.W[i] - 2*h
            c3, _ = self.compute_cost(X_batch, Y_batch, labda)
            grad_W[i] = (c2-c3) / (2*h)

        return grad_W, grad_b


    def plot_performance(self, n_epochs, costs_training, costs_val, title):
       """Plots performance curves

        Args:
            n_epochs       (int): number of training epochs
            cost_training (list): cost per epoch on the training set
            cost_val      (list): cost per epoch on the validation set
            title          (str): plot title
       """
       epochs = np.arange(n_epochs)

       fig, ax = plt.subplots(figsize=(10, 8))
       ax.plot(epochs, costs_training, label="Training set")
       ax.plot(epochs, costs_val, label="Validation set")
       ax.legend()
       ax.set(xlabel='Number of epochs', ylabel='Cost')
       ax.grid()

       plt.savefig("plots/" + title + ".png", bbox_inches="tight")


    def mini_batch_gd(self, X, Y, labda=0, n_batch=100, eta=0.01, n_epochs=40,
            verbose=True, plot_performance=False, title=None):
        """Trains the model using mini-batch gradient descent

        Args:
            X    (np.ndarray): data matrix (D, N)
            Y    (np.ndarray): one-hot-encoding labels matrix (C, N)
            labda     (float): regularization term
            n_batch     (int): number of batches
            eta       (float): learning rate
            n_epochs    (int): number of training epochs
            verbose    (bool): decides on textual output
            plot_performance (bool): decides whether to plot costs
            title       (str): title for cost plots

        Returns:
            acc_train (float): the accuracy on the training set
            acc_val   (float): the accuracy on the validation set
            acc_test  (float): the accuracy on the testing set
        """
        if plot_performance:
            costs_training = np.zeros(n_epochs)
            costs_val = np.zeros(n_epochs)

        for n in range(n_epochs):
            for j in range(n_batch):
                N = int(X.shape[1] / n_batch)
                j_start = (j) * N
                j_end = (j+1) * N

                X_batch = X[:, j_start:j_end]
                Y_batch = Y[:, j_start:j_end]

                grad_W, grad_b = self.compute_gradients(X_batch, Y_batch, labda)
                self.W -= eta * grad_W
                self.b -= eta * grad_b

            if plot_performance:
                costs_training[n], _ = self.ComputeCost(X, Y, labda)
                costs_val[n], _ = self.ComputeCost(self.X_val, self.Y_val, labda)

            if self.lr_decay:
                eta = eta * 0.9

        if plot_performance:
            self.plot_performance(n_epochs, costs_training, costs_val, title)

        acc_train = self.compute_accuracy(self.X_train, self.y_train)
        acc_val = self.compute_accuracy(self.X_val, self.y_val)
        acc_test = self.compute_accuracy(self.X_test, self.y_test)

        if verbose:
            print("The accuracy on the training set is: " + str(acc_train))
            print("The accuracy on the validation set is: " + str(acc_val))
            print("The accuracy on the testing set is: " + str(acc_test))

        return acc_train, acc_val, acc_test


### UNIT TESTS
class CustomAssertions:
    def assert_array_almost_equal(self, arr1, arr2, dec=4):
        """ Checks whether two numpy arrays are almost equal

        Args:
            arr1 (np.ndarray): the first array for comparison
            arr2 (np.ndarray): the second array for comparison
            dec         (int): denotes the number of decimals
        """
        np.testing.assert_almost_equal(arr1, arr2, decimal=dec)


class TestMethods(unittest.TestCase, CustomAssertions):
    X_train, Y_train, y_train = \
        load_batch("datasets/cifar-10-batches-py/data_batch_1")
    X_val, Y_val, y_val = \
        load_batch("datasets/cifar-10-batches-py/data_batch_2")
    X_test, Y_test, y_test = \
        load_batch("datasets/cifar-10-batches-py/test_batch")

    labels = unpickle(
        'datasets/cifar-10-batches-py/batches.meta')[ b'label_names']

    data = {
        'X_train': X_train,
        'Y_train': Y_train,
        'y_train': y_train,
        'X_val': X_val,
        'Y_val': Y_val,
        'y_val': y_val,
        'X_test': X_test,
        'Y_test': Y_test,
        'y_test': y_test
    }

    clf = Classifier(data, labels, mcsvm=False)

    W_test = np.random.normal(0, 0.01, (10, 3072))
    b_test = np.random.normal(0, 0.01, (10, 1))

    def test_array_equality_mcsvm(self):
        mcsvmClf = Classifier(data, labels, mcsvm=True)

        grad_W_test_ana_mcsvm, grad_b_test_ana_mcsvm = mcsvmClf.compute_gradients(
            self.clf.X_train[:, :2], self.clf.Y_train[:, :2], labda=0)
        grad_W_test_num_mcsvm, grad_b_test_num_mcsvm = mcsvmClf.compute_gradients_num(
            self.clf.X_train[:, :2], self.clf.Y_train[:, :2], labda=0)

        self.assert_array_almost_equal(grad_W_test_ana_mcsvm, grad_W_test_num_mcsvm)
        self.assert_array_almost_equal(grad_b_test_ana_mcsvm, grad_b_test_num_mcsvm)


if __name__ == '__main__':
    lambdas = [0, 0, .1, 1]
    etas = [.1, .01, .01, .01]

    X_test, Y_test, y_test = \
        load_batch("datasets/cifar-10-batches-py/test_batch")

    labels = unpickle(
        'datasets/cifar-10-batches-py/batches.meta')[ b'label_names']


    ### BONUS ASSIGNMENT 1
    # (A) Train on all data
    X_train1, Y_train1, y_train1 = \
        load_batch("datasets/cifar-10-batches-py/data_batch_1")
    X_train2, Y_train2, y_train2 = \
        load_batch("datasets/cifar-10-batches-py/data_batch_2")
    X_train3, Y_train3, y_train3 = \
        load_batch("datasets/cifar-10-batches-py/data_batch_3")
    X_train4, Y_train4, y_train4 = \
        load_batch("datasets/cifar-10-batches-py/data_batch_4")
    X_train5, Y_train5, y_train5 = \
        load_batch("datasets/cifar-10-batches-py/data_batch_5")

    X_train = np.concatenate((X_train1, X_train2, X_train3, X_train4, X_train5),
            axis=1)
    Y_train = np.concatenate((Y_train1, Y_train2, Y_train3, Y_train4, Y_train5),
            axis=1)
    y_train = np.concatenate((y_train1, y_train2, y_train3, y_train4, y_train5))
    X_val = X_train[:, -1000:]
    Y_val = Y_train[:, -1000:]
    y_val = y_train[-1000:]
    X_train = X_train[:, :-1000]
    Y_train = Y_train[:, :-1000]
    y_train = y_train[:-1000]

    data = {
        'X_train': X_train,
        'Y_train': Y_train,
        'y_train': y_train,
        'X_val': X_val,
        'Y_val': Y_val,
        'y_val': y_val,
        'X_test': X_test,
        'Y_test': Y_test,
        'y_test': y_test
    }

    clf = Classifier(data, labels)

    for i in range(4):
        acc_train_set = []
        acc_val_set = []
        acc_test_set = []
        for j in range(5):
            acc_train, acc_val, acc_test = clf.mini_batch_gd(
                    X_train,
                    Y_train,
                    labda=lambdas[i],
                    eta=etas[i],
                    verbose=False)

            acc_train_set.append(acc_train)
            acc_val_set.append(acc_val)
            acc_test_set.append(acc_test)
        print("Setting " + str(i) + ":\n")
        print("Train mean acc:" + str(statistics.mean(acc_train_set)))
        print("Val mean acc:" + str(statistics.mean(acc_val_set)))
        print("Test mean acc:" + str(statistics.mean(acc_test_set)))
        print("Train stdev acc:" + str(statistics.stdev(acc_train_set)))
        print("Val stdev acc:" + str(statistics.stdev(acc_val_set)))
        print("Test stdev acc:" + str(statistics.stdev(acc_test_set)))

        np.random.seed(0)

        # Param settings 1
        clf.mini_batch_gd(X_train, Y_train, title=str(i) + "_cost_plot_full_data",
                labda=lambdas[i], eta=etas[i], plot_performance=True)


    # (D) Learning rate decay
    clf = Classifier(data, labels, lr_decay=True)

    # Param settings 2
    acc_train_set = []
    acc_val_set = []
    acc_test_set = []
    for j in range(5):
        acc_train, acc_val, acc_test = clf.mini_batch_gd(
                X_train,
                Y_train,
                labda=lambdas[1],
                eta=etas[1],
                verbose=False)

        acc_train_set.append(acc_train)
        acc_val_set.append(acc_val)
        acc_test_set.append(acc_test)
    print("Settting " + str(1) + " with learning rate decay:\n")
    print("Train mean acc:" + str(statistics.mean(acc_train_set)))
    print("Val mean acc:" + str(statistics.mean(acc_val_set)))
    print("Test mean acc:" + str(statistics.mean(acc_test_set)))
    print("Train stdev acc:" + str(statistics.stdev(acc_train_set)))
    print("Val stdev acc:" + str(statistics.stdev(acc_val_set)))
    print("Test stdev acc:" + str(statistics.stdev(acc_test_set)))

    np.random.seed(0)
    clf.mini_batch_gd(X_train, Y_train, title="2_cost_plot_full_data_lr_decay", labda=0,
            eta=.01, plot_performance=True)

    # (E) Xavier initialization
    clf = Classifier(data, labels, xavier=True)
    clf.mini_batch_gd(X_train, Y_train, title="2_cost_plot_full_data_xavier", labda=0,
            eta=.01, plot_performance=True)


    ### BONUS ASSIGNMENT 2
    np.random.seed(None)
    X_train, Y_train, y_train = \
        load_batch("datasets/cifar-10-batches-py/data_batch_1")
    X_val, Y_val, y_val = \
        load_batch("datasets/cifar-10-batches-py/data_batch_2")

    data = {
        'X_train': X_train,
        'Y_train': Y_train,
        'y_train': y_train,
        'X_val': X_val,
        'Y_val': Y_val,
        'y_val': y_val,
        'X_test': X_test,
        'Y_test': Y_test,
        'y_test': y_test
    }

    clf = Classifier(data, labels, mcsvm=True)

    for i in range(4):
        acc_train_set = []
        acc_val_set = []
        acc_test_set = []
        for j in range(10):
            acc_train, acc_val, acc_test = clf.mini_batch_gd(
                    X_train,
                    Y_train,
                    labda=lambdas[i],
                    eta=etas[i],
                    verbose=False)

            acc_train_set.append(acc_train)
            acc_val_set.append(acc_val)
            acc_test_set.append(acc_test)
        print("Settting " + str(i) + ":\n")
        print("Train mean acc:" + str(statistics.mean(acc_train_set)))
        print("Val mean acc:" + str(statistics.mean(acc_val_set)))
        print("Test mean acc:" + str(statistics.mean(acc_test_set)))
        print("Train stdev acc:" + str(statistics.stdev(acc_train_set)))
        print("Val stdev acc:" + str(statistics.stdev(acc_val_set)))
        print("Test stdev acc:" + str(statistics.stdev(acc_test_set)))

        np.random.seed(0)

        # Param settings 1
        clf.mini_batch_gd(X_train, Y_train, title=str(i) + "_cost_plot_mcsvm",
                labda=lambdas[i], eta=etas[i], plot_performance=True)

    # Unit testing
    unittest.main()

