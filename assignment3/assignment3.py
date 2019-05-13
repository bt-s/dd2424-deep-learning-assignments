#!/usr/bin/python3

"""assignment3regular.py: Implementation of a k-layer neural network with batch
normalization and a cross-entropy loss, applied to the CIFAR10 dataset.

For the 2019 DD2424 Deep Learning in Data Science course at KTH Royal Institute
of Technology"""

__author__ = "Bas Straathof"

import pickle
import matplotlib.pyplot as plt
import numpy as np
import unittest
import statistics
import re
import csv
from collections import OrderedDict


def load_batch(filename):
    """Loads a data batch

    Args:
        filename (str): filename of the data batch to be loaded

    Returns:
        X (np.ndarray): data matrix (D, N)
        Y (np.ndarray): one hot encoding of the labels (C, N)
        y (np.ndarray): vector containing the labels (N,)
    """
    with open(filename, 'rb') as f:
        data_dict = pickle.load(f, encoding='bytes')

        X = normalize_data((data_dict[b"data"]).T)
        y = np.array(data_dict[b"labels"])
        Y = (np.eye(10)[y]).T

    return X, Y, y


def normalize_data(X):
    """Normalize the data by substracting the mean and dividing by
    the standard deviation

    Args:
        X (np.ndarray): data matrix (D, N)

    Returns:
        X (np.ndarray): data matrix (D, N)
    """
    X_row_stdev = np.std(X, axis=1, keepdims=True)
    X_row_mean  = np.mean(X, axis=1, keepdims=True)
    X = (X - X_row_mean) / X_row_stdev

    return X


def unpickle(filename):
    """Unpickle a file

    Args:
        filename (str): filename of the data batch to be loaded

    Returns:
        file_dict (dict): a dictionary containing the contents of the file
    """
    with open(filename, 'rb') as f:
        file_dict = pickle.load(f, encoding='bytes')

    return file_dict


def train_on_one_data_batch():
    """Create training, validation and test sets where the training data
    consists of 10,000 images.

    Returns:
        data (dict):   all the separate data sets
        labels (list): correct image labels
    """
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

    return data, labels


def train_on_all_data_batches(val):
    """Create training, validation and test sets where the training data
    consists of all images minus a specified amount of validation images.

    Args:
        val (int): the amount of training data reserved for validation

    Returns:
        data (dict):   all the separate data sets
        labels (list): correct image labels
    """
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
    X_val = X_train[:, -val:]
    Y_val = Y_train[:, -val:]
    y_val = y_train[-val:]
    X_train = X_train[:, :-val]
    Y_train = Y_train[:, :-val]
    y_train = y_train[:-val]

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

    return data, labels


def make_layers(shapes, activations):
    """Create the layers of the network

    Args:
        shapes      (list): the shapes per layer as tuples
        activations (list): the activation functions per layer as strings

    Returns:
        layers (OrderedDict): specifies the shape and activation function of
        each layer
    """
    if len(shapes) != len(activations):
        raise RuntimeError('The size of shapes should equal the size of activations.')

    layers = OrderedDict([])

    for i, (shape, activation) in enumerate(zip(shapes, activations)):
        layers["layer%s" % i] = {"shape": shape, "activation": activation}

    return layers


class Classifier():
    """The mini-batch gradient descent classifier"""
    def __init__(self, data, labels, layers, alpha=0.8, batch_norm=False):
        """Constructor of the classifier class
        Args:
            data (dict): A dictionary containing the:
                - training, validation and testing:
                    - examples matrix
                    - one-hot-encoded labels matrix
                    - labels vector
            labels (list): label names (strings)
        """
        for k, v in data.items():
            setattr(self, k, v)

        self.labels     = labels
        self.layers     = layers
        self.k          = len(layers) - 1
        self.alpha      = alpha
        self.batch_norm = batch_norm

        self.activation_funcs = {'relu': self._relu, 'softmax': self._softmax}

        self.W, self.b, self.gamma, self.beta, self.mu_av, self.var_av, \
                self.activations = [], [], [], [], [], [], []

        for layer in layers.values():
            for k, v in layer.items():
                if k == "shape":
                    W, b, gamma, beta, mu_av, var_av = self._init_parameters(v)
                    self.W.append(W), self.b.append(b)
                    self.gamma.append(gamma), self.beta.append(beta)
                    self.mu_av.append(mu_av), self.var_av.append(var_av)
                elif k == "activation":
                    self.activations.append((v, self.activation_funcs[v]))

        if self.batch_norm:
            self.params = {"W": self.W, "b": self.b, "gamma": self.gamma,
                    "beta": self.beta}
        else:
            self.params = {"W": self.W, "b": self.b}


    @staticmethod
    def _init_parameters(d):
        """Initialize the weight and bias parameters using He initialization

        Args:
            d (tuple): shape of the layer

        Returns:
            W (np.ndarray): weight matrix of shape d
            b (np.ndarray): bias matrix of shape (d[0], 1)
        """

        #stdev = 2/np.sqrt(d[1])
        stdev = 1e-1

        W      = np.random.normal(0, stdev, size=(d[0], d[1]))
        b      = np.zeros(d[0]).reshape(d[0], 1)
        gamma  = np.ones((d[0], 1))
        beta   = np.zeros((d[0], 1))
        mu_av  = np.zeros((d[0], 1))
        var_av = np.zeros((d[0], 1))

        return W, b, gamma, beta, mu_av, var_av


    @staticmethod
    def _softmax(x):
        """Computes the softmax activation

        Args:
            x (np.ndarray): input matrix

        Returns:
            s (np.ndarray): softmax matrix
        """
        s = np.exp(x - np.max(x, axis=0)) / \
                np.exp(x - np.max(x, axis=0)).sum(axis=0)

        return s


    @staticmethod
    def _relu(x):
        """Computes the ReLU activation

        Args:
            x (np.ndarray): input matrix

        Returns:
            x (np.ndarray): relu matrix
        """
        x[x<0] = 0

        return x


    def check_gradients(self, grads_a, grads_n):
        """Maximum relative error between the analytical and numerical gradients

        Args:
            grads_a (np.ndarray): analytical gradients
            grads_n (np.ndarray): numerical gradients
        """
        num_layers = len(grads_a["W"])
        for l in range(num_layers):
            for key in grads_a:
                num = abs(grads_a[key][l].flat[:] - grads_n[key][l].flat[:])
                denom = np.asarray([max(abs(a), abs(b)) + 1e-10 for a,b in
                    zip(grads_a[key][l].flat[:], grads_n[key][l].flat[:])])
                max_rel_err = max(num / denom)
                print("The relative error for layer %d %s: %.6g" %
                        (l+1, key, max_rel_err))
        print()


    def evaluate_classifier(self, X, is_testing=False, is_training=False):
        """ Evaluate the classifier by applying ReLU to the hidden layers
        and taking the softmax of the final layer.

        Args:
            X     (np.ndarray): data matrix (D, N_batch)
            is_testing  (bool): flag to indicate the testing phase
            is_training (bool): flag to indicate the training phase

        Returns:
            H          (np.ndarray): intermediaary activaction values
            P          (np.ndarray): final probabilities
            S          (np.ndarray): linear transformations
            S_hat      (np.ndarray): normalized linear transformations
            means      (np.ndarray): mean vectors
            variancess (np.ndarray): variance vectors
        """
        N = X.shape[1]
        s = np.copy(X)

        if self.batch_norm:
            S, S_hat, means, variances, H = [], [], [], [], []

            for i, (W, b, gamma, beta, mu_av, var_av, activation) in enumerate(
                    zip(self.W, self.b, self.gamma, self.beta, self.mu_av,
                        self.var_av, self.activations)):

                H.append(s)
                s = W@s + b

                if i < self.k:
                    S.append(s)
                    if is_testing:
                        s = (s - mu_av) / np.sqrt(var_av + \
                                np.finfo(np.float64).eps)

                    else:
                        mu = np.mean(s, axis=1, keepdims=True)
                        means.append(mu)
                        var = np.var(s, axis=1, keepdims=True) * (N-1)/N
                        variances.append(var)

                        if is_training:
                            self.mu_av[i]  = self.alpha * mu_av + \
                                    (1-self.alpha) * mu
                            self.var_av[i] = self.alpha * var_av + \
                                    (1-self.alpha) * var

                        s = (s - mu) / np.sqrt(var + np.finfo(np.float64).eps)

                    S_hat.append(s)
                    s = activation[1](np.multiply(gamma, s) + beta)

                else:
                    P = activation[1](s)

            return H, P, S, S_hat, means, variances

        else:
            H = []
            for W, b, activation in zip(self.W, self.b, self.activations):
                if activation[0] == "relu":
                    s = activation[1](W@s + b)
                    H.append(s)
                if activation[0] == "softmax":
                    P = activation[1](W@s + b)

            return H, P


    def compute_cost(self, X, Y, labda, is_testing=False):
        """Computes the cost of the classifier using the cross-entropy loss

        Args:
            X     (np.ndarray): data matrix (D, N)
            Y     (np.ndarray): one-hot encoding labels matrix (C, N)
            labda (np.float64): regularization term
            is_testing  (bool): flag to indicate the testing phase

        Returns:
            lost (np.float64): current loss of the model
            cost (np.float64): current cost of the model
        """
        N = X.shape[1]

        if self.batch_norm:
            _, P, _, _, _, _ = self.evaluate_classifier(X, is_testing=is_testing)
        else:
            _, P = self.evaluate_classifier(X)

        loss = np.float64(1/N) * - np.sum(Y*np.log(P))

        squaredWeights = 0
        for W in self.W:
            squaredWeights += (np.sum(np.square(W)))
        cost = loss + labda * squaredWeights

        return loss, cost


    def compute_accuracy(self, X, y, is_testing=False):
        """ Computes the accuracy of the classifier
        Args:
            X    (np.ndarray): data matrix (D, N)
            y    (np.ndarray): labels vector (N)
            is_testing (bool): flag to indicate the testing phase

        Returns:
            acc (float): the accuracy of the model
        """
        if self.batch_norm:
            argMaxP = np.argmax(self.evaluate_classifier(
                X, is_testing=is_testing)[1], axis=0)
        else:
            argMaxP = np.argmax(self.evaluate_classifier(X)[1], axis=0)

        acc = argMaxP.T[argMaxP == np.asarray(y)].shape[0] / X.shape[1]

        return acc


    def compute_gradients(self, X_batch, Y_batch, labda):
        """ Analytically computes the gradients of the weight and bias parameters

        Args:
            X_batch (np.ndarray): data batch matrix (D, N)
            Y_batch (np.ndarray): one-hot-encoding labels batch vector (C, N)
            labda        (float): regularization term

        Returns:
            grads (dict): the updated analytical gradients
        """
        N = X_batch.shape[1]

        if self.batch_norm:
            grads = {"W": [], "b": [], "gamma": [], "beta": []}

            for key in self.params:
                for par in self.params[key]:
                    grads[key].append(np.zeros_like(par))

            # Forward pass
            H_batch, P_batch, S_batch, S_hat_batch, means_batch, vars_batch = \
                    self.evaluate_classifier(X_batch, is_training=True)

            # Backward pass
            G_batch = - (Y_batch - P_batch)

            grads["W"][self.k] = 1/N * G_batch@H_batch[self.k].T + \
                    2 * labda * self.W[self.k]
            grads["b"][self.k] = np.reshape(1/N * G_batch@np.ones(N),
                    (grads["b"][self.k].shape[0], 1))

            G_batch = self.W[self.k].T@G_batch
            H_batch[self.k][H_batch[self.k] <= 0] = 0
            G_batch = np.multiply(G_batch, H_batch[self.k] > 0)

            # for l = k-1, k-2, ..., 1
            for l in range(self.k - 1, -1, -1):
                grads["gamma"][l] = np.reshape(1/N * np.multiply(G_batch,
                    S_hat_batch[l])@np.ones(N), (grads["gamma"][l].shape[0], 1))
                grads["beta"][l]  = np.reshape(1/N * G_batch@np.ones(N),
                        (grads["beta"][l].shape[0], 1))

                G_batch = np.multiply(G_batch, self.gamma[l])

                G_batch = self.batch_norm_back_pass(G_batch, S_batch[l],
                        means_batch[l], vars_batch[l])

                grads["W"][l] = 1/N * G_batch@H_batch[l].T + 2 * labda * self.W[l]

                grads["b"][l] = np.reshape(1/N * G_batch@np.ones(N),
                                        (grads["b"][l].shape[0], 1))
                if l > 0:
                    G_batch = self.W[l].T@G_batch
                    H_batch[l][H_batch[l] <= 0] = 0
                    G_batch = np.multiply(G_batch, H_batch[l] > 0)

        else:
            grads = {"W": [], "b": []}
            for W, b in zip(self.W, self.b):
                grads["W"].append(np.zeros_like(W))
                grads["b"].append(np.zeros_like(b))

            # Forward pass
            H_batch, P_batch = self.evaluate_classifier(X_batch)

            # Backward pass
            G_batch = - (Y_batch - P_batch)

            # for l = k, k-1, ..., 2
            for l in range(len(self.layers) - 1, 0, -1):
                grads["W"][l] = 1/N * G_batch@H_batch[l-1].T + 2 * labda * self.W[l]
                grads["b"][l] = np.reshape(1/N * G_batch@np.ones(N),
                        (grads["b"][l].shape[0], 1))

                G_batch = self.W[l].T@G_batch
                H_batch[l-1][H_batch[l-1] <= 0] = 0
                G_batch = np.multiply(G_batch, H_batch[l-1] > 0)

            grads["W"][0] = 1/N * G_batch@X_batch.T + labda * self.W[0]
            grads["b"][0] = np.reshape(1/N * G_batch@np.ones(N), self.b[0].shape)

        return grads


    def batch_norm_back_pass(self, G_batch, S_batch, mean_batch, var_batch):
        """Computation of the batch normalization back pass

        Args:
            G_batch    (np.ndarray): gradients of the batch
            S_batch    (np.ndarray): linear transformations of the batch
            mean_batch (np.ndarray): mean vectors of the batch
            var_bath   (np.ndarray): variance vectors of the batch

        Returns:
            G_batch (np.ndarrat): batch normalized gradients
        """
        N = G_batch.shape[1]
        sigma1 = np.power(var_batch + np.finfo(np.float64).eps, -0.5)
        sigma2 = np.power(var_batch + np.finfo(np.float64).eps, -1.5)

        G1 = np.multiply(G_batch, sigma1)
        G2 = np.multiply(G_batch, sigma2)

        D = S_batch - mean_batch

        c = np.sum(np.multiply(G2, D), axis=1, keepdims=True)

        G_batch = G1 - 1/N * np.sum(G1, axis=1, keepdims=True) - \
                1/N * np.multiply(D, c)

        return G_batch


    def compute_gradients_num(self, X_batch, Y_batch, size=2,
            labda=np.float64(0), h=np.float64(1e-7)):
        """Numerically computes the gradients of the weight and bias parameters

        Args:
            X_batch (np.ndarray): data batch matrix (D, N)
            Y_batch (np.ndarray): one-hot-encoding labels batch vector (C, N)
            W       (np.ndarray): the weight matrix
            b       (np.ndarray): the bias matrix
            labda        (np.float64): penalty term
            h            (np.float64): marginal offset

        Returns:
            grads  (dict): the numerically computed gradients
        """
        if self.batch_norm:
            grads = {"W": [], "b": [], "gamma": [], "beta": []}
        else:
            grads = {"W": [], "b": []}

        for j in range(len(self.b)):
            for key in self.params:
                grads[key].append(np.zeros(self.params[key][j].shape))
                for i in range(len(self.params[key][j].flatten())):
                    old_par = self.params[key][j].flat[i]
                    self.params[key][j].flat[i] = old_par + h
                    _, c2 = self.compute_cost(X_batch, Y_batch, labda)
                    self.params[key][j].flat[i] = old_par - h
                    _, c3 = self.compute_cost(X_batch, Y_batch, labda)
                    self.params[key][j].flat[i] = old_par
                    grads[key][j].flat[i] = (c2-c3) / (2*h)

        return grads


    def plot(self, n_epochs, i_train, i_val, title, y_label, y_max=4):
       """Plots some function on the training and validation sets

        Args:
            n_epochs       (int): number of training epochs
            i_train (np.ndarray): input to plot per epoch on the training set
            i_val   (np.ndarray): input to plot per epoch on the validation set
            title          (str): plot title
            y_label        (str): y-axis label
            y_max        (float): max y-limit of plot
       """
       epochs = np.arange(n_epochs)

       fig, ax = plt.subplots(figsize=(10, 8))
       ax.plot(epochs, i_train, label="Training set")
       ax.plot(epochs, i_val, label="Validation set")
       ax.legend()
       ax.set(xlabel='Number of epochs', ylabel=y_label)
       ax.set_ylim([0, y_max])
       ax.grid()

       plt.savefig("plots/" + title + ".png", bbox_inches="tight")


    def mini_batch_gd(self, X, Y, labda=0, batch_s=100, eta_min=1e-5,
            eta_max=1e-1, n_s = 800, n_epochs=40, plot_id="", verbose=True,
            plot=False):
        """Trains the model using mini-batch gradient descent

        Args:
            X   (np.ndarray): data matrix (D, N)
            Y   (np.ndarray): one-hot-encoding labels matrix (C, N)
            labda    (float): regularization term
            batch_s    (int): batche size
            eta_min  (float): min learning rate
            eta_max  (float): max learning rate
            n_s        (int): number of learning rate update steps
            n_epochs   (int): number of training epochs
            plot_id    (str): title for cost plots
            verbose   (bool): boolean for deciding on textual output
            plot      (bool): boolean for deciding whether to plot costs

        Returns:
            acc_train (float): the accuracy on the training set
            acc_val   (float): the accuracy on the validation set
            acc_test  (float): the accuracy on the testing set
        """
        if plot:
            costs_train, loss_train, acc_train, costs_val, loss_val, acc_val = \
                    np.zeros(n_epochs), np.zeros(n_epochs), np.zeros(n_epochs), \
                    np.zeros(n_epochs), np.zeros(n_epochs), np.zeros(n_epochs)

        n_batch = int(np.floor(X.shape[1] / batch_s))
        eta = eta_min
        t = 0
        for n in range(n_epochs):
            for j in range(n_batch):
                N = int(X.shape[1] / n_batch)
                j_start = (j) * N
                j_end = (j+1) * N

                X_batch = X[:, j_start:j_end]
                Y_batch = Y[:, j_start:j_end]

                grads = self.compute_gradients(X_batch, Y_batch, labda)

                for key in self.params:
                    for par, grad in zip(self.params[key], grads[key]):
                        par -= eta * grad

                if t <= n_s:
                    eta = eta_min + t/n_s * (eta_max - eta_min)

                elif t <= 2*n_s:
                    eta = eta_max - (t - n_s)/n_s * (eta_max - eta_min)

                t = (t+1) % (2*n_s)

            if plot:
                loss_train[n], costs_train[n] = self.compute_cost(X, Y, labda)
                loss_val[n], costs_val[n] = self.compute_cost(
                        self.X_val, self.Y_val, labda)

                acc_train[n] = self.compute_accuracy(self.X_train, self.y_train)
                print("The accuracy after epoch %s is: %s" %
                        (str(n), str(acc_train[n])))
                acc_val[n] = self.compute_accuracy(self.X_val, self.y_val)


        if plot:
            self.plot(n_epochs, costs_train, costs_val, plot_id + "_cost_plot",
                    y_label="Cost", y_max=4)
            self.plot(n_epochs, loss_train, loss_val, plot_id + "_loss_plot",
                    y_label="Loss", y_max=2.5)
            self.plot(n_epochs, acc_train, acc_val, plot_id + "_acc_plot",
                    y_label="Accuracy", y_max=0.8)

        acc_train = self.compute_accuracy(self.X_train, self.y_train)
        acc_val   = self.compute_accuracy(self.X_val, self.y_val)
        acc_test  = self.compute_accuracy(self.X_test, self.y_test,
                is_testing=True)

        if verbose:
            print("The accuracy on the training set is: "   + str(acc_train))
            print("The accuracy on the validation set is: " + str(acc_val))
            print("The accuracy on the testing set is: "    + str(acc_test))

        return acc_train, acc_val, acc_test


### UNIT TESTS
class TestMethods(unittest.TestCase):
    def test_sizes(self):
        data, labels = train_on_one_data_batch()

        layers = make_layers(
                shapes=[(50, 3072), (10, 50)],
                activations=["relu", "softmax"])

        clf = Classifier(data, labels, layers)

        grads = clf.compute_gradients(clf.X_train, clf.Y_train, labda=0)

        self.assertEqual(clf.X_train.shape, (3072, 10000))
        self.assertEqual(clf.Y_train.shape, (10, 10000))
        self.assertEqual(np.shape(clf.y_train), (10000,))
        self.assertEqual(clf.W[0].shape, (50, 3072))
        self.assertEqual(clf.b[0].shape, (50, 1))
        self.assertEqual(clf.W[1].shape, (10, 50))
        self.assertEqual(clf.b[1].shape, (10, 1))
        self.assertEqual(clf.evaluate_classifier(clf.X_train)[0][0].shape,
                        (50, 10000))
        self.assertEqual(clf.evaluate_classifier(clf.X_train)[1].shape,
                        (10, 10000))
        self.assertAlmostEqual(sum(sum(clf.evaluate_classifier(
            clf.X_train[:, 0].reshape((clf.X_train.shape[0], 1)))[1])), 1)
        self.assertIsInstance(clf.compute_cost(clf.X_train, clf.Y_train,
            labda=0)[1], float)
        self.assertEqual(grads["W"][0].shape, clf.W[0].shape)
        self.assertEqual(grads["b"][0].shape, clf.b[0].shape)
        self.assertEqual(grads["W"][1].shape, clf.W[1].shape)
        self.assertEqual(grads["b"][1].shape, clf.b[1].shape)


    def test_array_equality_nobn(self):
        data, labels = train_on_one_data_batch()

        layers = make_layers(
                shapes=[(50, 30), (50, 50), (50, 50), (10, 50)],
                activations=["relu", "relu", "relu", "softmax"])

        clf = Classifier(data, labels, layers)

        grads_ana = clf.compute_gradients(
                clf.X_train[:30, :5],
                clf.Y_train[:30, :5],
                labda=0)

        grads_num = clf.compute_gradients_num(
                clf.X_train[:30, :5],
                clf.Y_train[:30, :5],
                labda=0)

        clf.check_gradients(grads_ana, grads_num)


    def test_array_equality_bn(self):
        data, labels = train_on_one_data_batch()
        layers = make_layers(
                shapes=[(50, 30), (50, 50), (10, 50)],
                activations=["relu", "relu", "softmax"])

        clf = Classifier(data, labels, layers, batch_norm=True)

        grads_ana = clf.compute_gradients(
                clf.X_train[:30, :5],
                clf.Y_train[:30, :5],
                labda=0)

        grads_num = clf.compute_gradients_num(
                clf.X_train[:30, :5],
                clf.Y_train[:30, :5],
                labda=0)

        clf.check_gradients(grads_ana, grads_num)


if __name__ == '__main__':
    def train_model(layers, batch_norm, labda=0.005):
        """Train a model with three layers"""
        data, labels = train_on_all_data_batches(val=5000)
        layers = make_layers(
                shapes=[(50, 3072), (50, 50), (10, 50)],
                activations=["relu", "relu", "softmax"])

        acc_train_set = []
        acc_val_set = []
        acc_test_set = []
        for j in range(5):
            clf = Classifier(data, labels, layers, batch_norm=batch_norm)
            acc_train, acc_val, acc_test = clf.mini_batch_gd(
                    data['X_train'], data['Y_train'], labda=labda,
                    batch_s=100, eta_min=1e-5, eta_max=1e-1, n_s=2250,
                    n_epochs=20, verbose=False)

            acc_train_set.append(acc_train)
            acc_val_set.append(acc_val)
            acc_test_set.append(acc_test)

        print("Train mean acc:" + str(statistics.mean(acc_train_set)))
        print("Val mean acc:" + str(statistics.mean(acc_val_set)))
        print("Test mean acc:" + str(statistics.mean(acc_test_set)))
        print("Train stdev acc:" + str(statistics.stdev(acc_train_set)))
        print("Val stdev acc:" + str(statistics.stdev(acc_val_set)))
        print("Test stdev acc:" + str(statistics.stdev(acc_test_set)))


    three_layers = make_layers(
            shapes=[(50, 3072), (50, 50), (10, 50)],
            activations=["relu", "relu", "softmax"])
    nine_layers = make_layers(
            shapes=[(50, 3072), (30, 50), (20, 30), (20, 20), (10, 20),
                (10, 10), (10, 10), (10, 10), (10, 10)],
            activations=["relu", "relu", "relu", "relu", "relu", "relu",
                "relu", "relu", "softmax"])

    train_model(three_layers, batch_norm=False) # 3-layer model w\o batch norm
    train_model(three_layers, batch_norm=True) # 3-layer model w\ batch norm
    train_model(nine_layers, batch_norm=False) # 9-layer model w\o batch norm
    train_model(nine_layers, batch_norm=True) # 9-layer model w\ batch norm
    train_model(nine_layers, batch_norm=True, labda=0.0072) # best 9-layer model

    ## UNIT TESTING
    np.random.seed(0)
    unittest.main()

