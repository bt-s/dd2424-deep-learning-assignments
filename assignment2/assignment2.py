#!/usr/bin/python3

"""assignment2regular.py: Implementation of a two layer neural network,
with cross-entropy loss, applied to the CIFAR10 dataset.

For the DD2424 Deep Learning in Data Science course at KTH Royal Institute of
Technology"""

__author__ = "Bas Straathof"

import pickle
import matplotlib.pyplot as plt
import numpy as np
import unittest
import statistics
import re
import csv


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
        dataDict = pickle.load(f, encoding='bytes')

        X = (dataDict[b"data"]).T
        X_row_mean = np.mean(X, axis=1, keepdims=True)
        X_row_stdev = np.std(X, axis=1, keepdims=True)

        X = (X - X_row_mean) / X_row_stdev

        y = np.array(dataDict[b"labels"])
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
    def __init__(self, data, labels, M=50, W1=None, b1=None, W2=None, b2=None):
        """Class constructor

        Args:
            data (dict): A dictionary containing the:
                - training, validation and testing:
                    - examples matrix
                    - one-hot-encoded labels matrix
                    - labels vector
            labels (list): label names (strings)
            M (int): number of hidden layers
            W1 (np.ndarray): weight matrix (M, D)
            b1 (np.ndarray): bias matrix (M, 1)
            W2 (np.ndarray): weight matrix (C, M)
            b2 (np.ndarray): bias matrix (C, 1)
        """
        for k, v in data.items():
            setattr(self, k, v)

        self.labels = labels
        self.M = M

        D = self.X_train.shape[0]
        N = self.X_train.shape[1]
        C = len(self.labels)

        # W1 (M, D)
        self.W1 = W1 if W1 != None else np.random.normal(
                0, 1/np.sqrt(D), (M, D))

        # b1 (M, 1)
        self.b1 = b1 if b1 != None else np.zeros((M, 1))

        # W2 (C, M)
        self.W2 = W2 if W2 != None else np.random.normal(
                0, 1/np.sqrt(M), (C, M))

        # b2 (C, 1)
        self.b2 = b2 if b2 != None else np.zeros((C, 1))


    def init_parameters(self, Ws, bs):
        """Overwrite weight and bias parameters

        Args:
            Ws (dict): weight parameters shapes {k=W_name, v=W_shape}
            bs (dict): bias parameters shapes {k=b_name, v=b_shape}
        """
        new_params = {}
        for W_name, W_shape in Ws.items():
            new_params[W_name] = np.random.normal(0, 1/np.sqrt(W_shape[1]), W_shape)

        for b_name, b_shape in bs.items():
            new_params[b_name] = np.zeros(b_shape)

        return new_params


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


    def evaluate_classifier(self, X):
        """ Evaluate the classifier by applying ReLU to the hidden layer
        and taking the softmax of the final layer.

        Args:
            X (np.ndarray): data matrix (D, N_batch)

        Returns:
            h (np.ndarray): intermediaary ReLU activaction values
            p (np.ndarray): a stable softmax matrix
        """
        h = self._relu(self.W1@X + self.b1)
        p = self._softmax(self.W2@h + self.b2)

        return h, p


    def compute_cost(self, X, Y, labda):
        """ Computes the cost of the classifier using the cross-entropy loss

        Args:
            X (np.ndarray): data matrix (D, N)
            Y (np.ndarray): one-hot encoding labels matrix (C, N)
            labda  (float): regularization term

        Returns:
            cost (float): current cost of the model
        """
        N = X.shape[1]

        _, P = self.evaluate_classifier(X)
        loss = 1/N * - np.sum(Y*np.log(P))
        cost = loss + labda * (np.sum(self.W1**2) + np.sum(self.W2**2))

        return loss, cost


    def compute_accuracy(self, X, y):
        """ Computes the accuracy of the classifier
        Args:
            X (np.ndarray): data matrix (D, N)
            y (np.ndarray): labels vector (N)

        Returns:
            acc (float): the accuracy of the model
        """
        argMaxP = np.argmax(self.evaluate_classifier(X)[1], axis=0)
        acc = argMaxP.T[argMaxP == np.asarray(y)].shape[0] / X.shape[1]

        return acc


    def compute_gradients(self, X_batch, Y_batch, labda):
        """ Analyticcaly omputes the gradients of the weight and bias parameters

        Args:
            X_batch (np.ndarray): data batch matrix (D, N)
            Y_batch (np.ndarray): one-hot-encoding labels batch vector (C, N)
            labda        (float): regularization term

        Returns:
            grad_W1 (np.ndarray): the gradient of W1
            grad_b1 (np.ndarray): the gradient of b1
            grad_W2 (np.ndarray): the gradient of W2
            grad_b2 (np.ndarray): the gradient of W2
        """
        N = X_batch.shape[1]

        # Forward pass
        H_batch, P_batch = self.evaluate_classifier(X_batch)

        # Backward pass
        G_batch = - (Y_batch - P_batch)

        grad_W2 = 1/N * G_batch@H_batch.T  + 2 * labda * self.W2
        grad_b2 = np.reshape(1/N * G_batch@np.ones(N), (Y_batch.shape[0], 1))

        G_batch = self.W2.T@G_batch
        H_batch[H_batch <= 0] = 0

        G_batch = np.multiply(G_batch, H_batch > 0)

        grad_W1 = 1/N * G_batch@X_batch.T + labda * self.W1
        grad_b1 = np.reshape(1/N * G_batch@np.ones(N), (self.M, 1))

        return grad_W1, grad_b1, grad_W2, grad_b2


    def compute_gradients_num(self, X_batch, Y_batch, labda=0, h=1e-7):
        """Numerically computes the gradients of the weight and bias parameters

        Args:
            X_batch (np.ndarray): data batch matrix (D, N)
            Y_batch (np.ndarray): one-hot-encoding labels batch vector (C, N)
            W       (np.ndarray): the weight matrix
            b       (np.ndarray): the bias matrix
            labda        (float): penalty term
            h            (float): marginal offset

        Returns:
            grad_W  (np.ndarray): the gradient of the weight parameter
            grad_b  (np.ndarray): the gradient of the bias parameter
        """
        grads = {}
        for j in range(1, 3):
            selfW = getattr(self, 'W' + str(j))
            selfB = getattr(self, 'b' + str(j))
            grads['W' + str(j)] = np.zeros(selfW.shape)
            grads['b' + str(j)] = np.zeros(selfB.shape)

            b_try = np.copy(selfB)
            for i in range(selfB.shape[0]):
                selfB = b_try[:]
                selfB[j] = selfB[j] + h
                _, c2 = self.compute_cost(X_batch, Y_batch, labda)
                getattr(self, 'b' + str(j))[i] = getattr(self, 'b' + str(j))[i] - 2*h
                _, c3 = self.compute_cost(X_batch, Y_batch, labda)
                grads['b' + str(j)][i] = (c2-c3) / (2*h)

            W_try = np.copy(selfW)
            for i in np.ndindex(selfW.shape):
                selfW = W_try[:,:]
                selfW[i] = selfW[i] + h
                _, c2 = self.compute_cost(X_batch, Y_batch, labda)
                getattr(self, 'W' + str(j))[i] = getattr(self, 'W' + str(j))[i] - 2*h
                _, c3 = self.compute_cost(X_batch, Y_batch, labda)
                grads['W' + str(j)][i] = (c2-c3) / (2*h)

        return grads['W1'], grads['b1'], grads['W2'], grads['b2']


    def plot_performance(self, n_epochs, i_train, i_val, title, y_label, y_max=4):
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
            plot_performance=False):
        """Trains the model using mini-batch gradient descent

        Args:
            X          (np.ndarray): data matrix (D, N)
            Y          (np.ndarray): one-hot-encoding labels matrix (C, N)
            labda           (float): regularization term
            batch_s           (int): batche size
            eta             (float): learning rate
            n_epochs          (int): number of training epochs
            verbose          (bool): boolean for deciding on textual output
            plot_performance (bool): boolean for deciding whether to plot costs
            title             (str): title for cost plots

        Returns:
            acc_train (float): the accuracy on the training set
            acc_val   (float): the accuracy on the validation set
            acc_test  (float): the accuracy on the testing set
        """
        if plot_performance:
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

                grad_W1, grad_b1, grad_W2, grad_b2 = self.compute_gradients(
                        X_batch, Y_batch, labda)
                self.W1 -= eta * grad_W1
                self.b1 -= eta * grad_b1
                self.W2 -= eta * grad_W2
                self.b2 -= eta * grad_b2

                if t <= n_s:
                    eta = eta_min + t/n_s * (eta_max - eta_min)

                elif t <= 2*n_s:
                    eta = eta_max - (t - n_s)/n_s * (eta_max - eta_min)

                t = (t+1) % (2*n_s)

            if plot_performance:
                loss_train[n], costs_train[n] = self.compute_cost(X, Y, labda)
                loss_val[n], costs_val[n] = self.compute_cost(
                        self.X_val, self.Y_val, labda)
                acc_train[n] = self.compute_accuracy(self.X_train, self.y_train)
                print(acc_train[n])
                acc_val[n] = self.compute_accuracy(self.X_val, self.y_val)

        if plot_performance:
            self.plot_performance(n_epochs, costs_train, costs_val, plot_id + "_cost_plot",
                    y_label="Cost", y_max=4)
            self.plot_performance(n_epochs, loss_train, loss_val, plot_id + "_loss_plot",
                    y_label="Loss", y_max=2.5)
            self.plot_performance(n_epochs, acc_train, acc_val, plot_id + "_acc_plot",
                    y_label="Accuracy", y_max=0.8)

        acc_train = self.compute_accuracy(self.X_train, self.y_train)
        acc_val   = self.compute_accuracy(self.X_val, self.y_val)
        acc_test  = self.compute_accuracy(self.X_test, self.y_test)

        if verbose:
            print("The accuracy on the training set is: "   + str(acc_train))
            print("The accuracy on the validation set is: " + str(acc_val))
            print("The accuracy on the testing set is: "    + str(acc_test))

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
    def test_sizes(self):
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

        clf = Classifier(data, labels)

        grad_W1_test, grad_b1_test, grad_W2_test, grad_b2_test = \
                clf.compute_gradients(X_train, Y_train, labda=0)

        self.assertEqual(clf.X_train.shape, (3072, 10000))
        self.assertEqual(clf.Y_train.shape, (10, 10000))
        self.assertEqual(np.shape(clf.y_train), (10000,))
        self.assertEqual(clf.W1.shape, (50, 3072))
        self.assertEqual(clf.b1.shape, (50, 1))
        self.assertEqual(clf.W2.shape, (10, 50))
        self.assertEqual(clf.b2.shape, (10, 1))
        self.assertEqual(clf.evaluate_classifier(X_train)[0].shape,
                        (50, 10000))
        self.assertEqual(clf.evaluate_classifier(X_train)[1].shape,
                        (10, 10000))
        self.assertAlmostEqual(sum(sum(clf.evaluate_classifier(
            X_train[:, 0].reshape((X_train.shape[0], 1)))[1])), 1)
        self.assertIsInstance(clf.compute_cost(X_train, Y_train, labda=0)[1],
                              float)
        self.assertEqual(grad_W1_test.shape, clf.W1.shape)
        self.assertEqual(grad_b1_test.shape, clf.b1.shape)
        self.assertEqual(grad_W2_test.shape, clf.W2.shape)
        self.assertEqual(grad_b2_test.shape, clf.b2.shape)


    def test_array_equality(self):
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

        clf = Classifier(data, labels)

        Ws = {'W1': (50, 30)}#, 'W2': (30, 40)}
        bs = {}#{'b1': (1, 30), 'b2': (1, 10)}
        params = clf.init_parameters(Ws, bs)

        clf.W1 = params["W1"]

        grad_W1_ana, grad_b1_ana, grad_W2_ana, grad_b2_ana = \
                clf.compute_gradients(clf.X_train[:30, :5],
                                          clf.Y_train[:30, :5],
                                          labda=0)

        grad_W1_num, grad_b1_num, grad_W2_num, grad_b2_num = clf.compute_gradients_num(
                clf.X_train[:30, :5],
                clf.Y_train[:30, :5],
                labda=0)

        self.assertArrayAlmostEqual(grad_W1_ana, grad_W1_num)
        self.assertArrayAlmostEqual(grad_W2_ana, grad_W2_num)
        self.assertArrayAlmostEqual(grad_b1_ana, grad_b1_num)
        self.assertArrayAlmostEqual(grad_b2_ana, grad_b2_num)


if __name__ == '__main__':
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

        labels = unpickle( 'datasets/cifar-10-batches-py/batches.meta')[ b'label_names']

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


    def replicate_figure_three():
        """Replicates Figure 3 from the assignment sheet"""
        data, labels = train_on_one_data_batch()
        clf = Classifier(data, labels)
        clf.mini_batch_gd(data['X_train'], data['Y_train'], labda=0.01,
                batch_s=100, eta_min=1e-5, eta_max=1e-1, n_s = 500,
                plot_id="fig3", plot_performance=True, n_epochs=10)


    def replicate_fig_four():
        """Replicates Figure 4 from the assignment sheet"""
        data, labels = train_on_one_data_batch()
        clf = Classifier(data, labels)
        clf.mini_batch_gd(data['X_train'], data['Y_train'], labda=0.01,
                batch_s=100, eta_min=1e-5, eta_max=1e-1, n_s = 800,
                plot_id="fig4", plot_performance=True, n_epochs=48)


    def train_best_classifier():
        """Train the best classifier"""
        data, labels = train_on_all_data_batches(val=1000)

        acc_train_set = []
        acc_val_set = []
        acc_test_set = []
        for j in range(10):
            clf = Classifier(data, labels)
            acc_train, acc_val, acc_test = clf.mini_batch_gd(
                    data['X_train'], data['Y_train'], labda=0.00895,
                    batch_s=100, eta_min=1e-5, eta_max=1e-1, n_s=980,
                    n_epochs=12, verbose=False)

            acc_train_set.append(acc_train)
            acc_val_set.append(acc_val)
            acc_test_set.append(acc_test)

        print("Train mean acc:" + str(statistics.mean(acc_train_set)))
        print("Val mean acc:" + str(statistics.mean(acc_val_set)))
        print("Test mean acc:" + str(statistics.mean(acc_test_set)))
        print("Train stdev acc:" + str(statistics.stdev(acc_train_set)))
        print("Val stdev acc:" + str(statistics.stdev(acc_val_set)))
        print("Test stdev acc:" + str(statistics.stdev(acc_test_set)))

        np.random.seed(0)
        clf = Classifier(data, labels)
        clf.mini_batch_gd(data['X_train'], data['Y_train'], labda=0.00895,
                batch_s=100, eta_min=1e-5, eta_max=1e-1, n_s=980,
                plot_id="best", plot_performance=True, n_epochs=12)


    ## REPLICATE FIGURE THREEE
    replicate_figure_three()

    ## REPLICATE FIGURE FOUR
    replicate_fig_four()

    ## BEST CLASSIFIER
    train_best_classifier()

    ## UNIT TESTING
    unittest.main()

