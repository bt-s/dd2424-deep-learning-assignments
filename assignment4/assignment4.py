#!/usr/bin/python3

"""assignment4regular.py: Implementation of a vanilla recurrent neural network
to synthesize English text character by character.

Assignment 4 of the 2019 DD2424 Deep Learning in Data Science course at
KTH Royal Institute of Technology"""

__author__ = "Bas Straathof"

import sys
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import OrderedDict


def load_data(filename):
    """Read in a text file and preprocess it for the RNN

    Args:
        filename (str): filename of the text to be loaded

    Returns:
        data (dict):
          - book_data   (str): the text stored in a string
          - book_chars (dict): dict of unique characters in the text
          - vocab_len   (int): length of the vocabulary
          - char_to_ind (dict): mapping of characters to indices
          - ind_to_char (dict): mapping of indices to characters
    """
    book_data = open(filename, 'r', encoding='utf8').read()
    book_chars = list(set(book_data))

    data = {"book_data": book_data, "book_chars": book_chars,
            "vocab_len": len(book_chars), "char_to_ind": OrderedDict(
                (char, ix) for ix, char in enumerate(book_chars)),
            "ind_to_char": OrderedDict((ix, char) for ix, char in
                enumerate(book_chars))}

    return data


class RNN():
    """A vanilla recurrent neural network model"""
    def __init__(self, data, m=100, eta=.1, seq_length=25):
        """Initialize the RNN
        Args:
            data (dict): the data object containing
                - book_data   (str): the text stored in a string
                - book_chars (dict): dict of unique characters in the text
                - vocab_len   (int): length of the vocabulary
                - char_to_ind (dict): mapping of characters to indices
                - ind_to_char (dict): mapping of indices to characters
            m (int): dimensionality of the hidden state
            eta (float): learning rate
            seq_length (int): the length of the sequence that the model uses to
                              traverse the text
        """
        self.m, self.eta, self.N = m, eta, seq_length

        for k, v in data.items():
            setattr(self, k, v)

        self.b, self.c, self.U, self.V, self.W = \
                self._init_parameters(self.m, self.vocab_len)


    @staticmethod
    def _init_parameters(m, K, sig=0.01):
        """Initialize the RNN parameters

        Args:
            m (int): shape of the layer
            K (int): shape of the layer
            sig (float): scalar for the random initization

        Returns:
            b (np.ndarray): bias vector of length (m x 1)
            c (np.ndarray): bias vector of length (K x 1)
            U (np.ndarray): weight matrix of shape (m x K)
            V (np.ndarray): weight matrix of shape (K x m)
            W (np.ndarray): weight matrix of shape (m x m)
        """
        b = np.zeros((m, 1))
        c = np.zeros((K, 1))

        U = np.random.normal(0, sig, size=(m, K))
        W = np.random.normal(0, sig, size=(m, m))
        V = np.random.normal(0, sig, size=(K, m))

        return b, c, U, V, W


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
    def _tanh(x):
        """Computes the tanh activation

        Args:
            x (np.ndarray): input matrix

        Returns:
            x (np.ndarray): tanh matrix
        """
        return np.tanh(x)


    def evaluate_classifier(self, h, x):
        """ Evaluate the classifier

        Args:
            h (np.ndarray): hidden state sequence
            x (np.ndarray): sequence of input vectors

        Returns:
            a (np.ndarray): linear transformation of W and U + bias b
            h (np.ndarray): tanh activation of a
            o (np.ndarray): linear transformation of V + bias c
            p (np.ndarray): softmax activation of o
        """
        a = self.W@h + self.U@x + self.b
        h = self._tanh(a)
        o = self.V@h + self.c
        p = self._softmax(o)

        return a, h, o, p


    def synthesize_text(self, h, ix, n):
        """Synthesize text based on the hidden state sequence

        Args:
            h (np.ndarray): hidden state sequence
            ix       (int): index to obtain the RNN's first dummy input vector
            n        (int): length of the sequence to be generated

        Returns:
          txt (str): a synthesized string of length n
        """
        # The next input vector
        xnext = np.zeros((self.vocab_len, 1))
        # Use the index to set the net input vector
        xnext[ix] = 1 # 1-hot-encoding

        txt = ''
        for t in range(n):
            _, h, _, p = self.evaluate_classifier(h, xnext)
            # Sample from the vocabulary based on the flattened probability
            # vector p and the uniform distribution
            ix = np.random.choice(range(self.vocab_len), p=p.flat)
            xnext = np.zeros((self.vocab_len, 1))
            xnext[ix] = 1 # 1-hot-encoding
            txt += self.ind_to_char[ix]

        return txt


    def compute_gradients(self, inputs, targets, hprev):
        """ Analytically computes the gradients of the weight and bias parameters

        Args:
            inputs      (list): indices of the chars of the input sequence
            targets     (list): indices of the chars of the target sequence
            hprev (np.ndarray): previous learnt hidden state sequence

        Returns:
            grads (dict): the updated analytical gradients dU, dW, dV, db and dc
            loss (float): the current loss
            h (np.ndarray): newly learnt hidden state sequence
        """
        n = len(inputs)
        loss = 0

        # Dictionaries for storing values during the forward pass
        aa, xx, hh, oo, pp = {}, {}, {}, {}, {}
        hh[-1] = np.copy(hprev)

        # Forward pass
        for t in range(n):
            xx[t] = np.zeros((self.vocab_len, 1))
            xx[t][inputs[t]] = 1 # 1-hot-encoding

            aa[t], hh[t], oo[t], pp[t] = self.evaluate_classifier(hh[t-1], xx[t])

            loss += -np.log(pp[t][targets[t]][0]) # update the loss

        # Dictionary for storing the gradients
        grads = {"W": np.zeros_like(self.W), "U": np.zeros_like(self.U),
                 "V": np.zeros_like(self.V), "b": np.zeros_like(self.b),
                 "c": np.zeros_like(self.c), "o": np.zeros_like(pp[0]),
                 "h": np.zeros_like(hh[0]), "h_next": np.zeros_like(hh[0]),
                 "a": np.zeros_like(aa[0])}

        # Backward pass
        for t in reversed(range(n)):
            grads["o"] = np.copy(pp[t])
            grads["o"][targets[t]] -= 1

            grads["V"] += grads["o"]@hh[t].T
            grads["c"] += grads["o"]

            grads["h"] = self.V.T@grads["o"] + grads["h_next"]
            grads["a"] = np.multiply(grads["h"], (1 - np.square(hh[t])))

            grads["U"] += grads["a"]@xx[t].T
            grads["W"] += grads["a"]@hh[t-1].T
            grads["b"] += grads["a"]

            grads["h_next"] = self.W.T@grads["a"]

        # Drop redundant gradients
        grads = {k: grads[k] for k in grads if k not in ["o", "h", "h_next", "a"]}

        # Clip the gradients
        for grad in grads:
            grads[grad] = np.clip(grads[grad], -5, 5)

        # Update the hidden state sequence
        h = hh[n-1]

        return grads, loss, h


    def compute_gradients_num(self, inputs, targets, hprev, h, num_comps=20):
        """Numerically computes the gradients of the weight and bias parameters

        Args:
            inputs      (list): indices of the chars of the input sequence
            targets     (list): indices of the chars of the target sequence
            hprev (np.ndarray): previous learnt hidden state sequence
            h     (np.float64): marginal offset
            num_comps    (int): number of entries per gradient to compute

        Returns:
            grads (dict): the numerically computed gradients dU, dW, dV,
                          db and dc
        """
        rnn_params = {"W": self.W, "U": self.U, "V": self.V, "b": self.b, "c": self.c}
        num_grads  = {"W": np.zeros_like(self.W), "U": np.zeros_like(self.U),
                      "V": np.zeros_like(self.V), "b": np.zeros_like(self.b),
                      "c": np.zeros_like(self.c)}

        for key in rnn_params:
            for i in range(num_comps):
                old_par = rnn_params[key].flat[i] # store old parameter
                rnn_params[key].flat[i] = old_par + h
                _, l1, _ = self.compute_gradients(inputs, targets, hprev)
                rnn_params[key].flat[i] = old_par - h
                _, l2, _ = self.compute_gradients(inputs, targets, hprev)
                rnn_params[key].flat[i] = old_par # reset parameter to old value
                num_grads[key].flat[i] = (l1 - l2) / (2*h)

        return num_grads


    def check_gradients(self, inputs, targets, hprev, num_comps=20):
        """Check similarity between the analytical and numerical gradients

        Args:
            inputs      (list): indices of the chars of the input sequence
            targets     (list): indices of the chars of the target sequence
            hprev (np.ndarray): previous learnt hidden state sequence
            num_comps    (int): number of gradient comparisons
        """
        grads_ana, _, _ = self.compute_gradients(inputs, targets, hprev)
        grads_num = self.compute_gradients_num(inputs, targets, hprev, 1e-5)

        print("Gradient checks:")
        for grad in grads_ana:
            num   = abs(grads_ana[grad].flat[:num_comps] -
                    grads_num[grad].flat[:num_comps])
            denom = np.asarray([max(abs(a), abs(b)) + 1e-10 for a,b in
                zip(grads_ana[grad].flat[:num_comps],
                    grads_num[grad].flat[:num_comps])
            ])
            max_rel_error = max(num / denom)

            print("The maximum relative error for the %s gradient is: %e." %
                    (grad, max_rel_error))
        print()


def main(argv):
    """Main function of the vanilla RNN to synthesize English text character
    by character.

    Args:
        argv[1] (str): filename of the input text
    """
    # Book position tracker, iteration, epoch
    e, n, epoch = 0, 0, 0
    num_epochs = 10

    data = load_data(argv[1])
    rnn = RNN(data)

    rnn_params = {"W": rnn.W, "U": rnn.U, "V": rnn.V, "b": rnn.b, "c": rnn.c}

    mem_params = {"W": np.zeros_like(rnn.W), "U": np.zeros_like(rnn.U),
                  "V": np.zeros_like(rnn.V), "b": np.zeros_like(rnn.b),
                  "c": np.zeros_like(rnn.c)}

    while epoch < num_epochs:
        if n == 0 or e >= (len(rnn.book_data) - rnn.N - 1):
            if epoch != 0: print("Finished %i epochs." % epoch)
            hprev = np.zeros((rnn.m, 1))
            e = 0
            epoch += 1

        inputs = [rnn.char_to_ind[char] for char in rnn.book_data[e:e+rnn.N]]
        targets = [rnn.char_to_ind[char] for char in rnn.book_data[e+1:e+rnn.N+1]]

        grads, loss, hprev = rnn.compute_gradients(inputs, targets, hprev)

        # Compute the smooth loss
        if n == 0 and epoch == 1: smooth_loss = loss
        smooth_loss = 0.999 * smooth_loss + 0.001 * loss

        # Check gradients
        if n == 0: rnn.check_gradients(inputs, targets, hprev)

        # Print the loss
        if n % 100 == 0: print('Iteration %d, smooth loss: %f' % (n, smooth_loss))

        # Print synthesized text
        if n % 500 == 0:
            txt = rnn.synthesize_text(hprev, inputs[0], 200)
            print('\nSynthesized text after %i iterations:\n %s\n' % (n, txt))
            print('Smooth loss: %f' % smooth_loss)

        # Adagrad
        for key in rnn_params:
            mem_params[key] += grads[key] * grads[key]
            rnn_params[key] -= rnn.eta / np.sqrt(mem_params[key] +
                    np.finfo(float).eps) * grads[key]

        e += rnn.N
        n += 1


if __name__ == '__main__':
    main(sys.argv)

