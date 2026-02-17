from builtins import range
import numpy as np
from random import shuffle


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]

    for i in range(num_train):
        one_hot = np.zeros(num_classes)
        one_hot[y[i]] = 1.0
        scores = X[i].dot(W)

        # compute the probabilities in numerically stable way
        scores -= np.max(scores)

        p = np.exp(scores)
        p /= p.sum()  # normalize
        logp = np.log(p[y[i]])
        x_col = np.expand_dims(X[i], axis=1)
        diff = np.expand_dims(p - one_hot, axis=0)

        loss -= logp  # negative log probability is the loss
        dW[:] += x_col @ diff

    # normalized hinge loss plus regularization
    loss = loss / num_train + reg * np.sum(W * W)
    dW /= num_train
    dW += reg * W

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.

    num_classes = W.shape[1]
    num_train = X.shape[0]
    one_hot = np.zeros((num_train, num_classes))
    one_hot[np.arange(num_train), y] = 1.0
    scores = X @ W
    scores -= np.max(scores, axis=1, keepdims=True)
    p = np.exp(scores)
    p /= p.sum(axis=1, keepdims=True)
    logp = np.log(p)
    logp =one_hot*logp
    sum=np.sum(one_hot * logp)
    loss = -np.sum(one_hot * logp) / num_train + reg * np.sum(W * W)

    diff = p - one_hot
    dW = X.T @ diff
    dW /= num_train
    dW += reg * W


    return loss, dW


