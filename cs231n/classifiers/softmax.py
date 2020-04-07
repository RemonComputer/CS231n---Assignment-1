from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

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
    num_train = X.shape[0]
    num_classes = W.shape[1]

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    for i in range(num_train):
        s = X[i].dot(W)
        s -= np.max(s) #to improve the numerical stability of the computation
        s_yi = np.exp(s[y[i]])
        s_sum = np.sum(np.exp(s))
        loss += -np.log(s_yi / s_sum)
       
        for j in range(num_classes):
            softmax_output_j = np.exp(s[j])/sum(np.exp(s))
            if j == y[i]:
                dW[:,j] += (-1 + softmax_output_j) *X[i]
            else:
                dW[:,j] += softmax_output_j *X[i]
                
    

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    loss = loss / num_train
    loss += reg * np.sum (W * W)
    dW = dW / num_train
    dW += reg* W 


    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[0]
    num_classes = W.shape[1]


    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    s = X.dot(W)
    s = s - np.max(s,axis=1, keepdims =True)
    s_sum = np.sum(np.exp(s), axis=1, keepdims =True)
    softmax = np.exp(s) / s_sum
    dim = np.arange(num_train), y
    loss = np.sum(-np.log(softmax[dim]))
    
    #to calculate the gradient
    softmax[dim] -= 1
    dW = X.T.dot(softmax)
    
    loss = loss / num_train
    loss += reg * np.sum (W * W)
    dW = dW / num_train
    dW += reg* W 
    
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
