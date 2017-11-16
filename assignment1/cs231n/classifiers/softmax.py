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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0];
  num_class = W.shape[1];
  scores = np.zeros([num_train,num_class])
  scores_softmax = np.zeros([num_train,num_class])
  for i in list(range(num_train)):
      scores = X[i].dot(W)
      shift_scores = scores - max(scores) # for numeric stability and computation load
      for j in list(range(num_class)):
          scores_softmax[i,j] = np.exp(shift_scores[j])/np.sum(np.exp(shift_scores))
          if j == y[i]:
              dW[:,j] += (scores_softmax[i,j]-1)*X[i]
          else:
              dW[:,j] += scores_softmax[i,j]*X[i]
      loss += -np.log(scores_softmax[i,y[i]])
  loss = loss/num_train + reg*np.sum(W*W)
  dW = dW/num_train + 2*reg*W
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_class = W.shape[1]
  scores = X.dot(W) # (N,C)
  scores_shift = scores - np.max(scores,axis=1).reshape(-1,1)
  scores_softmax = np.exp(scores_shift)/np.sum(np.exp(scores_shift),axis=1).reshape(-1,1)
  scores_softmax_y = scores_softmax[range(num_train),list(y)].reshape(-1,1) # (N,1)
  loss = -np.sum(np.log(scores_softmax_y))
  coeff = scores_softmax.copy()
  coeff[range(num_train),list(y)] -= 1
  dW = (X.T).dot(coeff)
  loss = loss/num_train + reg*np.sum(W*W)
  dW = dW/num_train + 2*reg*W
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

