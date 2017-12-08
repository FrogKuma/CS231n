from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################
        C,H,W = input_dim
        self.params['W1'] = weight_scale*np.random.randn(num_filters,C,filter_size,filter_size)
        self.params['b1'] = np.zeros(num_filters)
        self.params['W2'] = weight_scale*np.random.randn((H//2)*(W//2)*num_filters,hidden_dim)
        self.params['b2'] = np.zeros(hidden_dim)
        self.params['W3'] = weight_scale*np.random.randn(hidden_dim,num_classes)
        self.params['b3'] = np.zeros(num_classes)
        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        #out_conv,cache_conv = conv_forward_naive(X,W1,b1,conv_param) # conv
        #out_ReLu1,cache_ReLu1 = relu_forward(out_conv) #ReLu 1
        #out_pool,cache_pool = max_pool_forward_naive(out_ReLu1,pool_param) # pooling
        out_pool,cache_pool = conv_relu_pool_forward(X,W1,b1,conv_param,pool_param)
        #out_fc1,cache_fc1 = affine_forward(out_pool,W2,b2) # fc 1
        #out_ReLu2,cache_ReLu2 = relu_forward(out_fc1) #ReLu 2
        out_ReLu2,cache_ReLu2 = affine_relu_forward(out_pool,W2,b2)
        scores,cache_fc2 = affine_forward(out_ReLu2,W3,b3) # fc 2
        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        #H,W = 2*W2.shape[0],2*W2.shape[1]
        #N = X.shape[0]
        #F = W1.shape[0]
        #loss,dout_sf = softmax_loss(scores,y) # dout of softmax (N,O), O = # of output class
        #loss += self.reg*(np.sum(W1*W1)+np.sum(W2*W2)+np.sum(W3*W3))
        #dW3 = out_ReLu2.T.dot(dout_sf) # (h,O)
        #dout_Relu2 = dout_sf.dot(W3.T) #(N,h)
        #dout_fc1 = dout_Relu2*(out_fc1>=0) # (N,h)
        #dW2 = out_pool.reshape(N,-1).T.dot(dout_fc1)
        #dout_pool = dout_fc1.dot(W2.T).reshape(out_pool.shape)
        #_,dW1,_ = conv_relu_pool_backward(dout_pool,cache_pool)

        loss,dout = softmax_loss(scores,y)
        loss += self.reg*(np.sum(W1*W1)+np.sum(W2*W2)+np.sum(W3*W3))
        dX3,dW3,db3 = affine_backward(dout,cache_fc2)
        dX2,dW2,db2 = affine_relu_backward(dX3,cache_ReLu2)
        dX1,dW1,db1 = conv_relu_pool_backward(dX2,cache_pool)
        dW1 += 2*self.reg*W1
        dW2 += 2*self.reg*W2
        dW3 += 2*self.reg*W3
        grads['W1'],grads['b1'] = dW1,db1
        grads['W2'],grads['b2'] = dW2,db2
        grads['W3'],grads['b3'] = dW3,db3
        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
