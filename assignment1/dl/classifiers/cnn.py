from builtins import object
import numpy as np

from ..layers import *
from ..fast_layers import *
from ..layer_utils import *

class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(
        self,
        input_dim=(3, 32, 32),
        num_filters=32,
        filter_size=7,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
        dtype=np.float32,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in the convolutional layer
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

        C,H,W=input_dim
        pad=(filter_size - 1) // 2
        stride=1
        self.params['W1'] = np.random.normal(0.0, weight_scale, (num_filters,input_dim[0],filter_size,filter_size))
        self.params['b1'] = np.zeros((num_filters,))
        H_conv = 1 + (H + 2 * pad - filter_size) // stride
        W_conv = 1 + (W + 2 * pad - filter_size) // stride
        H_pool = H_conv // 2
        W_pool = W_conv // 2
        D=num_filters*H_pool*W_pool
        self.params['W2'] = np.random.normal(0.0, weight_scale, (D, hidden_dim))
        self.params['b2'] = np.zeros((hidden_dim,))
        self.params['W3'] = np.random.normal(0.0, weight_scale, (hidden_dim, num_classes))
        self.params['b3'] = np.zeros((num_classes,))

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]
        W3, b3 = self.params["W3"], self.params["b3"]

        # pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size
        filter_size = W1.shape[2]
        conv_param = {"stride": 1, "pad": (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {"pool_height": 2, "pool_width": 2, "stride": 2}

        scores = None
        out,conv_relu_pool_cache=conv_relu_pool_forward(X,W1,b1,conv_param,pool_param)
        out,affine_relu_cache=affine_relu_forward(out,W2,b2)
        scores,scores_cache=affine_forward(out,W3,b3)

        if y is None:
            return scores

        loss, grads = 0, {}
        loss,dZ3=softmax_loss(scores,y)
        reg_loss = 0.5 * self.reg * (np.sum(self.params['W1'] ** 2) + np.sum(self.params['W2'] ** 2)+np.sum(self.params['W3'] ** 2))
        loss += reg_loss
        dZ3,dW3,db3=affine_backward(dZ3,scores_cache)
        grads['W3'] = dW3 + self.reg * self.params['W3']
        grads['b3'] = db3
        dZ2,dW2,db2=affine_relu_backward(dZ3,affine_relu_cache)
        grads['W2'] = dW2 + self.reg * self.params['W2']
        grads['b2'] = db2
        dX1,dW1,db1=conv_relu_pool_backward(dZ2,conv_relu_pool_cache)
        grads['W1'] = dW1 + self.reg * self.params['W1']
        grads['b1'] = db1

        return loss, grads
