from builtins import range
from builtins import object
import os
import numpy as np

from dl.data_utils import get_CIFAR10_data
from dl.layers import *
from dl.layer_utils import *

from dl.gradient_check import eval_numerical_gradient


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(
            self,
            input_dim=3 * 32 * 32,
            hidden_dim=100,
            num_classes=10,
            weight_scale=1e-3,
            reg=0.0,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        self.params['W1'] = np.random.normal(0.0, weight_scale, (input_dim, hidden_dim))
        self.params['b1'] = np.zeros((hidden_dim,))
        self.params['W2'] = np.random.normal(0.0, weight_scale, (hidden_dim, num_classes))
        self.params['b2'] = np.zeros((num_classes,))
    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        cache = {}
        h1, h1_cache = affine_forward(X, self.params['W1'], self.params['b1'])
        # h1=np.dot(X,self.params['W1'])+self.params['b1']
        # h1_diff=np.abs(h1 - h_1).sum()
        cache['first_layer'] = h1_cache
        # a1=np.maximum(0,h1)
        a1, a1_cache = relu_forward(h1)
        cache['first_activation'] = a1_cache
        h2, h2_cache = affine_forward(a1, self.params['W2'], self.params['b2'])
        # h2=np.dot(a1,self.params['W2'])+self.params['b2']
        cache['second_layer'] = h2_cache

        # h2_diff = np.abs(h2 - h_2).sum()
        scores = h2

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}

        loss, dZ2 = softmax_loss(scores, y)
        reg_loss = 0.5 * self.reg * (np.sum(self.params['W1'] ** 2) + np.sum(self.params['W2'] ** 2))
        loss += reg_loss
        # grad_d_W2=np.dot(a1.T,dZ2)#dL/dw2=dL/dZ2(upstream)*dZ2/dW(local) -- weights step
        # grad_d_W2/=N
        # grad_d_W2+=self.reg*self.params['W2']

        # Layer 2 Backward
        # dL/dW2 = dZ2(upstream) * A1(local)
        # dL/dA1 = dZ2(upstream) * W2(local)
        dA1, dW2, dB2 = affine_backward(dZ2,
                                        cache[
                                            'second_layer'])  # dL/dW2=dZ2/dW2(local) *dL/dZ2 (upstream) --weights step
        dW2 += self.reg * self.params['W2']
        grads['W2'] = dW2
        grads['b2'] = dB2
        # ReLU Backward
        # dL/dZ1 = dL/dA1(upstream) * Mask(local)
        # dA1 = np.dot(dZ2, self.params['W2'].T)  # dL/dA1=dL/dZ2(upstream)*dZ2/dA1(local) -- pass back step
        dZ1 = relu_backward(dA1, cache['first_activation'])  # dZ1=relu_back(dA1)

        # dZ1=dA1*(h1>0)#dL/dZ1=dL/dZ2(upstream)*dZ2/da1(weights)*da1/dz1(activation)
        # dW1=np.dot(dZ1,X.T)#dL/dW1=dL/dz1*dz1/dW1
        # Layer 1 Backward
        # dL/dW1 = dZ1(upstream) * X(local)
        _, dW1, dB1 = affine_backward(dZ1, cache[
            'first_layer'])  # dL/dW1=dZ1/dW1 (local) * dL/dZ1 (upstream) -- weights step
        dW1 += self.reg * self.params['W1']
        grads['W1'] = dW1
        grads['b1'] = dB1

        # grad_a1=(h1>0)*grad_dW2 # relu backwards

        return loss, grads

    def save(self, fname):
        """Save model parameters."""
        fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
        params = self.params
        np.save(fpath, params)
        print(fname, "saved.")

    def load(self, fname):
        """Load model parameters."""
        fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
        if not os.path.exists(fpath):
            print(fname, "not available.")
            return False
        else:
            params = np.load(fpath, allow_pickle=True).item()
            self.params = params
            print(fname, "loaded.")
            return True


class FullyConnectedNet(object):
    """Class for a multi-layer fully connected neural network.

    Network contains an arbitrary number of hidden layers, ReLU nonlinearities,
    and a softmax loss function. This will also implement dropout and batch/layer
    normalization as options. For a network with L layers, the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional and the {...} block is
    repeated L - 1 times.

    Learnable parameters are stored in the self.params dictionary and will be learned
    using the Solver class.
    """

    def __init__(
            self,
            hidden_dims,
            input_dim=3 * 32 * 32,
            num_classes=10,
            dropout_keep_ratio=1,
            normalization=None,
            reg=0.0,
            weight_scale=1e-2,
            dtype=np.float32,
            seed=None,
    ):
        """Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout_keep_ratio: Scalar between 0 and 1 giving dropout strength.
            If dropout_keep_ratio=1 then the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
            are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
            initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
            this datatype. float32 is faster but less accurate, so you should use
            float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers.
            This will make the dropout layers deteriminstic so we can gradient check the model.
        """
        self.normalization = normalization
        self.use_dropout = dropout_keep_ratio != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        self.params[f'W1'] = np.random.normal(0.0,weight_scale,(input_dim, hidden_dims[0]))
        self.params[f'b1'] = np.zeros((hidden_dims[0]))
        if self.normalization is not None:
            self.params[f'gamma1'] =np.ones((hidden_dims[0],))
            self.params[f'beta1'] = np.zeros((hidden_dims[0],))
        for layer_num in range(1, self.num_layers-1):
            if self.normalization is not None:
                self.params[f'gamma{layer_num+1}']=np.ones((hidden_dims[layer_num],))
                self.params[f'beta{layer_num+1}']=np.zeros((hidden_dims[layer_num],))
            self.params[f'W{layer_num + 1}'] = np.random.normal(0.0,weight_scale,(hidden_dims[layer_num - 1], hidden_dims[layer_num]))
            self.params[f'b{layer_num + 1}'] = np.zeros((hidden_dims[layer_num]))

        self.params[f'W{self.num_layers}'] = np.random.normal(0.0,weight_scale,(hidden_dims[-1], num_classes))
        self.params[f'b{self.num_layers}'] = np.zeros((num_classes,))


        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout_keep_ratio}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype.
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """Compute loss and gradient for the fully connected net.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
            scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
            names to gradients of the loss with respect to those parameters.
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None
        h=X
        caches={}
        for layer in range(self.num_layers-1):
            h,cache=affine_forward(h,self.params[f'W{layer+1}'],self.params[f'b{layer+1}'])
            caches[f'layer_{layer+1}']=cache
            if self.normalization=="batchnorm":
                h,cache=batchnorm_forward(h,self.params[f'gamma{layer+1}'],self.params[f'beta{layer+1}'],self.bn_params[layer])
                caches[f'batchnorm_{layer+1}'] = cache
            elif self.normalization=="layernorm":
                h,cache=layernorm_forward(h,self.params[f'gamma{layer+1}'],self.params[f'beta{layer+1}'],self.bn_params[layer])
                caches[f'layernorm_{layer+1}'] = cache
            h,cache=relu_forward(h)
            caches[f'activation_{layer+1}'] = cache
            if self.use_dropout:
                h,cache=dropout_forward(h,self.dropout_param)
                caches[f'dropout_{layer+1}'] = cache
        scores,cache=affine_forward(h,self.params[f'W{self.num_layers}'],self.params[f'b{self.num_layers}'])
        caches[f'layer_{self.num_layers}']=cache

        # If test mode return early.
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        loss,dZ=softmax_loss(scores,y)
        w_sums=sum(np.sum(self.params[f'W{w+1}']**2) for w in range(self.num_layers))

        reg_loss=0.5*self.reg*w_sums
        loss+=reg_loss
        for layer in range(self.num_layers,0,-1):
            dZ,dW,dB = affine_backward(dZ,caches[f'layer_{layer}'])
            grads[f'W{layer}']=dW+self.reg*self.params[f'W{layer}']
            grads[f'b{layer}']=dB

            if layer>1:
                if self.use_dropout:
                    dZ=dropout_backward(dZ,caches[f'dropout_{layer-1}'])
                dZ=relu_backward(dZ,caches[f'activation_{layer-1}'])

                if self.normalization == "batchnorm":
                    dZ,dgamma,dbeta = batchnorm_backward_alt(dZ, caches[f'batchnorm_{layer - 1}'])
                    grads[f'gamma{layer-1}'] = dgamma
                    grads[f'beta{layer-1}'] = dbeta
                elif self.normalization == "layernorm":
                    dZ,dgamma,dbeta = layernorm_backward(dZ, caches[f'layernorm_{layer - 1}'])
                    grads[f'gamma{layer-1}'] = dgamma
                    grads[f'beta{layer - 1}'] = dbeta


        return loss, grads

    def save(self, fname):
        """Save model parameters."""
        fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
        params = self.params
        np.save(fpath, params)
        print(fname, "saved.")

    def load(self, fname):
        """Load model parameters."""
        fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
        if not os.path.exists(fpath):
            print(fname, "not available.")
            return False
        else:
            params = np.load(fpath, allow_pickle=True).item()
            self.params = params
            print(fname, "loaded.")
            return True


