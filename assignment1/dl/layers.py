from builtins import range
import numpy as np

from dl.im2col import im2col_naive, col2im


def affine_forward(x, w, b):
    """Computes the forward pass for an affine (fully connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    D_shape = np.prod(x.shape[1:])
    x_reshaped = x.reshape(x.shape[0], D_shape)
    out = np.dot(x_reshaped, w) + b
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """Computes the backward pass for an affine (fully connected) layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    D_shape = np.prod(x.shape[1:])
    x_reshaped = x.reshape(x.shape[0], D_shape)
    dx = np.dot(dout, w.T).reshape(-1, *x.shape[1:])
    dw = np.dot(x_reshaped.T, dout)
    db = np.sum(dout, axis=0)  # np.dot(dout.T,np.ones(dout.shape[0]).reshape(-1))
    return dx, dw, db


def relu_forward(x):
    """Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    out = np.maximum(0, x)
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache

    dx = (x > 0) * dout
    return dx


def softmax_loss(x, y):
    """Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None

    x -= np.max(x, axis=1, keepdims=True)
    N = x.shape[0]
    p = np.exp(x)
    p /= p.sum(axis=1, keepdims=True)
    one_hot = np.zeros_like(x)
    one_hot[np.arange(N), y] = 1
    correct_class_probs = p[np.arange(N), y]

    loss = -np.sum(np.log(correct_class_probs + 1e-15)) / N

    diff = p - one_hot  # dL/dS , now we want to multiply by dS/dX
    dx = diff / N
    return loss, dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == "train":
        mean = np.mean(x, axis=0)
        var = np.var(x, axis=0)
        x_meaned = (x - mean)
        inv_std = 1 / np.sqrt(var + eps)
        x_normalied = x_meaned * inv_std
        out = gamma * x_normalied + beta
        running_mean = momentum * running_mean + (1 - momentum) * mean
        running_var = momentum * running_var + (1 - momentum) * var
        cache = (x_meaned, inv_std, gamma, beta, running_mean, running_var, eps)
    elif mode == "test":
        x_normalied = (x - running_mean) / np.sqrt(running_var + eps)
        out = x_normalied * gamma + beta
        cache = (gamma, beta, running_mean, running_var, eps)
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    (x_meaned, inv_std, gamma, beta, running_mean, running_var, eps) = cache

    N = x_meaned.shape[0]
    x_normalized = x_meaned *   inv_std
    dx_normalized = dout * gamma
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(x_normalized * dout, axis=0)
    dvar = np.sum(dx_normalized * x_meaned * -0.5 * ((inv_std) ** 3), axis=0)
    dmean = np.sum(dx_normalized * -inv_std, axis=0) + dvar * (np.sum(-2 * x_meaned, axis=0) / N)
    dx = dx_normalized * (inv_std) + dvar * (2 * x_meaned / N) + dmean / N

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.
    See the jupyter notebook for more hints.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    x_meaned, inv_std, gamma, beta, running_mean, running_var, eps = cache
    N, D = dout.shape

    x_norm = x_meaned * inv_std
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(x_norm * dout, axis=0)
    dx_hat = dout * gamma
    dx = (1 / N) * inv_std * (N * dx_hat - np.sum(dx_hat, axis=0) - x_norm * np.sum(dx_hat * x_norm, axis=0))

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.

    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get("eps", 1e-5)

    mean = np.mean(x, axis=1, keepdims=True)
    var = np.var(x, axis=1, keepdims=True)
    x_meaned = (x - mean)
    inv_std = 1 / np.sqrt(var + eps)
    x_normalied = x_meaned * inv_std
    out = gamma * x_normalied + beta
    cache = (x_meaned, inv_std, gamma, beta, eps)
    return out, cache


def layernorm_backward(dout, cache):
    """Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    x_meaned, inv_std, gamma, beta, eps = cache
    N,D=x_meaned.shape
    x_norm = x_meaned * inv_std
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(x_norm * dout, axis=0)
    dx_hat = dout * gamma
    dx = (1 / D) * inv_std * (D * dx_hat - np.sum(dx_hat, axis=1,keepdims=True) - x_norm * np.sum(dx_hat * x_norm, axis=1,keepdims=True))
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """Forward pass for inverted dropout.

    Note that this is different from the vanilla version of dropout.
    Here, p is the probability of keeping a neuron output, as opposed to
    the probability of dropping a neuron output.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None

    if mode == "train":
        mask = (np.random.rand(*x.shape) < p) / p
        out = x * mask
    elif mode == "test":
        out = x

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """Backward pass for inverted dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        dx = dout * mask
    elif mode == "test":
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """

    out = None
    pad=conv_param['pad']
    stride=conv_param['stride']
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    H_tag = 1 +(H+2*pad-HH)//stride
    W_tag = 1 +(W+2*pad-WW)//stride
    out=np.zeros((N,F,H_tag,W_tag))
    #x_padded=np.pad(x, ((0,0),(0,0),(pad,pad),(pad,pad)),mode='constant',constant_values=0)
    X_col = im2col_naive(x,HH,WW,pad,stride)
    kernel = w.reshape(F,-1)
    out_col = np.dot(kernel, X_col)+b.reshape(-1,1)
    # Reshape columns back to (N, F, H', W')
    out = out_col.reshape(F,H_tag,W_tag,N).transpose(3,0,1,2)

    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    x,w,b,conv_param=cache
    pad,stride=conv_param['pad'],conv_param['stride']
    N,C,H,W=x.shape
    F,_,HH,WW=w.shape
    H_tag=1+(H+2*pad-HH)//stride
    W_tag=1+(W+2*pad-WW)//stride
    dout_col=dout.transpose(1,2,3,0).reshape(F,-1)
    kernel_col=w.reshape(F,-1)
    x_col=im2col_naive(x,HH,WW,pad,stride)
    dw=(dout_col@x_col.T).reshape(w.shape)
    db = np.sum(dout_col, axis=1)
    dx=(kernel_col.T@dout_col)
    dx=col2im(dx,x.shape,HH,WW,pad,stride)
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here, eg you can assume:
      - (H - pool_height) % stride == 0
      - (W - pool_width) % stride == 0

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    N, C, H, W = x.shape
    H_tag = 1 + (H -pool_height) // stride
    W_tag = 1 + (W -pool_width) // stride
    out = np.zeros((N, C, H_tag, W_tag))


    for i in range(N):#iterate over images
        image=x[i]
        for j in range(C):
            for H_ in range(H_tag):
                for W_ in range(W_tag):
                    h_start=H_*stride
                    w_start=W_*stride
                    window=image[j,h_start:h_start+pool_height,w_start:w_start+pool_width]
                    max=np.max(window)
                    out[i,j,H_,W_]=max
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = np.zeros(cache[0].shape)
    x, pool_param = cache
    pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
    N, C, H, W = x.shape
    H_tag = 1 + (H - pool_height) // stride
    W_tag = 1 + (W - pool_width) // stride
    for i in range(N):#iterate over images
        image=x[i]
        for j in range(C):
            for H_ in range(H_tag):
                for W_ in range(W_tag):
                    h_start=H_*stride
                    w_start=W_*stride
                    window=image[j,h_start:h_start+pool_height,w_start:w_start+pool_width]
                    mask=np.argmax(window)
                    mask_index=np.unravel_index(mask,window.shape)

                    dx[i,j,h_start+mask_index[0],w_start+mask_index[1]]+=dout[i,j,H_,W_]
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None


    N, C, H, W = x.shape
    x_reshaped = x.transpose(0, 2, 3, 1).reshape(N * H * W, C)
    out_reshaped, cache = batchnorm_forward(x_reshaped, gamma, beta, bn_param)
    out = out_reshaped.reshape(N, H, W, C).transpose(0, 3, 1, 2)


    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    N, C, H, W = dout.shape
    dout_reshaped = dout.transpose(0, 2, 3, 1).reshape(N * H * W, C)
    dx_reshaped, dgamma, dbeta = batchnorm_backward(dout_reshaped, cache)
    dx = dx_reshaped.reshape(N, H, W, C).transpose(0, 3, 1, 2)

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """Computes the forward pass for spatial group normalization.

    In contrast to layer normalization, group normalization splits each entry in the data into G
    contiguous pieces, which it then normalizes independently. Per-feature shifting and scaling
    are then applied to the data, in a manner identical to that of batch normalization and layer
    normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (1, C, 1, 1)
    - beta: Shift parameter, of shape (1, C, 1, 1)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get("eps", 1e-5)

    N, C, H, W = x.shape

    x_group = x.reshape(N, G, C // G, H, W)
    mean = np.mean(x_group, axis=(2, 3, 4), keepdims=True)
    var = np.var(x_group, axis=(2, 3, 4), keepdims=True)
    x_norm = (x_group - mean) / np.sqrt(var + eps)
    x_norm = x_norm.reshape(N, C, H, W)
    out = gamma * x_norm + beta
    cache = (x_group, mean, var, gamma, beta, eps, G)

    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (1, C, 1, 1)
    - dbeta: Gradient with respect to shift parameter, of shape (1, C, 1, 1)
    """
    dx, dgamma, dbeta = None, None, None


    x_group, mean, var, gamma, beta, eps, G = cache
    N, C, H, W = dout.shape

    dbeta = np.sum(dout, axis=(0, 2, 3), keepdims=True)
    x_norm = (x_group - mean) / np.sqrt(var + eps)
    x_norm = x_norm.reshape(N, C, H, W)
    dgamma = np.sum(dout * x_norm, axis=(0, 2, 3), keepdims=True)
    dx_norm = dout * gamma
    dx_norm_group = dx_norm.reshape(N, G, C // G, H, W)
    M = (C // G) * H * W

    std = np.sqrt(var + eps)
    dvar = np.sum(dx_norm_group * (x_group - mean) * -0.5 * (std ** -3), axis=(2, 3, 4), keepdims=True)
    dmean = np.sum(dx_norm_group * -1.0 / std, axis=(2, 3, 4), keepdims=True) + \
            dvar * np.sum(-2.0 * (x_group - mean), axis=(2, 3, 4), keepdims=True) / M

    dx_group = dx_norm_group / std + dvar * 2.0 * (x_group - mean) / M + dmean / M
    dx = dx_group.reshape(N, C, H, W)

    return dx, dgamma, dbeta


def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


def print_mean_std(x, axis=0):
    print(f"  means: {x.mean(axis=axis)}")
    print(f"  stds:  {x.std(axis=axis)}\n")

