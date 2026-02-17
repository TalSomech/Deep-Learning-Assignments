import math
from typing import Optional, Tuple

import torch
import torchvision
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import functional as F
from torchvision.models import feature_extraction
import numpy as np

from a2_helper import load_coco_captions, train_captioner, decode_captions
from dl.grad import rel_error, compute_numeric_gradient
from dl.utils import attention_visualizer


def hello_rnn_lstm_captioning():
    print("Hello from rnn_lstm_captioning.py!")


class ImageEncoder(nn.Module):
    """
    Convolutional network that accepts images as input and outputs their spatial
    grid features. This module servesx as the image encoder in image captioning
    model. We will use a tiny RegNet-X 400MF model that is initialized with
    ImageNet-pretrained weights from Torchvision library.

    NOTE: We could use any convolutional network/image encoder architecture, but we opt for a
    tiny RegNet model so it can train decently with a single Colab GPU.
    """

    def __init__(self, pretrained: bool = True, verbose: bool = True):
        """
        Args:
            pretrained: Whether to initialize this model with pretrained weights
                from Torchvision library.
            verbose: Whether to log expected output shapes during instantiation.
        """
        super().__init__()
        self.cnn = torchvision.models.regnet_x_400mf(pretrained=pretrained)

        # Torchvision models return global average pooled features by default.
        # Our attention-based models may require spatial grid features. So we
        # wrap the ConvNet with torchvision's feature extractor. We will get
        # the spatial features right before the final classification layer.
        self.backbone = feature_extraction.create_feature_extractor(
            self.cnn, return_nodes={"trunk_output.block4": "c5"}
        )
        # We call these features "c5", a name that may sound familiar from the
        # object detection assignment. :-)

        # Pass a dummy batch of input images to infer output shape.
        dummy_out = self.backbone(torch.randn(2, 3, 224, 224))["c5"]
        self._out_channels = dummy_out.shape[1]

        if verbose:
            print("For input images in NCHW format, shape (2, 3, 224, 224)")
            print(f"Shape of output c5 features: {dummy_out.shape}")

        # Input image batches are expected to be float tensors in range [0, 1].
        # However, the backbone here expects these tensors to be normalized by
        # ImageNet color mean/std (as it was trained that way).
        # We define a function to transform the input images before extraction:
        self.normalize = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    @property
    def out_channels(self):
        """
        Number of output channels in extracted image features. You may access
        this value freely to define more modules to go with this encoder.
        """
        return self._out_channels

    def forward(self, images: torch.Tensor):
        # Input images may be uint8 tensors in [0-255], change them to float
        # tensors in [0-1]. Get float type from backbone (could be float32/64).
        if images.dtype == torch.uint8:
            images = images.to(dtype=self.cnn.stem[0].weight.dtype)
            images /= 255.0

        # Normalize images by ImageNet color mean/std.
        images = self.normalize(images)

        # Extract c5 features from encoder (backbone) and return.
        # shape: (B, out_channels, H / 32, W / 32)
        features = self.backbone(images)["c5"]
        return features


# Recurrent Neural Network
def rnn_step_forward(x, prev_h, Wx, Wh, b):
    """
    Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
    activation function.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Args:
        x: Input data for this timestep, of shape (N, D).
        prev_h: Hidden state from previous timestep, of shape (N, H)
        Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
        Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
        b: Biases, of shape (H,)

    Returns a tuple of:
        next_h: Next hidden state, of shape (N, H)
        cache: Tuple of values needed for the backward pass.
    """
    next_h, cache = None, None
    # Replace "pass" statement with your code
    dot_wh = prev_h @ Wh
    dot_wx = x @ Wx
    a = dot_wh + dot_wx + b
    next_h = torch.tanh(a)

    cache = (x, prev_h, Wx, Wh, next_h)
    return next_h, cache


def rnn_step_backward(dnext_h, cache):
    """
    Backward pass for a single timestep of a vanilla RNN.

    Args:
        dnext_h: Gradient of loss with respect to next hidden state, of shape (N, H)
        cache: Cache object from the forward pass

    Returns a tuple of:
        dx: Gradients of input data, of shape (N, D)
        dprev_h: Gradients of previous hidden state, of shape (N, H)
        dWx: Gradients of input-to-hidden weights, of shape (D, H)
        dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
        db: Gradients of bias vector, of shape (H,)
    """
    dx, dprev_h, dWx, dWh, db = None, None, None, None, None
    # Replace "pass" statement with your code
    x, prev_h, Wx, Wh, next_h = cache
    dtanh = dnext_h * (1 - next_h ** 2)
    dx = dtanh @ Wx.T
    dprev_h = dtanh @ Wh.T
    dWx = x.T @ dtanh
    dWh = prev_h.T @ dtanh
    db = dtanh.sum(axis=0)

    # dprev_h=
    return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
    """
    Run a vanilla RNN forward on an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The RNN uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the RNN forward, we return the hidden states for all timesteps.

    Args:
        x: Input data for the entire timeseries, of shape (N, T, D).
        h0: Initial hidden state, of shape (N, H)
        Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
        Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
        b: Biases, of shape (H,)

    Returns a tuple of:
        h: Hidden states for the entire timeseries, of shape (N, T, H).
        cache: Values needed in the backward pass
    """
    h, cache = None, []
    # Replace "pass" statement with your code
    N, T, D = x.shape
    _, H = h0.shape
    prev_h = h0
    h_list = []
    for t in range(T):
        token = x[:, t, :]
        next_h, curr_cache = rnn_step_forward(token, prev_h, Wx, Wh, b)
        cache.append(curr_cache)
        h_list.append(next_h)
        prev_h = next_h
    h = torch.stack(h_list, dim=1)
    return h, cache


def rnn_backward(dh, cache):
    """
    Compute the backward pass for a vanilla RNN over an entire sequence of data.

    Args:
        dh: Upstream gradients of all hidden states, of shape (N, T, H).

    NOTE: 'dh' contains the upstream gradients produced by the
    individual loss functions at each timestep, *not* the gradients
    being passed between timesteps (which you'll have to compute yourself
    by calling rnn_step_backward in a loop).

    Returns a tuple of:
        dx: Gradient of inputs, of shape (N, T, D)
        dh0: Gradient of initial hidden state, of shape (N, H)
        dWx: Gradient of input-to-hidden weights, of shape (D, H)
        dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
        db: Gradient of biases, of shape (H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    # Replace "pass" statement with your code
    N, T, H = dh.shape
    x, _, _, _, _ = cache[0]
    _, D = x.shape

    dx = torch.zeros((N, T, D), **to_double)
    dh0 = torch.zeros((N, H), **to_double)
    dWx = torch.zeros((D, H), **to_double)
    dWh = torch.zeros((H, H), **to_double)
    db = torch.zeros(H, **to_double)
    for t in range(T - 1, -1, -1):
        curr_cache = cache[t]
        curr_dh = dh[:, t, :] + dh0
        dx[:, t, :], dh0, dWx_t, dWh_t, db_t = rnn_step_backward(curr_dh, curr_cache)
        dWx += dWx_t
        dWh += dWh_t
        db += db_t
    return dx, dh0, dWx, dWh, db


class RNN(nn.Module):
    """
    Single-layer vanilla RNN module.

    You don't have to implement anything here but it is highly recommended to
    read through the code as you will implement subsequent modules.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        """
        Initialize an RNN. Model parameters to initialize:
            Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
            Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
            b: Biases, of shape (H,)

        Args:
            input_dim: Input size, denoted as D before
            hidden_dim: Hidden size, denoted as H before
        """
        super().__init__()

        # Register parameters
        self.Wx = nn.Parameter(
            torch.randn(input_dim, hidden_dim).div(math.sqrt(input_dim))
        )
        self.Wh = nn.Parameter(
            torch.randn(hidden_dim, hidden_dim).div(math.sqrt(hidden_dim))
        )
        self.b = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, x, h0):
        """
        Args:
            x: Input data for the entire timeseries, of shape (N, T, D)
            h0: Initial hidden state, of shape (N, H)

        Returns:
            hn: The hidden state output
        """
        hn, _ = rnn_forward(x, h0, self.Wx, self.Wh, self.b)
        return hn

    def step_forward(self, x, prev_h):
        """
        Args:
            x: Input data for one time step, of shape (N, D)
            prev_h: The previous hidden state, of shape (N, H)

        Returns:
            next_h: The next hidden state, of shape (N, H)
        """
        next_h, _ = rnn_step_forward(x, prev_h, self.Wx, self.Wh, self.b)
        return next_h


class WordEmbedding(nn.Module):
    """
    Simplified version of torch.nn.Embedding.

    We operate on minibatches of size N where
    each sequence has length T. We assume a vocabulary of V words, assigning each
    word to a vector of dimension D.

    Args:
        x: Integer array of shape (N, T) giving indices of words. Each element idx
      of x muxt be in the range 0 <= idx < V.

    Returns a tuple of:
        out: Array of shape (N, T, D) giving word vectors for all input words.
    """

    def __init__(self, vocab_size: int, embed_size: int):
        super().__init__()

        # Register parameters
        self.W_embed = nn.Parameter(
            torch.randn(vocab_size, embed_size).div(math.sqrt(vocab_size))
        )

    def forward(self, x):
        out = None
        # Replace "pass" statement with your code
        out = self.W_embed[x]

        return out


def temporal_softmax_loss(x, y, ignore_index=None):
    """
    A temporal version of softmax loss for use in RNNs. We assume that we are
    making predictions over a vocabulary of size V for each timestep of a
    timeseries of length T, over a minibatch of size N. The input x gives scores
    for all vocabulary elements at all timesteps, and y gives the indices of the
    ground-truth element at each timestep. We use a cross-entropy loss at each
    timestep, *summing* the loss over all timesteps and *averaging* across the
    minibatch.

    As an additional complication, we may want to ignore the model output at some
    timesteps, since sequences of different length may have been combined into a
    minibatch and padded with NULL tokens. The optional ignore_index argument
    tells us which elements in the caption should not contribute to the loss.

    Args:
        x: Input scores, of shape (N, T, V)
        y: Ground-truth indices, of shape (N, T) where each element is in the
            range 0 <= y[i, t] < V

    Returns a tuple of:
        loss: Scalar giving loss
    """
    loss = None

    # Replace "pass" statement with your code
    loss = F.cross_entropy(x.view(-1, x.shape[2]), y.reshape(-1), ignore_index=ignore_index, reduction='sum') / len(x)

    return loss


class CaptioningRNN(nn.Module):
    """
    A CaptioningRNN produces captions from images using a recurrent
    neural network.

    The RNN receives input vectors of size D, has a vocab size of V, works on
    sequences of length T, has an RNN hidden dimension of H, uses word vectors
    of dimension W, and operates on minibatches of size N.

    Note that we don't use any regularization for the CaptioningRNN.

    You will implement the `__init__` method for model initialization and
    the `forward` method first, then come back for the `sample` method later.
    """

    def __init__(
            self,
            word_to_idx,
            input_dim: int = 512,
            wordvec_dim: int = 128,
            hidden_dim: int = 128,
            cell_type: str = "rnn",
            image_encoder_pretrained: bool = True,
            ignore_index: Optional[int] = None,
    ):
        """
        Construct a new CaptioningRNN instance.

        Args:
            word_to_idx: A dictionary giving the vocabulary. It contains V
                entries, and maps each string to a unique integer in the
                range [0, V).
            input_dim: Dimension D of input image feature vectors.
            wordvec_dim: Dimension W of word vectors.
            hidden_dim: Dimension H for the hidden state of the RNN.
            cell_type: What type of RNN to use; either 'rnn' or 'lstm'.
        """
        super().__init__()
        if cell_type not in {"rnn", "lstm", "attn"}:
            raise ValueError('Invalid cell_type "%s"' % cell_type)

        self.cell_type = cell_type
        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.items()}

        vocab_size = len(word_to_idx)
        self.hidden_dim = hidden_dim
        self._null = word_to_idx["<NULL>"]
        self._start = word_to_idx.get("<START>", None)
        self._end = word_to_idx.get("<END>", None)
        self.ignore_index = ignore_index
        self.model = None
        # Replace "pass" statement with your code
        self.W_embed = WordEmbedding(vocab_size=vocab_size, embed_size=wordvec_dim)
        if self.cell_type == 'rnn':
            self.model = RNN(input_dim=wordvec_dim, hidden_dim=hidden_dim)
        elif self.cell_type == 'lstm':
            self.model = LSTM(input_dim=wordvec_dim, hidden_dim=hidden_dim)
        elif self.cell_type == 'attn':
            self.model = AttentionLSTM(input_dim=wordvec_dim, hidden_dim=hidden_dim)
        self.image_encoder = ImageEncoder(pretrained=image_encoder_pretrained)
        self.im_to_h0 = nn.Linear(input_dim, hidden_dim)
        self.temporal_affine = nn.Linear(hidden_dim, vocab_size)

    def forward(self, images, captions):
        """
        Compute training-time loss for the RNN. We input images and the GT
        captions for those images, and use an RNN (or LSTM) to compute loss. The
        backward part will be done by torch.autograd.

        Args:
            images: Input images, of shape (N, 3, 112, 112)
            captions: Ground-truth captions; an integer array of shape (N, T + 1)
                where each element is in the range 0 <= y[i, t] < V

        Returns:
            loss: A scalar loss
        """
        # Cut captions into two pieces: captions_in has everything but the last
        # word and will be input to the RNN; captions_out has everything but the
        # first word and this is what we will expect the RNN to generate. These
        # are offset by one relative to each other because the RNN should produce
        # word (t+1) after receiving word t. The first element of captions_in
        # will be the START token, and the first element of captions_out will
        # be the first word.
        captions_in = captions[:, :-1]
        captions_out = captions[:, 1:]

        loss = 0.0
        # Replace "pass" statement with your code
        if self.cell_type=="attn":
            im_features = self.image_encoder(images)
            h0=self.im_to_h0(im_features.permute(0,2,3,1)).permute(0,3,1,2)
        else:
            im_features = self.image_encoder(images).mean(dim=(2, 3))
            h0 = self.im_to_h0(im_features) #(1)
        word_embeddings = self.W_embed(captions_in) #(2)
        hn = self.model(word_embeddings, h0) #(3)
        N, T, _ = hn.shape
        temp_scores = self.temporal_affine(hn.view(-1, hn.shape[2])).reshape(N, T, len(self.word_to_idx))#(4)
        loss = temporal_softmax_loss(temp_scores, captions_out, self.ignore_index)#(5)

        return loss

    def sample(self, images, max_length=15):
        """
        Run a test-time forward pass for the model, sampling captions for input
        feature vectors.

        At each timestep, we embed the current word, pass it and the previous hidden
        state to the RNN to get the next hidden state, use the hidden state to get
        scores for all vocab words, and choose the word with the highest score as
        the next word. The initial hidden state is computed by applying an affine
        transform to the image features, and the initial word is the <START>
        token.

        For LSTMs you will also have to keep track of the cell state; in that case
        the initial cell state should be zero.

        Args:
            images: Input images, of shape (N, 3, 112, 112)
            max_length: Maximum length T of generated captions

        Returns:
            captions: Array of shape (N, max_length) giving sampled captions,
                where each element is an integer in the range [0, V). The first
                element of captions should be the first sampled word, not the
                <START> token.
        """
        N = images.shape[0]
        captions = self._null * images.new(N, max_length).fill_(1).long()

        if self.cell_type == "attn":
            attn_weights_all = images.new(N, max_length, 4, 4).fill_(0).float()
            ct = torch.zeros((N, self.hidden_dim), **to_float)
        elif self.cell_type == "lstm":
            ct = torch.zeros((N, self.hidden_dim),**to_float)
        # Replace "pass" statement with your code
        if self.cell_type=='attn':
            im_features = self.image_encoder(images)
            projected = self.im_to_h0(im_features.permute(0,2,3,1)).permute(0,3,1,2)
            h0=projected.mean(dim=(2,3))
            ct=h0
        else:
            im_features = self.image_encoder(images).mean(dim=(2, 3))
            h0 = self.im_to_h0(im_features)
        ht = h0
        prev_word=self._start * images.new(N,).fill_(1).long()
        for t in range(max_length):
            curr_embed = self.W_embed(prev_word)
            if self.cell_type == 'rnn':
                ht = self.model.step_forward(curr_embed, ht)
            elif self.cell_type == 'lstm':
                ht,ct=self.model.step_forward(curr_embed,ht,ct)
            elif self.cell_type == 'attn':
                attn, attn_weights = dot_product_attention(ht, projected)
                attn_weights_all[:,t]=attn_weights
                ht,ct=self.model.step_forward(curr_embed,ht,ct,attn)
            scores_t = self.temporal_affine(ht)
            curr_word=torch.argmax(scores_t,dim=1)
            captions[:, t] = curr_word
            prev_word=curr_word
        if self.cell_type == "attn":
            return captions, attn_weights_all.cpu()
        else:
            return captions

    def save(self, path: str):
        """
        Save the model checkpoint.

        Args:
            path (str): Path to save the checkpoint (.pt or .pth)
        """
        checkpoint = {
            "state_dict": self.state_dict(),
            "config": {
                "input_dim": self.im_to_h0.in_features,
                "wordvec_dim": self.W_embed.W_embed.shape[1],
                "hidden_dim": self.rnn.Wh.shape[0],
                "cell_type": self.cell_type,
                "ignore_index": self.ignore_index,
            },
            "word_to_idx": self.word_to_idx,
        }

        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path: str, image_encoder_pretrained: bool = True):
        """
        Load a CaptioningRNN model from a checkpoint.

        Args:
            path (str): Path to checkpoint
            image_encoder_pretrained (bool): Whether to load pretrained CNN weights

        Returns:
            CaptioningRNN: Loaded model
        """
        checkpoint = torch.load(path, map_location="cpu")

        config = checkpoint["config"]
        word_to_idx = checkpoint["word_to_idx"]

        model = cls(
            word_to_idx=word_to_idx,
            input_dim=config["input_dim"],
            wordvec_dim=config["wordvec_dim"],
            hidden_dim=config["hidden_dim"],
            cell_type=config["cell_type"],
            image_encoder_pretrained=image_encoder_pretrained,
            ignore_index=config["ignore_index"],
        )

        model.load_state_dict(checkpoint["state_dict"])
        model.eval()  # default to eval mode

        return model


class LSTM(nn.Module):
    """Single-layer, uni-directional LSTM module."""

    def __init__(self, input_dim: int, hidden_dim: int):
        """
        Initialize a LSTM. Model parameters to initialize:
            Wx: Weights for input-to-hidden connections, of shape (D, 4H)
            Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
            b: Biases, of shape (4H,)

        Args:
            input_dim: Input size, denoted as D before
            hidden_dim: Hidden size, denoted as H before
        """
        super().__init__()

        # Register parameters
        self.Wx = nn.Parameter(
            torch.randn(input_dim, hidden_dim * 4).div(math.sqrt(input_dim))
        )
        self.Wh = nn.Parameter(
            torch.randn(hidden_dim, hidden_dim * 4).div(math.sqrt(hidden_dim))
        )
        self.b = nn.Parameter(torch.zeros(hidden_dim * 4))

    def step_forward(
            self, x: torch.Tensor, prev_h: torch.Tensor, prev_c: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for a single timestep of an LSTM.
        The input data has dimension D, the hidden state has dimension H, and
        we use a minibatch size of N.

        Args:
            x: Input data for one time step, of shape (N, D)
            prev_h: The previous hidden state, of shape (N, H)
            prev_c: The previous cell state, of shape (N, H)
            Wx: Input-to-hidden weights, of shape (D, 4H)
            Wh: Hidden-to-hidden weights, of shape (H, 4H)
            b: Biases, of shape (4H,)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]
                next_h: Next hidden state, of shape (N, H)
                next_c: Next cell state, of shape (N, H)
        """
        next_h, next_c = None, None
        N, H = prev_h.shape
        # Replace "pass" statement with your code
        a = x @ self.Wx + prev_h @ self.Wh + self.b
        a_i = a[:, :H]
        a_f = a[:, H:2 * H]
        a_o = a[:, 2 * H:3 * H]
        a_g = a[:, 3 * H:]
        i = torch.sigmoid(a_i)
        f = torch.sigmoid(a_f)
        o = torch.sigmoid(a_o)
        g = torch.tanh(a_g)
        next_c = f * prev_c + i * g
        next_h = o * torch.tanh(next_c)
        return next_h, next_c

    def forward(self, x: torch.Tensor, h0: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for an LSTM over an entire sequence of data. We assume an
        input sequence composed of T vectors, each of dimension D. The LSTM
        uses a hidden size of H, and we work over a minibatch containing N
        sequences. After running the LSTM forward, we return the hidden states
        for all timesteps.

        Note that the initial cell state is passed as input, but the initial
        cell state is set to zero. Also note that the cell state is not returned;
        it is an internal variable to the LSTM and is not accessed from outside.

        Args:
            x: Input data for the entire timeseries, of shape (N, T, D)
            h0: Initial hidden state, of shape (N, H)

        Returns:
            hn: The hidden state output.
        """

        c0 = torch.zeros_like(
            h0
        )  # we provide the intial cell state c0 here for you!
        hn = None
        prev_h = h0
        prev_c = c0
        h_list = []
        N, T, D = x.shape
        for t in range(T):
            token = x[:, t, :]
            next_h, next_c = self.step_forward(token, prev_h, prev_c)
            h_list.append(next_h)
            prev_h = next_h
            prev_c = next_c
        hn = torch.stack(h_list, dim=1)
        # Replace "pass" statement with your code

        return hn


def dot_product_attention(prev_h, A):
    """
    A simple scaled dot-product attention layer.

    Args:
        prev_h: The LSTM hidden state from previous time step, of shape (N, H)
        A: **Projected** CNN feature activation, of shape (N, H, 4, 4),
         where H is the LSTM hidden state size

    Returns:
        attn: Attention embedding output, of shape (N, H)
        attn_weights: Attention weights, of shape (N, 4, 4)

    """
    N, H, D_a, _ = A.shape

    attn, attn_weights = None, None
    # Replace "pass" statement with your codes

    A_t=A.flatten(start_dim=2)
    attn_weights=((prev_h.unsqueeze(1)@A_t).squeeze(1))/math.sqrt(H)
    attn_weights=torch.softmax(attn_weights,dim=1)
    attn=(A_t@attn_weights.unsqueeze(-1)).squeeze(-1)
    attn_weights=attn_weights.reshape(N,D_a,D_a)

    return attn, attn_weights


class AttentionLSTM(nn.Module):
    """
    This is our single-layer, uni-directional Attention module.

    Args:
        input_dim: Input size, denoted as D before
        hidden_dim: Hidden size, denoted as H before
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        """
        Initialize a LSTM. Model parameters to initialize:
            Wx: Weights for input-to-hidden connections, of shape (D, 4H)
            Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
            Wattn: Weights for attention-to-hidden connections, of shape (H, 4H)
            b: Biases, of shape (4H,)
        """
        super().__init__()

        # Register parameters
        self.Wx = nn.Parameter(
            torch.randn(input_dim, hidden_dim * 4).div(math.sqrt(input_dim))
        )
        self.Wh = nn.Parameter(
            torch.randn(hidden_dim, hidden_dim * 4).div(math.sqrt(hidden_dim))
        )
        self.Wattn = nn.Parameter(
            torch.randn(hidden_dim, hidden_dim * 4).div(math.sqrt(hidden_dim))
        )
        self.b = nn.Parameter(torch.zeros(hidden_dim * 4))

    def step_forward(
            self,
            x: torch.Tensor,
            prev_h: torch.Tensor,
            prev_c: torch.Tensor,
            attn: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input data for one time step, of shape (N, D)
            prev_h: The previous hidden state, of shape (N, H)
            prev_c: The previous cell state, of shape (N, H)
            attn: The attention embedding, of shape (N, H)

        Returns:
            next_h: The next hidden state, of shape (N, H)
            next_c: The next cell state, of shape (N, H)
        """

        next_h, next_c = None, None
        N, H = prev_h.shape
        # Replace "pass" statement with your code
        a = x @ self.Wx + prev_h @ self.Wh+attn@self.Wattn + self.b
        a_i = a[:, :H]
        a_f = a[:, H:2 * H]
        a_o = a[:, 2 * H:3 * H]
        a_g = a[:, 3 * H:]
        i = torch.sigmoid(a_i)
        f = torch.sigmoid(a_f)
        o = torch.sigmoid(a_o)
        g = torch.tanh(a_g)
        next_c = f * prev_c + i * g
        next_h = o * torch.tanh(next_c)

        return next_h, next_c

    def forward(self, x: torch.Tensor, A: torch.Tensor):
        """
        Forward pass for an LSTM over an entire sequence of data. We assume an
        input sequence composed of T vectors, each of dimension D. The LSTM uses
        a hidden size of H, and we work over a minibatch containing N sequences.
        After running the LSTM forward, we return hidden states for all timesteps.

        Note that the initial cell state is passed as input, but the initial cell
        state is set to zero. Also note that the cell state is not returned; it
        is an internal variable to the LSTM and is not accessed from outside.

        h0 and c0 are same initialized as the global image feature (meanpooled A)
        For simplicity, we implement scaled dot-product attention, which means in
        Eq. 4 of the paper (https://arxiv.org/pdf/1502.03044.pdf),
        f_{att}(a_i, h_{t-1}) equals to the scaled dot product of a_i and h_{t-1}.

        Args:
            x: Input data for the entire timeseries, of shape (N, T, D)
            A: The projected CNN feature activation, of shape (N, H, 4, 4)

        Returns:
            hn: The hidden state output
        """

        # The initial hidden state h0 and cell state c0 are initialized
        # differently in AttentionLSTM from the original LSTM and hence
        # we provided them for you.
        h0 = A.mean(dim=(2, 3))  # Initial hidden state, of shape (N, H)
        c0 = h0  # Initial cell state, of shape (N, H)

        hn = None
        prev_h = h0
        prev_c = c0
        h_list = []
        N, T, D = x.shape
        for t in range(T):
            token = x[:, t, :]
            attn, attn_weights = dot_product_attention(prev_h, A)
            next_h, next_c = self.step_forward(token, prev_h, prev_c,attn)
            h_list.append(next_h)
            prev_h = next_h
            prev_c = next_c
        hn = torch.stack(h_list, dim=1)
        # Replace "pass" statement with your code
        return hn


