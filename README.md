# 🧠 Deep Learning — Course Assignments

A collection of from-scratch and PyTorch implementations covering core deep learning topics, completed as part of my M.Sc. coursework.

## 📂 Repository Structure

```
assignments/
├── assignment1/   # Neural Network Fundamentals
├── assignment2/   # Sequence Models & Vision-Language
└── assignment3/   # Generative Models (Diffusion)
```

---

## Assignment 1 — Neural Network Fundamentals

Implements the building blocks of neural networks **from scratch in NumPy**, then transitions to PyTorch.

| Notebook | Topics |
|----------|--------|
| `01_softmax.ipynb` | Softmax classifier, cross-entropy loss, analytic gradients |
| `02_two_layer_net.ipynb` | Two-layer fully connected network, backpropagation |
| `03_FullyConnectedNets.ipynb` | Modular layer design, arbitrary-depth FC nets, SGD/Adam/RMSProp |
| `04_Dropout.ipynb` | Inverted dropout (forward & backward) |
| `05_BatchNormalization.ipynb` | Batch normalization, layer normalization, group normalization |
| `06_ConvolutionalNetworks.ipynb` | Convolution (im2col), max pooling, spatial batch norm |
| `07_PyTorch.ipynb` | PyTorch re-implementation of the above components |

**Key implementations** (`dl/`):

- Forward & backward passes for affine, ReLU, softmax, conv, pooling, dropout, and all normalization variants
- Modular `Solver` class with learning-rate scheduling
- Fully connected and convolutional network classifiers

---

## Assignment 2 — Sequence Models & Vision-Language

Implements RNNs, LSTMs, Transformers, and modern vision-language models in PyTorch.

| Notebook | Topics |
|----------|--------|
| `rnn_lstm_captioning.ipynb` | Vanilla RNN, LSTM, attention mechanism, image captioning on COCO |
| `Transformers.ipynb` | Transformer from scratch — self-attention, multi-head attention, encoder/decoder, positional encoding |
| `Self_Supervised_Learning.ipynb` | SimCLR contrastive learning framework |
| `CLIP_DINO.ipynb` | CLIP zero-shot classification, DINO self-supervised ViT |

**Key implementations**:

- `rnn_lstm_captioning.py` — RNN/LSTM cells, temporal softmax loss, `CaptioningRNN` with attention
- `transformers.py` — Scaled dot-product attention (loop & vectorized), `SelfAttention`, `MultiHeadAttention`, `LayerNormalization`, `FeedForwardBlock`, encoder/decoder blocks, full Transformer
- `dl/simclr/` — SimCLR contrastive loss, data augmentation pipeline
- `dl/clip_dino.py` — CLIP and DINO feature extraction & evaluation

---

## Assignment 3 — Generative Models (DDPM)

Implements Denoising Diffusion Probabilistic Models (DDPM) for conditional image generation.

| Notebook | Topics |
|----------|--------|
| `DDPM.ipynb` | Gaussian diffusion process, U-Net denoiser, classifier-free guidance, emoji generation |

**Key implementations** (`dl/`):

- `gaussian_diffusion.py` — Forward diffusion (q-sample), reverse sampling (p-sample), noise scheduling (linear / cosine / sigmoid), training loss
- `unet.py` — U-Net architecture with sinusoidal time embeddings, ResNet blocks, context conditioning, classifier-free guidance
- `ddpm_trainer.py` — Training loop with EMA

---

## 🛠 Tech Stack

- **NumPy** — from-scratch neural network layers (Assignment 1)
- **PyTorch** — all deep learning models (Assignments 1–3)
- **Torchvision** — pretrained feature extractors for captioning & CLIP/DINO
- **Jupyter Notebooks** — experiments, visualizations, and analysis

---

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/<your-username>/deep-learning-assignments.git
cd deep-learning-assignments

# Install dependencies (Assignment 3 has its own requirements.txt)
pip install numpy torch torchvision matplotlib jupyter

# Launch notebooks
jupyter notebook
```

> **Note**: Some notebooks download datasets (CIFAR-10, COCO) on first run.
