# 🧠 Deep Learning — M.Sc. Coursework

> **From-scratch implementations of foundational and modern deep learning systems** — including Transformers, vision-language models (CLIP, DINO), diffusion models (DDPM), and contrastive self-supervised learning, built in NumPy and PyTorch.

---

## ✨ Highlights

| Area | What's implemented |
|---|---|
| 🔢 **Foundations** | Backprop, BatchNorm, Dropout, Conv — all from scratch in NumPy |
| 🔤 **Sequence & Attention** | Transformer from scratch, RNN/LSTM image captioning on COCO |
| 🖼️ **Vision-Language (Multimodal)** | CLIP zero-shot classification, DINO self-supervised ViT |
| 🎨 **Generative Models** | DDPM with classifier-free guidance, U-Net denoiser, EMA training |
| 🔍 **Self-Supervised Learning** | SimCLR contrastive loss + augmentation pipeline |

---

## 📂 Repository Structure

```
assignments/
├── assignment1/   # Neural Network Fundamentals (NumPy → PyTorch)
├── assignment2/   # Sequence Models & Vision-Language
└── assignment3/   # Generative Models — DDPM Diffusion
```

---

## Assignment 1 — Neural Network Fundamentals

Builds the core building blocks of neural networks **from scratch in NumPy**, then reimplements them in PyTorch.

| Notebook | Topics |
|----------|--------|
| `01_softmax.ipynb` | Softmax classifier, cross-entropy loss, analytic gradients |
| `02_two_layer_net.ipynb` | Two-layer fully connected network, backpropagation |
| `03_FullyConnectedNets.ipynb` | Modular layer design, arbitrary-depth FC nets, SGD / Adam / RMSProp |
| `04_Dropout.ipynb` | Inverted dropout (forward & backward passes) |
| `05_BatchNormalization.ipynb` | Batch norm, layer norm, group norm |
| `06_ConvolutionalNetworks.ipynb` | Convolution (im2col), max pooling, spatial batch norm |
| `07_PyTorch.ipynb` | PyTorch re-implementation of all the above |

**Key implementations** (`dl/`):
- Forward & backward passes for affine, ReLU, softmax, conv, pooling, dropout, and all normalization variants
- Modular `Solver` class with pluggable optimizers and learning-rate scheduling
- Fully connected and convolutional network classifiers

---

## Assignment 2 — Sequence Models & Vision-Language (Multimodal)

Implements the full spectrum from recurrent models to modern vision-language architectures.

| Notebook | Topics |
|----------|--------|
| `rnn_lstm_captioning.ipynb` | Vanilla RNN, LSTM, attention mechanism — image captioning on COCO |
| `Transformers.ipynb` | **Transformer from scratch** — self-attention, multi-head attention, encoder/decoder, positional encoding |
| `Self_Supervised_Learning.ipynb` | SimCLR contrastive learning framework |
| `CLIP_DINO.ipynb` | **CLIP zero-shot classification**, DINO self-supervised ViT feature extraction |

**Key implementations**:
- `rnn_lstm_captioning.py` — RNN/LSTM cells, temporal softmax loss, `CaptioningRNN` with attention
- `transformers.py` — Scaled dot-product attention (loop & vectorized), `SelfAttention`, `MultiHeadAttention`, `LayerNorm`, `FeedForwardBlock`, full encoder/decoder Transformer
- `dl/simclr/` — SimCLR contrastive loss and data augmentation pipeline
- `dl/clip_dino.py` — CLIP and DINO feature extraction, zero-shot evaluation

> 💡 **CLIP** is a multimodal model that jointly embeds images and text — this notebook covers zero-shot image classification using OpenAI's pretrained CLIP. **DINO** uses self-supervised ViT representations without labels.

---

## Assignment 3 — Generative Models (DDPM)

Full implementation of Denoising Diffusion Probabilistic Models for **conditional image generation**.

| Notebook | Topics |
|----------|--------|
| `DDPM.ipynb` | Gaussian diffusion process, U-Net denoiser, classifier-free guidance, emoji generation |

**Key implementations** (`dl/`):
- `gaussian_diffusion.py` — Forward diffusion (q-sample), reverse sampling (p-sample), noise scheduling (linear / cosine / sigmoid), training loss
- `unet.py` — U-Net with sinusoidal time embeddings, ResNet blocks, context conditioning, **classifier-free guidance**
- `ddpm_trainer.py` — Full training loop with **EMA (Exponential Moving Average)**

---

## 🛠 Tech Stack

| Tool | Usage |
|------|-------|
| **NumPy** | From-scratch neural network layers (Assignment 1) |
| **PyTorch** | All deep learning models (Assignments 1–3) |
| **Torchvision** | Pretrained feature extractors for captioning & CLIP/DINO |
| **Jupyter Notebooks** | Experiments, visualizations, and analysis |

---

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/TalSomech/Deep-Learning-Assignments.git
cd Deep-Learning-Assignments

# Install core dependencies
pip install numpy torch torchvision matplotlib jupyter

# Assignment 3 has additional requirements
pip install -r assignment3/requirements.txt

# Launch notebooks
jupyter notebook
```

> **Note**: Some notebooks download datasets (CIFAR-10, COCO) on first run.
