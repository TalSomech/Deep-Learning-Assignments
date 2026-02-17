import torch
import numpy as np


def sim(z_i, z_j):
    """Normalized dot product between two vectors.

    Inputs:
    - z_i: 1xD tensor.
    - z_j: 1xD tensor.

    Returns:
    - A scalar value that is the normalized dot product between z_i and z_j.
    """
    norm_dot_product = None
    dot_product = torch.dot(z_i.flatten(), z_j.flatten())
    norm_z_i = torch.linalg.norm(z_i)
    norm_z_j = torch.linalg.norm(z_j)
    norm_dot_product = dot_product / (norm_z_i * norm_z_j)

    return norm_dot_product


def simclr_loss_naive(out_left, out_right, tau):
    """Compute the contrastive loss L over a batch (naive loop version).

    Input:
    - out_left: NxD tensor; output of the projection head g(), left branch in SimCLR model.
    - out_right: NxD tensor; output of the projection head g(), right branch in SimCLR model.
    Each row is a z-vector for an augmented sample in the batch. The same row in out_left and out_right form a positive pair.
    In other words, (out_left[k], out_right[k]) form a positive pair for all k=0...N-1.
    - tau: scalar value, temperature parameter that determines how fast the exponential increases.

    Returns:
    - A scalar value; the total loss across all positive pairs in the batch. See notebook for definition.
    """
    N = out_left.shape[0]  # total number of training examples

     # Concatenate out_left and out_right into a 2*N x D tensor.
    out = torch.cat([out_left, out_right], dim=0)  # [2*N, D]

    total_loss = 0
    for k in range(N):  # loop through each positive pair (k, k+N)
        z_k, z_k_N = out[k], out[k+N]

        # Similarities for z_k
        sim_z_k = torch.tensor([sim(z_k, out[i]) for i in range(2*N)])
        exp_sim_z_k = torch.exp(sim_z_k / tau)

        # Similarities for z_k_N
        sim_z_k_N = torch.tensor([sim(z_k_N, out[i]) for i in range(2 * N)])
        exp_sim_z_k_N = torch.exp(sim_z_k_N / tau)

        mask_k = torch.ones(2 * N, dtype=torch.bool)
        mask_k[k] = False
        l_k_k_N = -torch.log(
            exp_sim_z_k[k+N]/
            torch.sum(exp_sim_z_k[mask_k])
        )

        mask_k_N = torch.ones(2 * N, dtype=torch.bool)
        mask_k_N[k + N] = False
        l_k_N_k = -torch.log(
            exp_sim_z_k_N[k] /
            torch.sum(exp_sim_z_k_N[mask_k_N])
        )

        total_loss += l_k_k_N + l_k_N_k

    # In the end, we need to divide the total loss by 2N, the number of samples in the batch.
    total_loss = total_loss / (2*N)
    return total_loss


def sim_positive_pairs(out_left, out_right):
    """Normalized dot product between positive pairs.

    Inputs:
    - out_left: NxD tensor; output of the projection head g(), left branch in SimCLR model.
    - out_right: NxD tensor; output of the projection head g(), right branch in SimCLR model.
    Each row is a z-vector for an augmented sample in the batch.
    The same row in out_left and out_right form a positive pair.

    Returns:
    - A Nx1 tensor; each row k is the normalized dot product between out_left[k] and out_right[k].
    """
    pos_pairs = None

    dot_products = (out_left * out_right).sum(dim=1, keepdim=True)
    norm_left = torch.linalg.norm(out_left, dim=1, keepdim=True)
    norm_right = torch.linalg.norm(out_right, dim=1, keepdim=True)

    pos_pairs = dot_products / (norm_left * norm_right)

    return pos_pairs


def compute_sim_matrix(out):
    """Compute a 2N x 2N matrix of normalized dot products between all pairs of augmented examples in a batch.

    Inputs:
    - out: 2N x D tensor; each row is the z-vector (output of projection head) of a single augmented example.
    There are a total of 2N augmented examples in the batch.

    Returns:
    - sim_matrix: 2N x 2N tensor; each element i, j in the matrix is the normalized dot product between out[i] and out[j].
    """
    sim_matrix = None

    out_norm = out / torch.linalg.norm(out, dim=1, keepdim=True)

    sim_matrix = torch.matmul(out_norm, out_norm.T)

    return sim_matrix


def simclr_loss_vectorized(out_left, out_right, tau, device='cuda'):
    """Compute the contrastive loss L over a batch (vectorized version). No loops are allowed.

    Inputs and output are the same as in simclr_loss_naive.
    """
    N = out_left.shape[0]

    # Concatenate out_left and out_right into a 2*N x D tensor.
    out = torch.cat([out_left, out_right], dim=0)  # [2*N, D]

    # Compute similarity matrix between all pairs of augmented examples in the batch.
    sim_matrix = compute_sim_matrix(out)  # [2*N, 2*N]

    exponential = torch.exp(sim_matrix / tau)

    # This binary mask zeros out terms where k=i.
    mask = (torch.ones_like(exponential, device=device) - torch.eye(2 * N, device=device)).to(device).bool()

    # We apply the binary mask.
    exponential = exponential.masked_select(mask).view(2 * N, -1)  # [2*N, 2*N-1]

    # Hint: Compute the denominator values for all augmented samples. This should be a 2N x 1 vector.
    denom = torch.sum(exponential, dim=1, keepdim=True)

    sim_pos_lr = sim_positive_pairs(out_left, out_right)
    sim_pos_rl = sim_positive_pairs(out_right, out_left)
    sim_pos = torch.cat([sim_pos_lr, sim_pos_rl], dim=0)

    numerator = torch.exp(sim_pos / tau)

    loss = torch.sum(-torch.log(numerator / denom)) / (2 * N)

    return loss


def rel_error(x,y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

