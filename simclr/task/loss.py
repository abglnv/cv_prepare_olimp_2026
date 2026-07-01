"""
NT-Xent: the normalized temperature-scaled cross-entropy loss (SimCLR Eq. 1).
                                                        ***TASK: fill in the TODO.***

A batch of N images gives two views each -> 2N projections. For each anchor, its positive
is the other view of the same image; the other 2N-2 projections are negatives. You L2-
normalize the projections (so dot product = cosine similarity), scale by 1/temperature,
mask out each row's self-similarity, and apply cross-entropy toward the positive index.
"""
import torch.nn as nn


class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        """z1, z2: [N, D] projections of the two views. Returns a scalar loss.

        TODO:
        """
        raise NotImplementedError("TODO: implement NT-Xent")
