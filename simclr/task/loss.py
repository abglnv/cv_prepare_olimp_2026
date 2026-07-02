"""
NT-Xent: the normalized temperature-scaled cross-entropy loss (SimCLR Eq. 1).
                                                        ***TASK: fill in the TODO.***

A batch of N images gives two views each -> 2N projections. For each anchor, its positive
is the other view of the same image; the other 2N-2 projections are negatives. You L2-
normalize the projections (so dot product = cosine similarity), scale by 1/temperature,
mask out each row's self-similarity, and apply cross-entropy toward the positive index.
"""
import torch.nn as nn
import torch 
import torch.nn.functional as F

class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        """z1, z2: [N, D] projections of the two views. Returns a scalar loss.

        TODO:
        """
        N = z1.shape[0]

        z = torch.cat([z1, z2], dim=0)  #stack 
        z = F.normalize(z, dim=1) 

        sim = z @ z.T / self.temperature
        sim.fill_diagonal_(float("-inf"))  # kill yourself 

        targets = torch.cat([
            torch.arange(N, 2*N),  # rows 0..N-1  (A_i) -> B_i
            torch.arange(0, N),   # rows N..2N-1 (B_i) -> A_i
        ]).to(z.device)

        return F.cross_entropy(sim, targets)