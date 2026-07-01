"""
kNN evaluation of the learned representation (no labels used during training).
                                                        ***TASK: fill in the TODOs.***

Build a memory bank of L2-normalized features h = f(x) over the train set, then for each
test image retrieve its k nearest neighbours by cosine similarity and predict the label by
a temperature-weighted vote. This is the standard weighted-kNN monitor (Wu et al., 2018).
"""
import torch


@torch.no_grad()
def _extract_features(model, loader, device):
    """L2-normalized features and labels for every image in `loader`.

    TODO: set model.eval(); for each (x, y) run h = model.features(x) on `device`,
    L2-normalize h (dim=1), collect h and y; return (feats [M, D], labels [M]).
    """
    raise NotImplementedError("TODO: extract normalized features + labels")


@torch.no_grad()
def knn_evaluate(model, memory_loader, test_loader, device,
                 k=200, temperature=0.1, num_classes=100):
    """Top-1 accuracy (%) of a weighted kNN classifier over the frozen features.

    TODO:
      * build the memory bank: bank_feats, bank_labels = _extract_features(model, memory_loader, device)
      * for each test batch (x, y):
          - q = normalized features of x                          [B, D]
          - sim = q @ bank_feats.T                                [B, M]
          - take top-k per row -> (sim_w, idx); neighbour_labels = bank_labels[idx]   [B, k]
          - weights = exp(sim_w / temperature)                    (closer -> larger vote)
          - accumulate weighted votes per class ([B, num_classes], e.g. scatter_add_)
          - prediction = argmax over classes; count correct
      * return 100 * correct / total.
    """
    raise NotImplementedError("TODO: implement weighted kNN accuracy")
