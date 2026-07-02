"""
kNN evaluation of the learned representation (no labels used during training).
                                                        ***TASK: fill in the TODOs.***

Build a memory bank of L2-normalized features h = f(x) over the train set, then for each
test image retrieve its k nearest neighbours by cosine similarity and predict the label by
a temperature-weighted vote. This is the standard weighted-kNN monitor (Wu et al., 2018).
"""
import torch
import torch.nn.functional as F

@torch.no_grad()
def _extract_features(model, loader, device):
    """L2-normalized features and labels for every image in `loader`.

    TODO: set model.eval(); for each (x, y) run h = model.features(x) on `device`,
    L2-normalize h (dim=1), collect h and y; return (feats [M, D], labels [M]).
    """
    model.eval()
    feats, labels = [], []
    for x, y in loader: 
        x = x.to(device, non_blocking=True)
        h = model.features(x)
        h = F.normalize(h, dim=1)
        feats.append(h)
        labels.append(y.to(device))
    return torch.cat(feats, dim=0), torch.cat(labels, dim=0)


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
    bank_feats, bank_labels = _extract_features(model, memory_loader, device)
    correct, total = 0, 0

    for x, y in test_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device)

        q = F.normalize(model.features(x), dim=1)
        sim = q @ bank_feats.T 

        sim_w, idx = sim.topk(k, dim=1)     
        neighbour_labels = bank_labels[idx]   
        weights = torch.exp(sim_w / temperature)  

        votes = torch.zeros(x.size(0), num_classes, device=device)
        votes.scatter_add_(1, neighbour_labels, weights) 

        pred = votes.argmax(dim=1)        
        correct += (pred == y).sum().item()
        total += y.size(0)

    return 100.0 * correct / total    
