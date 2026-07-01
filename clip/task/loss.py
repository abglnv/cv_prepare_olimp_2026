"""
CLIP's symmetric contrastive loss (Radford et al., 2021, Figure 3), adapted to labels.
                                                        ***TASK: fill in the TODO.***

In vanilla CLIP each image pairs with exactly one text, so the target is the diagonal
(image i matches text i). Here the "text" is a class label, and a batch can contain several
images of the same class -- which all share one class embedding. So the positive of image i
is every column j with the same label.
"""
import torch.nn as nn


class CLIPLoss(nn.Module):
    def forward(self, image_emb, class_emb, labels, logit_scale):
        """image_emb, class_emb: [N, D]; labels: [N]; logit_scale: scalar param.

        TODO:

        This same target works for both batch regimes: with unique-class batches
        (dataset one_per_class=True) `same` is the identity, so it becomes the diagonal.
        """
        raise NotImplementedError("TODO: implement the CLIP symmetric loss")
