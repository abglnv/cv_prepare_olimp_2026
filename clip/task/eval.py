"""
Zero-shot classification, the CLIP evaluation protocol (Sec 3.1.2).
                                                        ***TASK: fill in the TODO.***

Encode every class label into a prototype (CLIP's cached "zero-shot classifier"),
L2-normalize it, and classify each test image by the highest cosine similarity to a
prototype -- a multinomial logistic regression with L2-normalized inputs and weights,
no bias (argmax of the cosine similarities).
"""
import torch


@torch.no_grad()
def zeroshot_evaluate(model, test_loader, num_classes, device):
    """Top-1 accuracy (%) of the zero-shot (cosine-to-class-prototype) classifier.

    TODO:
    """
    raise NotImplementedError("TODO: implement zero-shot accuracy")
