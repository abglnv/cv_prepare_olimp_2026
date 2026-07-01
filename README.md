# cv_prepare_olimp_2026 — computer-vision exercises

Hands-on tasks on **CIFAR-100 @ 32×32**. You fill in the `# TODO` stubs and run everything
from [`playground.ipynb`](playground.ipynb). There are no solutions in this repo — implement
the pieces yourself.

## Exercises

| folder | topic | you implement |
|---|---|---|
| `cls/task/` | the ConvNeXt "modernizing a ResNet" tour | each step's block: `BasicBlock`, inverted bottleneck, ConvNeXt micro block, ViT attention |
| `playground.ipynb` | data augmentation | `mixup`, `cutmix`, `random_erase` (inline) |
| `simclr/task/` | SimCLR self-supervised pretraining | augmentations, projection head, NT-Xent loss, kNN eval |
| `clip/task/` | CLIP with class embeddings | dataloaders, class encoder, symmetric loss, zero-shot eval |

`cls/train.py` (the fixed training harness) and `cls/graph.py` (plots) are provided — do not
edit them. `simclr/` and `clip/` are self-contained packages (their own
`dataset/model/loss/eval/train`), run from `playground.ipynb`.

## Run

Open `playground.ipynb` (a CUDA GPU is expected) and work top to bottom. The "Testing"
section trains each subsystem once you've filled in its TODOs — SimCLR reports a kNN
monitor, CLIP a zero-shot top-1 (chance is 1% on CIFAR-100).
