"""
SimCLR data pipeline for CIFAR-100 @ 32x32.   ***TASK: fill in the TODOs.***

Two things live here:
  * the stochastic augmentation t ~ T that turns one image into a random view, and a
    `TwoCropTransform` that applies it twice to make a positive pair (x_i, x_j);
  * dataloaders: a contrastive loader (pairs, for pretraining) and two eval loaders
    (a memory/train bank and a test set, both lightly transformed) for kNN.

Augmentation policy to implement (SimCLR Sec 3 / Appendix A, adapted to 32x32):
  random resized crop + horizontal flip  ->  color jitter (p=0.8)  ->  grayscale (p=0.2).
Gaussian blur is dropped -- it barely helps on 32x32 images.

CIFAR-100 is read from the (already cached) HuggingFace `uoft-cs/cifar100`.
"""
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
from datasets import load_dataset

# per-channel CIFAR-100 statistics
MEAN = (0.5071, 0.4865, 0.4409)
STD = (0.2673, 0.2564, 0.2762)
IMG_SIZE = 32


def simclr_transform(size=IMG_SIZE, s=0.5):
    """The stochastic augmentation t ~ T: one PIL image -> one random augmented view.

    TODO: return a torchvision transform (T.Compose([...])) that applies, in order:
      1. T.RandomResizedCrop(size, scale=(0.2, 1.0))          (random crop + resize)
      2. T.RandomHorizontalFlip()
      3. color distortion: T.RandomApply([T.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)], p=0.8)
                           then T.RandomGrayscale(p=0.2)
      4. T.ToTensor()
      5. T.Normalize(MEAN, STD)
    """
    color_jitter = T.ColorJitter(0.8*s,0.8*s,0.8*s,0.2*s)
    return T.Compose([
        T.RandomResizedCrop(size, scale=(0.2,1.0)),
        T.RandomHorizontalFlip(),
        T.RandomApply([color_jitter], p=0.8),
        T.RandomGrayscale(p=0.2),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])


def eval_transform(size=IMG_SIZE):
    """Deterministic transform for kNN (no random augmentation): ToTensor + Normalize."""
    return T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])


class TwoCropTransform:
    """Apply the SimCLR augmentation twice -> a positive pair (view1, view2)."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        # TODO: return two independent augmentations of the same image: (t(x), t'(x)).
        return self.base_transform(x), self.base_transform(x)


class _HFImages(Dataset):
    """Wrap a HuggingFace CIFAR-100 split; apply `transform` to the PIL image."""

    def __init__(self, hf_split, transform):
        self.hf = hf_split
        self.transform = transform

    def __len__(self):
        return len(self.hf)

    def __getitem__(self, i):
        ex = self.hf[int(i)]
        return self.transform(ex["img"]), ex["fine_label"]


def make_dataloaders(batch_size, num_workers=8, s=0.5):
    """Return (contrastive_loader, memory_loader, test_loader).

    contrastive_loader yields ((view1, view2), labels) over the train split -- labels
    are unused during pretraining. memory_loader (train) and test_loader (test) use the
    deterministic transform and yield (image, label) for kNN evaluation.

    TODO:
      * load CIFAR-100: ds = load_dataset("uoft-cs/cifar100"); train/test = ds["train"], ds["test"]
      * wrap with _HFImages: contrastive uses TwoCropTransform(simclr_transform(s=s));
        memory (train) and test use eval_transform().
      * build three DataLoaders. The contrastive loader must shuffle=True, drop_last=True;
        the two eval loaders shuffle=False. (pin_memory=True, num_workers as given.)
    """
    df = load_dataset("uoft-cs/cifar100")
    train, test = df["train"], df["test"]

    contrastive_set = _HFImages(train, TwoCropTransform(simclr_transform(s=s)))
    memory_set      = _HFImages(train, eval_transform())
    test_set        = _HFImages(test,  eval_transform())

    contrastive_loader = DataLoader(contrastive_set, batch_size=batch_size, shuffle=True,
      drop_last=True, num_workers=num_workers, pin_memory=True)
    memory_loader = DataLoader(memory_set, batch_size=batch_size, shuffle=False,
      num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
      num_workers=num_workers, pin_memory=True)
    
    return contrastive_loader, memory_loader, test_loader