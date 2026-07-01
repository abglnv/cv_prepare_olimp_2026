
import torch
import torch.nn as nn

try:
    from train import NUM_CLASSES
except Exception:
    NUM_CLASSES = 100

DEPTHS = {"resnet18": (2, 2, 6, 2), "resnet34": (3, 3, 9, 3)}
WIDTHS = (96, 192, 384, 768)        # narrow residual stream; param-matched ~16M
EXPAND = 4
KERNEL = 3                          # 3x3 (no large-kernel step in this tour)


class LayerNorm2d(nn.Module):
    """LayerNorm over the channel dim of an [N, C, H, W] tensor (channels-first). PROVIDED."""

    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class Block(nn.Module):
    """Micro-design inverted block: 1x1 -> d3x3 -> LN -> GELU -> 1x1, identity residual."""

    def __init__(self, dim):
        super().__init__()
        # hidden = EXPAND * dim
        # TODO: build
        raise NotImplementedError("TODO: build the ConvNeXt micro block")

    def forward(self, x):
        # TODO:
        raise NotImplementedError("TODO: Block.forward")


class Net(nn.Module):
    def __init__(self, depths, widths=WIDTHS, num_classes=NUM_CLASSES):
        super().__init__()
        self.stem = nn.Sequential(                            # 32x32 CIFAR stem: 3x3 s1 (no downsample) -> 32
            nn.Conv2d(3, widths[0], 3, stride=1, padding=1),
            LayerNorm2d(widths[0]),
        )
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for i, (dim, n) in enumerate(zip(widths, depths)):
            if i > 0:                                          # separate downsampling before stages 2..4
                self.downsamples.append(nn.Sequential(
                    LayerNorm2d(widths[i - 1]),
                    nn.Conv2d(widths[i - 1], dim, 2, stride=2),
                ))
            self.stages.append(nn.Sequential(*[Block(dim) for _ in range(n)]))
        self.norm = LayerNorm2d(widths[-1])                   # final norm before the head
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(widths[-1], num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.stages[0](x)
        for ds, stage in zip(self.downsamples, self.stages[1:]):
            x = stage(ds(x))
        x = self.norm(x)
        x = torch.flatten(self.avgpool(x), 1)
        return self.fc(x)


def model_micro(size="resnet18", num_classes=None):
    """Inverted + micro design (GELU, one norm/act, BN->LN, separate downsampling)."""
    if num_classes is None:
        try:
            from train import NUM_CLASSES as num_classes
        except Exception:
            num_classes = 100
    assert size in DEPTHS, f"size must be one of {set(DEPTHS)}"
    return Net(DEPTHS[size], WIDTHS, num_classes)


if __name__ == "__main__":
    for s in ("resnet18", "resnet34"):
        m = model_micro(s)
        p = sum(x.numel() for x in m.parameters()) / 1e6
        y = m(torch.zeros(2, 3, 32, 32))
        print(f"{s:9s}: {p:5.2f}M params, out {tuple(y.shape)}")
