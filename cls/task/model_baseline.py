import torch
import torch.nn as nn

try:
    from train import NUM_CLASSES
except Exception:
    NUM_CLASSES = 100

DEPTHS = {"resnet18": (2, 2, 2, 2), "resnet34": (3, 4, 6, 3)}   # plain ResNet depths
WIDTHS = (64, 128, 256, 512)


def conv3x3(cin, cout, stride=1):
    return nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    """Two 3x3 convs with a residual connection (the ResNet-18/34 block)."""
    expansion = 1

    def __init__(self, cin, cout, stride=1):
        super().__init__()
        # TODO: build the block:
        raise NotImplementedError("TODO: build BasicBlock")

    def forward(self, x):
        # TODO: 
        raise NotImplementedError("TODO: BasicBlock.forward")


class Net(nn.Module):
    def __init__(self, depths, widths=WIDTHS, num_classes=NUM_CLASSES):
        super().__init__()
        self.inplanes = widths[0]
        self.stem = nn.Sequential(                          # 32x32 CIFAR stem: 3x3 s1 (no downsample) -> 32
            nn.Conv2d(3, widths[0], 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(widths[0]),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._stage(widths[0], depths[0], stride=1)
        self.layer2 = self._stage(widths[1], depths[1], stride=2)
        self.layer3 = self._stage(widths[2], depths[2], stride=2)
        self.layer4 = self._stage(widths[3], depths[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(widths[3], num_classes)
        self._init_weights()

    def _stage(self, planes, blocks, stride):
        layers = [BasicBlock(self.inplanes, planes, stride)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(planes, planes, 1))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = torch.flatten(self.avgpool(x), 1)
        return self.fc(x)


def model_baseline(size="resnet18", num_classes=None):
    """Plain ResNet-18/34 baseline (32x32 CIFAR /1 stem)."""
    if num_classes is None:                      # resolve live (after init_train set it)
        try:
            from train import NUM_CLASSES as num_classes
        except Exception:
            num_classes = 100
    assert size in DEPTHS, f"size must be one of {set(DEPTHS)}"
    return Net(DEPTHS[size], WIDTHS, num_classes)


if __name__ == "__main__":
    for s in ("resnet18", "resnet34"):
        m = model_baseline(s)
        p = sum(x.numel() for x in m.parameters()) / 1e6
        y = m(torch.zeros(2, 3, 32, 32))
        print(f"{s:9s}: {p:5.2f}M params, out {tuple(y.shape)}")
